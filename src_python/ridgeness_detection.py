import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.feature.util import _prepare_grayscale_input_2D
from common import img_gaussian_smooth
from utils import vap_imnew, vap_imshow, vap_imshow_set, vap_imsave, vap_impop_up
from utils import vap_time_decorator

#compute approximate derivative
def compute_derivatives(img, mode='constant', cval=0):
	"""Compute derivatives in x and y direction using the Sobel operator.
	:param img: ndarray, Input image.
	:param mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional. How to handle values outside the image borders.
	:param cval: Used in conjunction with mode 'constant', the value outside the image boundaries.
	:return: (im_x, im_y): (Derivative in x-direction, Derivative in y-direction)
	"""
	# im_y = ndi.sobel(img, axis=0, mode=mode, cval=cval) #horizontal derivative
	# im_x = ndi.sobel(img, axis=1, mode=mode, cval=cval) #vertical derivative
	im_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3) 
	im_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)
	# im_y, im_x = np.gradient(img)

	return im_x, im_y

#compute tensor 2x2 structure
def compute_tensor_struct_1(img, im_x, im_y, path, out_names, img_shw_step):
	#==============================================
    # configuration parameter
	H, W = img.shape
	SIGMA = 0
	MODE = 'constant'
	CVAL = 0

	#==============================================
	grad_vector = np.concatenate((np.expand_dims(im_x, -1), np.expand_dims(im_y, -1)), -1).reshape([H, W, 2, 1])

	Axx = ndi.gaussian_filter(im_x*im_x, [SIGMA, SIGMA], mode=MODE, cval=CVAL)
	Axy = ndi.gaussian_filter(im_x*im_y, [SIGMA, SIGMA], mode=MODE, cval=CVAL)
	Ayy = ndi.gaussian_filter(im_y*im_y, [SIGMA, SIGMA], mode=MODE, cval=CVAL)
	if img_shw_step:
		vap_imshow_set(1, 3, 1, Axx, "Axx")
		vap_imshow_set(1, 3, 2, Axy, "Axy")
		vap_imshow_set(1, 3, 3, Ayy, "Ayy")
		vap_imsave(path, out_names+['04Axx_Axy_Ayy'])

	Axx = np.expand_dims(Axx, -1)
	Axy = np.expand_dims(Axy, -1)
	Ayy = np.expand_dims(Ayy, -1)
	struct_tensor = np.reshape(np.concatenate((Axx,Axy,Axy,Ayy),-1), [H , W, 2, 2])

	return struct_tensor, grad_vector

def compute_tensor_struct_2(img, im_x, im_y, path, out_names, img_shw_step): 
	#==============================================
    # configuration parameter
	H, W = img.shape
	SIGMA = 0.5
	MODE = 'constant'
	CVAL = 0

	#==============================================
	grad_vector = np.concatenate((np.expand_dims(im_x, -1), np.expand_dims(im_y, -1)), -1).reshape([H, W, 2, 1])
	grad_vector_t = np.transpose(grad_vector, (0, 1, 3, 2))
	struct_tensor = np.matmul(grad_vector,grad_vector_t)

	return struct_tensor, grad_vector

def divergence(f):
	"""
	Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
	:param f: List of ndarrays, where every item of the list is one dimension of the vector field
	:return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
	"""
	num_dims = len(f)
	print(f)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

def vap_ridgeness_detection(img, img_path, img_output_names, img_shw_step = False):
	#==============================================
	# Pre-Process
	img = _prepare_grayscale_input_2D(img)
	[H, W] = img.shape
	if img_shw_step:
		vap_imshow(img, "Input image")
		vap_imsave(img_path, img_output_names+['01Input_image'])
	
	#==============================================
	# Configuration setting
	SIGMA = 0.5
	MODE = 'constant'
	CVAL = 0

	#==============================================
	# Implement algorithm
	""" 
	Step 1: Gaussian smoothing 
	"""
	img = img_gaussian_smooth(img, MODE, CVAL)
	if img_shw_step:
		vap_imshow(img, "Smooth_version_of_image")
		vap_imsave(img_path, img_output_names+['02Smooth_version_of_image'])

	""" 
	Step 2: Compute derivatives 
	"""
	[im_x, im_y] = compute_derivatives(img, mode=MODE, cval=CVAL)
	if img_shw_step:
		vap_imshow_set(1, 2, 1, im_x, "X derivative")
		vap_imshow_set(1, 2, 2, im_y, "Y derivative")
		vap_imsave(img_path, img_output_names+['03X_Y_derivative'])

	""" 
	Step 3: Build structure tensor 
	"""
	struct_tensor, grad_vector = compute_tensor_struct_1(img, im_x, im_y, img_path, img_output_names, img_shw_step)
	# struct_tensor, grad_vector = compute_tensor_struct_2(img, im_x, im_y, img_path, img_output_names, img_shw_step)

	""" 
	Step 4: find dominant gradient vector 
	"""
	### Compute eigenvalue, eigenvector for each structure tensor
	[eig_val, eig_vector] = np.linalg.eig(struct_tensor)
	# eig_vector = eig_vector.transpose((0, 1, 3, 2))

	### Find dominant gradient by finding the greatest eigen value
	dominant_idx = np.argmax(eig_val, axis=-1)
	[axV, axU] = np.indices([H, W])
	dominant_vector = np.reshape(eig_vector[axV.flat, axU.flat, dominant_idx.flat], [H, W, 2])
	dominant_vector_t = np.reshape(dominant_vector, (H, W, 1, 2))
	if img_shw_step:
		idx = slice(None, None, 10)
		a = np.linspace(0, W-1, W)
		b = np.linspace(0, H-1, H)

		vap_imnew()
		plt.quiver(a[idx], b[idx], dominant_vector[idx, idx, 0], dominant_vector[idx, idx, 1], pivot='mid')
		plt.title("Dominate vector", loc='center')
		vap_imsave(img_path, img_output_names+['05dominant_vector'])

	### Dominant vector with direction
	sign_mask = np.matmul(dominant_vector_t, grad_vector).reshape(H, W)
	dominant_vector[sign_mask < 0] *= -1
	dominant_vector[sign_mask == 0] *= 0
	
	if img_shw_step:
		vap_imnew()
		plt.quiver(np.arange(W), np.arange(H), dominant_vector[:,:,0], dominant_vector[:,:,1], pivot='mid')
		plt.title("Dominate vector with direction", loc='center')
		vap_imsave(img_path, img_output_names+['06dominant_vector_with_direction'])

	""" 
	Step 5: Compute divergence 
	"""
	vector_u = dominant_vector[:, :, 0]
	vector_v = dominant_vector[:, :, 1]
	ridge_img = -divergence([vector_v, vector_u])
	ridge_img[ridge_img < 0.25] = 0

	if img_shw_step:
		vap_imshow(ridge_img, "Original Ridgeness Image")
		vap_imsave(img_path, img_output_names+['07Original_Ridgeness_Image'])

	""" 
	Step 6: Discard ridge point with large horizontal component 
	"""
	theta = np.abs(np.arctan2(vector_v, vector_u))
	mask = np.logical_and(theta > np.pi*0.4, theta < np.pi*0.6)
	ridge_img[mask] = 0

	if img_shw_step:
		vap_imshow(ridge_img, "Theta Filter Ridgeness image")
		vap_imsave(img_path, img_output_names+['08Theta_Filter_Ridgeness_image'])

	""" 
	Step 7: Confident Filter image 
	"""
	ridge_img *= (1 - np.exp(-np.power(eig_val[:,:,0] - eig_val[:,:,1], 2) / 0.001))
	ridge_img[ridge_img < 0.5] = 0
	
	if img_shw_step:
		vap_imshow(ridge_img, "Confident Filter Image")
		vap_imsave(img_path, img_output_names+['09Confident_Filter_Image'])

	return np.uint8(ridge_img*128)