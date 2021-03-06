# # -*- coding: utf-8 -*-
# """ridge lane detection chinhnt24.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1SlXurPh_LzBJHg_QzxsoAH7-mYqAlCjv
# """

# import cv2
# import os

# import numpy as np
# import matplotlib.pyplot as plt

# from skimage import io
# from skimage.feature.util import _prepare_grayscale_input_2D
# from scipy import ndimage as ndi
# import time
# # from google.colab.patches import cv2_imshow
# # from google.colab import drive
# # drive.mount('/content/drive')

# def image_show_only(show_img, img, title):
#     if show_img: 
#         io.imshow(img)
#         plt.title(title, loc = 'center')
#         io.show()
#     return

# def image_show_set(show_img, show_dx, show_dy, pos, img, title):
#     if show_img:
#         plt.subplot(show_dx, show_dy, pos)
#         io.imshow(img)
#         plt.title(title, loc = 'center')
#     return

# def image_show_popup(show_img = False):
#     if show_img:
#         io.show()    
#     return
    
# def compute_derivatives(image, mode='constant', cval=0):
#     """Compute derivatives in x and y direction using the Sobel operator.

#     :param image: ndarray, Input image.
#     :param mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional. How to handle values outside the image borders.
#     :param cval: Used in conjunction with mode 'constant', the value outside the image boundaries.
#     :return: (imx, imy): (Derivative in x-direction, Derivative in y-direction)
#     """
#     imy = ndi.sobel(image, axis=0, mode=mode, cval=cval)
#     imx = ndi.sobel(image, axis=1, mode=mode, cval=cval)
#     return imx, imy

# def divergence(f):
#     """
#     Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
#     :param f: List of ndarrays, where every item of the list is one dimension of the vector field
#     :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
#     """
#     num_dims = len(f)
#     return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

# def lane_marking_detection(image, title, show_step_result=False):
#     # =============================================
#     # show input image
#     image = _prepare_grayscale_input_2D(image) # check image 2-dim and convert to float
#     row, col = image.shape
#     # image=image[(int)(row/2):row,:]
#     # row, col = image.shape
#     print("row col ",row,col)

#     image_show_set(show_step_result, 1, 2, 1, image, "Input Image")

#     # =============================================
#     # configure setting
#     SIGMA = 1.5
#     MODE = 'constant'
#     CVAL = 0
#     # default truncate = 4.0 
#     # => window size w = 2*int(truncate*sigma + 0.5) + 1 = 13

#     # =============================================
#     # Gaussian smooth. hard cording for now
#     image = ndi.gaussian_filter(image, [SIGMA, SIGMA], mode=MODE, cval=CVAL)
#     # image = ndi.gaussian_filter(image, [1, 0], mode=MODE, cval=CVAL)
#     # image[:108, :] = ndi.gaussian_filter(image[:108, :], [0, 4], mode=MODE, cval=CVAL)
#     # image[108:125, :] = ndi.gaussian_filter(image[108:125, :], [0, 3], mode=MODE, cval=CVAL)
#     # image[125:150, :] = ndi.gaussian_filter(image[125:150, :], [0, 2], mode=MODE, cval=CVAL)
#     # image[150:, :] = ndi.gaussian_filter(image[150:, :], [0, 1], mode=MODE, cval=CVAL)
#     image_show_set(show_step_result, 1, 2, 2, image, "Gaussian Smooth Image")
#     image_show_popup(show_step_result)
    

#     # =============================================
#     # Compute the structure tensor field based on gradient vector field
#     imx, imy = compute_derivatives(image, mode=MODE, cval=CVAL)
#     # image_show_set(show_step_result, 1, 2, 1, imx, "X derivative")
#     # image_show_set(show_step_result, 1, 2, 2, imy, "Y derivative")
#     # image_show_popup(show_step_result)
#     image_show_only(show_step_result,imx,"X derivative")
#     image_show_only(show_step_result,imy,"Y derivative")
#     # image_show_only(show_step_result,imx*imy,"imx*imy")


#     imx = np.expand_dims(imx, -1)
#     imy = np.expand_dims(imy, -1)
#     gradient_vector = np.concatenate((imx, imy), axis=2)
#     gradient_vector=np.reshape(gradient_vector,(row,col,2,1))
#     gradient_vector_t=np.reshape(gradient_vector,(row,col,1,2))

#     #create w array
#     # imx = np.expand_dims(imx, -1)
#     # imy = np.expand_dims(imy, -1)
#     # w_array = np.concatenate((imx, imy), axis=2)
#     # w_array = np.reshape(w_array, [row, col, 2, 1])
#     # w_array_t =np.transpose(w_array,(0,1,3,2))

#     # w_array= ndi.gaussian_filter(w_array, [SIGMA, SIGMA, SIGMA, SIGMA], mode=MODE, cval=CVAL)
#     # w_array_t=ndi.gaussian_filter(w_array_t, [SIGMA, SIGMA, SIGMA, SIGMA], mode=MODE, cval=CVAL)
    
#     # print("w_array_shape",w_array.shape)
#     # print("w_array_tshape",w_array_t.shape)

#     # structure tensor
#     # Axx = ndi.gaussian_filter(imx * imx, [SIGMA, SIGMA], mode=MODE, cval=CVAL)
#     # Axy = ndi.gaussian_filter(imx * imy, [SIGMA, SIGMA], mode=MODE, cval=CVAL)
#     # Ayy = ndi.gaussian_filter(imy * imy, [SIGMA, SIGMA], mode=MODE, cval=CVAL)

#     # # image_show_only(show_step_result,Axx,"Axx")
#     # # image_show_only(show_step_result,Axy,"Axy")
#     # # image_show_only(show_step_result,Ayy,"Ayy")
    

#     # # Create a local 2x2 structure tensor for each pixel, shape should be [row, col, 2, 2]
#     # Axx = np.expand_dims(Axx, -1)
#     # Axy = np.expand_dims(Axy, -1)
#     # Ayy = np.expand_dims(Ayy, -1)
#     # structure_tensor = np.reshape(np.concatenate((Axx, Axy, Axy, Ayy), -1), [row, col, 2, 2])
#     structure_tensor=np.empty((row,col,2,2))
#     for x in range(len(structure_tensor)):
#         for y in range(len(structure_tensor[x])):
#             structure_tensor[x,y]=gradient_vector[x,y].dot(gradient_vector_t[x,y])
#             # structure_tensor[x,y]=ndi.gaussian_filter(structure_tensor[x,y], [SIGMA, SIGMA], mode=MODE, cval=CVAL)

#     structure_tensor=ndi.gaussian_filter(structure_tensor, [SIGMA, SIGMA, 0.1, 0.1], mode=MODE, cval=CVAL)
    
#     # structure_tensor = w_array*w_array_t
#     # print("Structure tensor",structure_tensor.shape)
#     # # for x in structure_tensor: structure_tensor = w_array*w_array_t
#     print("Structure tensor",structure_tensor.shape)
#     # for x in structure_tensor:
#     #   for y in x:
#     #     y=ndi.gaussian_filter(y,[SIGMA, SIGMA],mode=MODE,cval=CVAL)
    
#     # structure_tensor=ndi.gaussian_filter(structure_tensor, [SIGMA, SIGMA, SIGMA, SIGMA], mode=MODE, cval=CVAL)

#     # Compute eigen vector and eigen value for each structure tensor
#     eigvalue, eigvector = np.linalg.eig(structure_tensor)
#     print("eigvector.shape",eigvector.shape,eigvalue.shape)
#     print("eigvalue ",eigvalue)
#     eigvector = eigvector.transpose((0, 1, 3, 2))

#     # Find dominant gradient by finding the greatest eigen value
#     dominant_index = np.argmax(eigvalue, axis=-1)
#     print("dominant_index shape",dominant_index.shape)
#     print("dominant index ",dominant_index)
#     ax_v, ax_u = np.indices([row, col])
#     print("ax_v",ax_v.shape)
#     print("ax_u",ax_u.shape)
    
#     dominant_vector = np.reshape(eigvector[ax_v.flat, ax_u.flat, dominant_index.flat], [row, col, 2])
#     # dominant_vector = eigvector[ax_v.flat, ax_u.flat, dominant_index.flat]
    
  
#     # if show_step_result:
#     #     plt.quiver(np.arange(col), np.arange(row), dominant_vector[:, :, 0], dominant_vector[:, :, 1], pivot='mid')
#     #     plt.title("Dominate vector", loc='center')
#     #     plt.show()

  
   
#     print("gradient vector shape",gradient_vector.shape)
#     dominant_vector_t=np.reshape(dominant_vector,(row,col,1,2))
#     print("dominant vector ",dominant_vector_t.shape)

#     sign_mask = np.empty((row,col,1))
    
#     for x in range(len(sign_mask)):
#         for y in range(len(sign_mask[x])):
#             sign_mask[x,y]=dominant_vector_t[x,y].dot(gradient_vector[x,y])
#     # sign_mask = np.tensordot(dominant_vector_t,gradient_vector,axes=([2], [2]))
#     sign_mask=np.reshape(sign_mask,(row,col))
#     print("sign_mask .shape",sign_mask.shape,dominant_vector_t.shape)
#     # sign_mask = np.sum(sign_mask, axis=-1)

#     dominant_vector[sign_mask < 0] *= -1
#     # dominant_vector[sign_mask > 0] *= 1
#     dominant_vector[sign_mask == 0] *= 0
#     dominant_vector[sign_mask > 0] *= 1
    
    
#     print("sign_mask .shape",sign_mask.shape, dominant_vector.shape)
    

#     # if show_step_result:
#     #     plt.quiver(np.arange(col), np.arange(row), dominant_vector[:, :, 0], dominant_vector[:, :, 1], pivot='mid')
#     #     plt.title("Dominate vector with direction", loc='center')
#     #     plt.show()

#     vectors_u = dominant_vector[:, :, 0]
#     vectors_v = dominant_vector[:, :, 1]
#     ridges_image = -divergence((vectors_v,vectors_u))
#     ridges_image[ridges_image < 0.25] = 0

#     image_show_only(show_step_result, ridges_image, "Ori Ridge Image, {}".format(title))

#     # =============================================
#     # Discard ridge point with large horizontal component
#     theta = np.abs(np.arctan2(vectors_v, vectors_u))

#     # According to the paper, the boundary should be 0.25 and 0.75, however, too many lane ridge points were filtered
#     # out. Here I changed the boundary to 0.4 nad 0.6
#     mask = np.logical_and(theta > np.pi * 0.4, theta < np.pi * 0.6)
#     ridges_image[mask] = 0
#     image_show_only(show_step_result, ridges_image, "Theta Filter Ridges imaga")

#     # =============================================
#     # Confidence filter
#     ridges_image *= (1 - np.exp(-np.power(eigvalue[:, :, 0] - eigvalue[:, :, 1], 2) / 0.001))
#     ridges_image[ridges_image < 0.25] = 0
#     image_show_only(show_step_result, ridges_image, "Confidence Filter Image, {}".format(title))

#     # image_show_set(True, 1, 2, 1, image, "input image")
#     # image_show_set(True, 1, 2, 2, ridges_image, "ridges_image")
#     # image_show_popup(True)
#     return ridges_image

# def test_image():
#     # =============================================
#     # Read image 
#     image_path = '/home/chinhnt24/rawvideo_30_fov_006601.jpg'
#     image = cv2.imread(image_path, 0)

#     # =============================================
#     # Resize image
#     scale = 1.0
#     image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    
#     # =============================================
#     ridge = lane_marking_detection(image, 'ridge', show_step_result=True)
# def get_ridge_image(image):
#     scale = 1.0
#     image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    
#     # =============================================
#     ridge = lane_marking_detection(image, 'ridge', show_step_result=True)
#     return ridge
# def sortKeyFunc(s):
#     return int(os.path.basename(s)[:-4])

# def calculate_log_chromaticity_pixel_values(image):
#     row,col,_=image.shape
#     log_chromaticity_matrix=np.empty((row,col,2))
#     log_chromaticity_matrix[:,:,0]=np.log(image[:,:,0]/(image[:,:,1].astype(float)+1e-100)+1e-100)
#     log_chromaticity_matrix[:,:,1]=np.log(image[:,:,2]/(image[:,:,1].astype(float)+1e-100)+1e-100)
#     return log_chromaticity_matrix
# def create_multiple_sample(image,slope):
#     row,col,_=image.shape
#     new_sample=np.empty((row,col,3),np.uint8)
#     for i in range(row):
#         for j in range(col):
#             value=image[i,j] #"BGR"
#             new_sample[i,j][0]=max(0,mtest_video_with_illuminant_invariancein(255,value[0]+slope))
#             new_sample[i,j][1]=max(0,min(255,value[1]+slope))
#             new_sample[i,j][2]=max(0,min(255,value[2]+slope))

#     return new_sample
# def calculate_grayscale_project_in_theta_line(log_matrix,theta):
#     row, col,_ =log_matrix.shape
#     output=np.zeros((row,col))
#     if theta == 0:
#         output[:,:]=log_matrix[:,:,1]
#     else:
#         d=-1/np.tan(theta)
#         output[:,:]=np.sqrt(((log_matrix[:,:,0]-d*log_matrix[:,:,1])/(np.tan(theta)-d))**2+ (np.tan(theta)*(log_matrix[:,:,0]-d*log_matrix[:,:,1])/(np.tan(theta)-d))**2)
#     # for i in range(row):
#     #     for j in range(col):
#     #         if theta==0:
#     #              output[i,j]=log_chromaticity_matrix[i,j,1]
#     #         else:
#     #             val=log_chromaticity_matrix[i,j]
#     #             d=-1/np.tan(theta)
#     #             b=val[0]-d*val[1]
#     #             project_x=b/(np.tan(theta)-d)

#     #             project_y=d*project_x+b
#     #             projected_value=np.exp(np.sqrt(project_x*project_x+project_y*project_y))#
#     #             output[i,j]=projected_value
#     return output
# def calculate_historgram(gray):
#     row,col=gray.shape
#     bin_width=3.5*np.power(row*col,-1/3)*np.std(gray)
#     # bin_width=1
#     print("bin_width",bin_width)
#     curr=0
#     bins=[]
#     while curr<20:
#         bins.append(curr)
#         curr=curr+bin_width
#     hist,bins=np.histogram(gray,bins)
#     plt.hist(gray,bins)
#     plt.title("histogram")
#     plt.show()
#     print("bins shape",bins.shape)
#     print("hist",hist)
#     print("bins",bins)
# def calculate_theta_angle(rgb_image):
#     # out = cv2.VideoWriter('/home/chinhnt24/out_test_ill_invariance_color.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (960,540),True)
#     log_chromaticity_matrix=calculate_log_chromaticity_pixel_values(rgb_image)
#     for i in range(360):
#         output=calculate_grayscale_project_in_theta_line(log_chromaticity_matrix,np.pi/180*i)
#         # cv2.imshow("sample",rbg_image)
#         # calculate_historgram(output)
#         # output_=(output*10)
#         # output_=cv2.cvtColor(output_,cv2.COLOR_GRAY2BGR)
#         # out.write(output_)
#         cv2.imshow("output ",output)
#         cv2.waitKey(0)
#     # out.release()
# def calculate_road_classifier():
#     return None
# def road_detect_based_on_illuminant_invariance(rgb_image):
#     theta=calculate_theta_angle(rgb_image)
#     # calculate_road_classifier()
# def test_video_withoutput(): 
    
#     cap = cv2.VideoCapture("/home/chinhnt24/test_2.mp4")
#     out = cv2.VideoWriter('/home/chinhnt24/out_test_2.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (2560,720),True)
#     print(cap.isOpened())
#     count=0
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if (frame is None):
#           break;
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         ridge=get_ridge_image(gray)
#         ridge=ridge*127.5
#         ridge=np.array(ridge).astype(np.uint8)
#         ridge= cv2.cvtColor(ridge,cv2.COLOR_GRAY2BGR)
#         vis = np.concatenate((frame, ridge), axis=1)
#         print("process ",count)
#         # print("vis shape",vis.shape,gray.shape,ridge.shape)
#         count=count+1
#         # cv2_imshow(vis)
#         # vis= cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)
#         out.write(vis)
#         print(vis.shape, gray.shape)
#         # if (count==25):
#         #   break;

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# def test_video2(): 
    
#     cap = cv2.VideoCapture("/home/chinhnt24/test_2.mp4")
#     print(cap.isOpened())
#     count=0
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if (frame is None):
#           break;
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         ridge=get_ridge_image(gray)
#         ridge=ridge*127.5
#         ridge=np.array(ridge).astype(np.uint8)
#         # ridge= cv2.cvtColor(ridge,cv2.COLOR_GRAY2BGR)
#         vis = np.concatenate((gray, ridge), axis=1)
#         print("process ",count)
#         # print("vis shape",vis.shape,gray.shape,ridge.shape)
#         count=count+1
#         # image_show_only(True,vis,"img")
#         cv2.imshow("imshow",vis)
#         cv2.waitKey(0)
#         # vis= cv2.cvtColor(vis,cv2.COLOR_GRAY2BGR)

#         print(vis.shape, gray.shape)
#         if (count==1):
#           break;

#     cap.release()
#     cv2.destroyAllWindows()
# def test_video_with_illuminant_invariance():

#     cap = cv2.VideoCapture("/home/chinhnt24/test_2.mp4")
#     count=0
#     rgb_list=[]
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if (frame is None):
#           break;
#         rgb_list.append(frame)
#         count=count+1
#         if (count==1):
#           break;
#     cap.release()
#     road_detect_based_on_illuminant_invariance(rgb_list)
#     cv2.destroyAllWindows()
# def test_image_with_illuminant_invariance():
#     start=time.time()
#     image_path = '/home/chinhnt24/rawvideo_30_fov_006601.jpg'
#     image = cv2.imread(image_path,1)
#     scale = 0.5
#     image = cv2.resize(image, (0,0), fx=scale, fy=scale)
#     road_detect_based_on_illuminant_invariance(image)
#     end=time.time()
#     print("processed ",end-start)
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage as ndi
from skimage.feature.util import _prepare_grayscale_input_2D
from common import img_gaussian_smooth
from utils import vap_imnew, vap_imshow, vap_imshow_set, vap_imsave
from utils import vap_time_decorator

#compute approximate derivative
def compute_derivatives(image, mode='constant', cval=0):
	"""Compute derivatives in x and y direction using the Sobel operator.
	:param image: ndarray, Input image.
	:param mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional. How to handle values outside the image borders.
	:param cval: Used in conjunction with mode 'constant', the value outside the image boundaries.
	:return: (imx, imy): (Derivative in x-direction, Derivative in y-direction)
	"""
	imy = ndi.sobel(image, axis=0, mode=mode, cval=cval)
	imx = ndi.sobel(image, axis=1, mode=mode, cval=cval)
	return imx, imy

def divergence(f):
	"""
	Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
	:param f: List of ndarrays, where every item of the list is one dimension of the vector field
	:return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
	"""
	num_dims = len(f)
	print("num dims",num_dims)
	return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])
def construct_A_b(L_point,R_point,Ev,camera_pitch):
	"""
	equation: Ax=b
	construct A, b for find x
	"""
	A=[]
	b=[]
	for point in L_point:
		v_L_=point[0]/Ev+np.tan(camera_pitch)
		left_equation=[1,-v_L_,v_L_,1/v_L_]
		A.append(left_equation)
		b.append(point[1])
	for point in R_point:
		v_R_=point[0]/Ev+np.tan(camera_pitch)
		right_equation=[1,-v_R_,v_R_,1/v_R_]
		A.append(right_equation)
		b.append(point[1])
	A=np.array(A)
	b=np.array(b)
	return A,b
def update_inliner_point_base_on_x(x,ridgeL,ridgeR,Ev,camera_pitch,error_max):
	L_point=[]
	R_point=[]
	z=0.0
	for point in ridgeL:
		v_L_=point[0]/Ev+np.tan(camera_pitch)
		error_Left=np.abs((x[0]+v_L_*x[1]-v_L_*x[2]+x[3]/v_L_)-point[1])
		# print("error_Left",error_Left)
		# if (error_Left<min_left):
		# 	min_left=error_Left
		if (error_Left<=error_max):
			L_point.append(point)
			z=z+1#/(1+error_Left)
	
	for point in ridgeR:
		v_R_=point[0]/Ev+np.tan(camera_pitch)
		error_Right=np.abs((x[0]+v_R_*x[1]-v_R_*x[2]+x[3]/v_R_)-point[1])
		# print("error_Right",error_Right)
		# if (error_Right<min_right):
		# 	min_right=error_Right
		if (error_Right<=error_max):
			R_point.append(point)
			z=z+1#/(1+error_Right)
	return L_point,R_point,z
def generate_curve_points_base_on_best_model_left(x,Ev,camera_pitch,row,col):
	L_point=[]
	for i in np.arange(int(row*0.5),row):
		v_L_=i/Ev+np.tan(camera_pitch)
		point=[i,(x[0]+v_L_*x[1]-v_L_*x[2]+x[3]/v_L_)]
		if (point[0]>0 and point[0]<row and point[1]>0 and point[1]<col):
			L_point.append(point)
	return np.array(L_point).astype(np.int)
def generate_curve_points_base_on_best_model_right(x,Ev,camera_pitch,row,col):
	R_point=[]
	for i in np.arange(int(row*0.5),row):
		v_R_=i/Ev+np.tan(camera_pitch)
		point=[i,(x[0]+v_R_*x[1]-v_R_*x[2]+x[3]/v_R_)]
		if (point[0]>0 and point[0]<row and point[1]>0 and point[1]<col):
			R_point.append(point)
	return np.array(R_point).astype(np.int)
def debug_point(L_point,R_point,row,col):
	checkridge=np.zeros((row,col,3),np.uint8)
	for point in L_point:
		checkridge[point[0],point[1],0]=0
		checkridge[point[0],point[1],1]=200
		checkridge[point[0],point[1],2]=200
	for point in R_point:
		checkridge[point[0],point[1],0]=200
		checkridge[point[0],point[1],1]=0
		checkridge[point[0],point[1],2]=200
		
	# for point in ridgeR:
	# 	result[point[0],point[1]]=200
	cv2.imshow("checkridge result",checkridge)
	cv2.waitKey(0)
def calculate_x_model(ridgeL,ridgeR,Ev,camera_pitch,error_max,row,col):
	count=0

	best_z=0
	best_x=np.zeros((4))
	count_th=30
	L_max=0
	while (best_z==0 or count<1000):
		# print("count ",count)
		count=count+1

		raiseException=True
		x=np.zeros((4))
		while raiseException:
			if (len(ridgeL)==0):
				L_num=0
				R_num=4
			if (len(ridgeR)==0):
				L_num=4
				R_num=0
			
			if (len(ridgeR)>0 and len(ridgeL)>0):
				L_num=2
				R_num=2
			

			L_index=np.random.randint(ridgeL.shape[0]-1,size=(L_num))
			R_index=np.random.randint(ridgeR.shape[0]-1,size=(R_num))
		
			L_point=ridgeL[L_index]
			R_point=ridgeR[R_index]
			# debug_point(L_point,R_point,row,col)

			A,b=construct_A_b(L_point,R_point,Ev,camera_pitch)
			raiseException=False
			try:
				# A=A+1e-3
				# x=np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
				x=np.linalg.solve(A,b)
			except:
				# print("exception")
				raiseException=True
			
		# print("jump out while")
		# Theta=np.cos(camera_pitch)/Eu*x[0]
		# L=H/(Eu*np.cos(camera_pitch))*x[1]
		# dL=H/(Eu*np.cos(camera_pitch))*x[2]
		# C0=4*np.cos(camera_pitch)**3/(Eu*H)*x[3]

		L_point, R_point, z=update_inliner_point_base_on_x(x,ridgeL,ridgeR,Ev,camera_pitch,error_max)

		if ((len(L_point)+len(R_point))>count_th):
			if (z>best_z):
				print("update best z",z)
				best_z=z
				best_x=x

			print("found good curve, try find more suitable model x")
			A,b=construct_A_b(L_point,R_point,Ev,camera_pitch)
			A=A+1e-6#*np.random.rand(1000,2)
			try:
				x=np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
				print("new x",x)
				L_point,R_point,z=update_inliner_point_base_on_x(x,ridgeL,ridgeR,Ev,camera_pitch,error_max)

				if ((len(L_point)+len(R_point))>count_th and z>best_z):
					print("update best z",z)
					best_z=z
					best_x=x
			except:
				continue
			
	return best_x
def vap_ransac_method(ridgeImage):
	row,col=ridgeImage.shape
	Eu=1200
	Ev=1200
	H=5
	camera_pitch=np.pi/180*70 #1.6 degree 

	ridgeIndex=np.array(np.where(ridgeImage>0))
	ridgeIndex=np.array(list(zip(ridgeIndex[0],ridgeIndex[1])))
	print("ridgeIndex shape",ridgeIndex.shape)
	
	ridgeL=[]
	ridgeR=[]
	
	for p in ridgeIndex:
		if (p[0]>int(row*0.7) and p[1]<=int(col*0.4)):
			ridgeL.append(p)
		if (p[0]>int(row*0.7) and p[1]>int(col*0.6)):
			ridgeR.append(p)
	ridgeL=np.array(ridgeL)
	ridgeR=np.array(ridgeR)

	# debug_point(ridgeL,ridgeR,row,col)
	print(ridgeL.shape,ridgeR.shape)
	
	error_max=10.0
	
	

	best_x_left=calculate_x_model(ridgeL,np.array([]),Ev,camera_pitch,error_max,row,col)
	best_x_right=calculate_x_model(np.array([]),ridgeR,Ev,camera_pitch,error_max,row,col)
	best_x=calculate_x_model(ridgeL,ridgeR,Ev,camera_pitch,error_max,row,col)
	
	
	print(best_x_left)
	print(best_x_right)
	L_point=generate_curve_points_base_on_best_model_left(best_x_left,Ev,camera_pitch,row,col)
	R_point=generate_curve_points_base_on_best_model_right(best_x_right,Ev,camera_pitch,row,col)

	L_point_common=generate_curve_points_base_on_best_model_left(best_x,Ev,camera_pitch,row,col)
	R_point_common=generate_curve_points_base_on_best_model_right(best_x,Ev,camera_pitch,row,col)

	print("generated point data:",L_point.shape,R_point.shape)
	result=np.zeros((row,col),np.uint8)
	for point in L_point:
		result[point[0],point[1]]=255
	for point in R_point:
		result[point[0],point[1]]=255

	result2=np.zeros((row,col),np.uint8)
	for point in L_point_common:
		result2[point[0],point[1]]=255
	for point in R_point_common:
		result2[point[0],point[1]]=255
	# cv2.imshow("curve result",result)
	# # cv2.imshow("input ridge",ridgeImage)
	# cv2.waitKey(0)

	return result,result2
	


def vap_ridgeness_detection(image, path, output_names, show_step_result = True):

	image = _prepare_grayscale_input_2D(image)
	[rowImg, colImg] = image.shape
	if show_step_result:
		vap_imshow(image, "Input image")
		vap_imsave(path, output_names+['01Input_image'])
	
	# Configuration setting
	SIGMA = 0.5
	MODE = 'constant'
	CVAL = 0

	# Gaussian smoothing
	image = img_gaussian_smooth(image, MODE, CVAL)
	if show_step_result:
		vap_imshow(image, "Smooth_version_of_image")
		vap_imsave(path, output_names+['02Smooth_version_of_image'])

	# Step 1: Compute derivatives
	imx, imy =compute_derivatives(image, mode=MODE, cval=CVAL)# np.gradient(image)#

	if show_step_result:
		vap_imshow_set(1, 2, 1, imx, "X derivative")
		vap_imshow_set(1, 2, 2, imy, "Y derivative")
		vap_imsave(path, output_names+['03X_Y_derivative'])

	# Step 3: Create a local 2x2 structure tensor for each pixel
	imx = np.expand_dims(imx,-1)
	imy = np.expand_dims(imy,-1)
	gradient_vector = np.concatenate((imx, imy), axis=2)
	gradient_vector=np.reshape(gradient_vector,(rowImg,colImg,2,1))
	gradient_vector_t=np.reshape(gradient_vector,(rowImg,colImg,1,2))
	structure_tensor=np.matmul(gradient_vector,gradient_vector_t)
	tensor_std=0.5
	structure_tensor=ndi.gaussian_filter(structure_tensor,[tensor_std,tensor_std,tensor_std,tensor_std],mode="constant")
	# for x in range(len(structure_tensor)):
	# 	for y in range(len(structure_tensor[x])):
	# 		structure_tensor[x,y]=gradient_vector[x,y].dot(gradient_vector_t[x,y])

	# Step 4: Compute eigenvalue, eigenvector for each structure tensor
	[eigValue, eigVector] = np.linalg.eig(structure_tensor)
	# eigVector = eigVector.transpose((0,1,3,2))



	# Step 5: Find dominant gradient by finding the greatest eigen value
	dominantIndex = np.argmax(eigValue, axis=-1)
	[axV, axU] = np.indices([rowImg, colImg])
	dominantVector = np.reshape(eigVector[axV.flat,axU.flat,dominantIndex.flat],[rowImg,colImg,2])
	print("dominantVector shape",dominantVector.shape)
	dominantVector_T=np.reshape(dominantVector,(rowImg,colImg,1,2))

	if show_step_result:
		vap_imnew()
		idx=slice(None,None,10)
		a=np.linspace(0,colImg-1,colImg)
		b=np.linspace(0,rowImg-1,rowImg)
		q=plt.quiver(a[idx], b[idx], dominantVector[idx,idx,0], dominantVector[idx,idx,1], pivot='mid')
		plt.title("Dominate vector", loc='center')
		vap_imsave(path, output_names+['05dominant_vector'])

	# Step 6: Dominant vector with direction
	signMask = np.matmul(dominantVector_T,gradient_vector)
	sign_max=np.amax(signMask)
	print("sign_max",np.unique(signMask))
	# for x in range(len(signMask)):
	# 	for y in range(len(signMask[x])):
	# 		signMask[x,y]=dominantVector_T[x,y].dot(gradient_vector[x,y])

	signMask=np.reshape(signMask,(rowImg,colImg))
	dominantVector[signMask < 0] *= -1
	dominantVector[abs(signMask) <1e-5]*=0
	dominantVector[signMask > 0] *= 1

	if show_step_result:
		vap_imnew()
		# idx=slice(None,None,10)
		# a=np.linspace(0,colImg-1,colImg)
		# b=np.linspace(0,rowImg-1,rowImg)
		# q=plt.quiver(a[idx], b[idx], dominantVector[idx,idx,0], dominantVector[idx,idx,1], pivot='mid')
		plt.quiver(np.arange(colImg), np.arange(rowImg), dominantVector[:,:,0], dominantVector[:,:,1], pivot='mid')
		plt.title("Dominate vector with direction", loc='center')
		# plt.show()
		vap_imsave(path, output_names+['06dominant_vector_with_direction'])

	# Step 7: Compute divergence
	vectorU = dominantVector[:,:,0]
	vectorV = dominantVector[:,:,1]
	ridgeImage = -divergence([vectorV, vectorU])
	ridgeImage[ridgeImage < 0.25] = 0

	if show_step_result:
		vap_imshow(ridgeImage, "Original Ridgeness Image")
		vap_imsave(path, output_names+['07Original_Ridgeness_Image'])

	# Step 8: Discard ridge point with large horizontal component
	theta = np.abs(np.arctan2(vectorV, vectorU))
	mask = np.logical_and(theta > np.pi*0.4, theta < np.pi*0.6)
	ridgeImage[mask] = 0

	if show_step_result:
		vap_imshow(ridgeImage, "Theta Filter Ridgeness image")
		vap_imsave(path, output_names+['08Theta_Filter_Ridgeness_image'])

	# Step 9: Confident Filter image
	ridgeImage *= (1 - np.exp(-(eigValue[:,:,0] - eigValue[:,:,1])**4 / 0.001))
	ridgeImage[ridgeImage < 0.5] = 0
	# ridgeImage[ridgeImage > 0.5] = 1
	ridgeImage[:,0:5]=0
	ridgeImage[:,colImg-10:colImg]=0

	if show_step_result:
		vap_imshow(ridgeImage, "Confident Filter Image")
		vap_imsave(path, output_names+['09Confident_Filter_Image'])

	#Step 10: RANSAC filter
	ransac_result,test_result=vap_ransac_method(ridgeImage)

	if show_step_result:
		vap_imshow(ransac_result, "Ransac Image")
		vap_imsave(path, output_names+['10RANSAC_Image'])
		vap_imshow(test_result, "Test Ransac Image")
		vap_imsave(path, output_names+['11 Test RANSAC_Image'])
	
	return ridgeImage
