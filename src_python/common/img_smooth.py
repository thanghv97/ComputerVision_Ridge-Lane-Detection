import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.stats import multivariate_normal

def gaussian_smooth_1(image, MODE, CVAL):
	H, W = image.shape
	
	image[:int(H/4), :] = ndi.gaussian_filter(image[:int(H/4), :], [0, 4], mode=MODE, cval=CVAL)
	image[int(H/4):int(H/2), :] = ndi.gaussian_filter(image[int(H/4):int(H/2), :], [0, 3], mode=MODE, cval=CVAL)
	image[int(H/2):int(3*H/4), :] = ndi.gaussian_filter(image[int(H/2):int(3*H/4), :], [0, 2], mode=MODE, cval=CVAL)
	image[int(3*H/4):, :] = ndi.gaussian_filter(image[int(3*H/4):, :], [0, 1], mode=MODE, cval=CVAL)
	
	return image

def gaussian_smooth_2(image, MODE, CVAL):
	# init parameters
	smooth_image = []
	row_img, col_img = image.shape

	sigma_y = 0.5
	sigma_x_vanish = 0.5
	sigma_x_bottom = 6
	sigma_x_slope = (sigma_x_bottom - sigma_x_vanish) / row_img
	row_margin = int(sigma_x_bottom / 2)

	# gaussian smooth
	for i in range(0, row_img):
		sigma_x = sigma_x_vanish + sigma_x_slope * i
		if i < row_margin:
			row_start, row_end, index = 0, i + row_margin, i
		elif i > row_img - row_margin:
			row_start, row_end, index = i - row_margin, row_img, row_margin - row_img + i
		else:
			row_start, row_end, index = i - row_margin, i + row_margin, row_margin

		# gaussian 
		smooth_rows = ndi.gaussian_filter(image[row_start:row_end], sigma=[sigma_x, sigma_y], mode = MODE, cval = CVAL)

		smooth_row = smooth_rows[index]
		smooth_image.append(smooth_row)
		pass
	output_image = np.array(smooth_image)
	
	return output_image

def gaussian_smooth_3(image, MODE, CVAL):
	H,W = image.shape
	image = ndi.gaussian_filter(image, [0.5,0], mode=MODE, cval=CVAL)
	output_image = np.array(image)
	sigma_high = 1.5
	N=12
	for i in range(N):
		output_image[int(i*H/N):int((i+1)*H/N), :] = ndi.gaussian_filter(image, [0, sigma_high-(sigma_high-(sigma_high-0.5)/(N-1)*i)], mode=MODE, cval=CVAL)[int(i*H/N):int((i+1)*H/N), :]
	
	return output_image

def gaussian_smooth_4(image, MODE, CVAL):
	# init parameters
	smooth_image = []
	row_img, col_img = image.shape

	sigma_y = 0.5
	sigma_x_vanish = 0.5
	sigma_x_bottom = 6
	sigma_x_slope = (sigma_x_bottom - sigma_x_vanish) / row_img
	row_margin = int(sigma_x_bottom / 2)

	# gaussian smooth
	for i in range(0, row_img):
		sigma_x = sigma_x_vanish + sigma_x_slope * i
		if i < row_margin:
			row_start, row_end, index = 0, i + row_margin, i
		elif i > row_img - row_margin:
			row_start, row_end, index = i - row_margin, row_img, row_margin - row_img + i
		else:
			row_start, row_end, index = i - row_margin, i + row_margin, row_margin

		# gaussian 
		mean = (sigma_x, sigma_y)
		cov = np.eye(2)
		kernel = multivariate_normal(mean = mean, cov = cov)
		smooth_rows = kernel.pdf(image[row_start: row_end])
		smooth_rows = ndi.gaussian_filter(image[row_start:row_end], sigma=[sigma_x, sigma_y], mode = MODE, cval = CVAL)

		smooth_row = smooth_rows[index]
		smooth_image.append(smooth_row)
		pass
	output_image = np.array(smooth_image)
	
	return output_image
	
def gaussian_smooth_5(image):
	std=np.std(image)
	return cv2.GaussianBlur(image,(5,5),sigmaX=std,sigmaY=std)

def img_gaussian_smooth(image,MODE,CVAL):
	# output_image = gaussian_smooth_1(image, MODE, CVAL)
	# output_image = gaussian_smooth_2(image, MODE, CVAL)
	# output_image = gaussian_smooth_3(image, MODE, CVAL)
	# output_image = gaussian_smooth_4(image, MODE, CVAL)
	output_image = gaussian_smooth_5(image)
	return output_image
	