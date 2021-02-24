from scipy import sparse
import cv2
import numpy as np
from scipy.sparse.linalg import spsolve
from common.img_illu_invar import derive_1D_shadow_free_image,compute_gray_img,compute_log_chromaticity
import os
import matplotlib.pyplot as plt
from skimage import io

from scipy import fftpack, signal



def solve_poission_equation_numerically(f_image):
    H,W =f_image.shape
    F = f_image.reshape(H*W)

    m1 = sparse.eye(H,H).tocsr()
    m2 = sparse.diags([1, -4, 1], [-1, 0, 1], shape=(W, W)).tocsr()
    m1_1 = sparse.diags([1, 0, 1], [-1, 0, 1], shape=(H, H)).tocsr()
    m1_2 = sparse.eye(W,W).tocsr()
    p1 = sparse.kron(m1,m2).tocsr()+sparse.kron(m1_1,m1_2).tocsr()

    m3 = np.zeros([H,H])
    m3[0,1]=1
    m3[-1,-2]=1
    m4 = sparse.eye(W,W).tocsr()
    p2 = sparse.kron(m3,m4).tocsr()

    m5 = sparse.eye(H,H).tocsr()
    m6 = np.zeros([W,W])
    m6[0,1] = 1
    m6[-1,-2] = 1
    p3 = sparse.kron(m5,m6).tocsr()

    A=p1+p2+p3

    #u_image is the shadow image
    u_image = spsolve(A, F)
    u_image = u_image.reshape(H,W)
    return u_image

def solve_poission_equation_numerically_2(S_T_gradient_rho_prime):
    H,W,_ = S_T_gradient_rho_prime.shape

    t=0
    print('t=',t)
    T_X_k_gradient_rho_prime_t = S_T_gradient_rho_prime[:,:,0]
    T_Y_k_gradient_rho_prime_t = S_T_gradient_rho_prime[:,:,1]

    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    a = np.zeros(H)
    a[1] = 1
    a_x_star = np.fft.fft(a)-1
    a_x = np.conjugate(a_x_star)
    a = np.zeros(W)
    a[1] = 1
    a_y_star = np.fft.fft(a)-1
    a_y = np.conjugate(a_y_star)
    del a
    epsilon = 1


    t+=1
    print('t=',t)
    T_X_k_gradient_rho_prime_t = signal.convolve2d(T_X_k_gradient_rho_prime_t,kernel,mode='same')
    T_Y_k_gradient_rho_prime_t = signal.convolve2d(T_Y_k_gradient_rho_prime_t,kernel,mode='same')

    F_x = np.fft.fft2(T_X_k_gradient_rho_prime_t)
    F_y = np.fft.fft2(T_Y_k_gradient_rho_prime_t)

    Z = (a_x_star[:,None]*F_x +a_y_star[None,:]*F_y)/((np.abs(a_x_star)**2)[:,None]+(np.abs(a_y_star)**2)[None,:])
    Z[0,0]=0

    T_X_gradient_rho_prime_t = np.fft.ifft(a_x[:,None]*Z)
    T_Y_gradient_rho_prime_t = np.fft.ifft(a_y[None,:]*Z)

    condition = True
    while condition:
        t+=1
        print('t=',t)
        T_X_k_gradient_rho_prime_t = signal.convolve2d(T_X_k_gradient_rho_prime_t,kernel,mode='same')
        T_Y_k_gradient_rho_prime_t = signal.convolve2d(T_Y_k_gradient_rho_prime_t,kernel,mode='same')

        F_x = np.fft.fft2(T_X_k_gradient_rho_prime_t)
        F_y = np.fft.fft2(T_Y_k_gradient_rho_prime_t)

        Z = (a_x_star[:,None]*F_x +a_y_star[None,:]*F_y)/((np.abs(a_x_star)**2)[:,None]+(np.abs(a_y_star)**2)[None,:])
        Z[0,0]=0
        print('count not equal nan Z', np.sum(Z!=Z))

        prev_T_X_gradient_rho_prime_t = T_X_gradient_rho_prime_t
        prev_T_Y_gradient_rho_prime_t = T_Y_gradient_rho_prime_t
        T_X_gradient_rho_prime_t = np.fft.ifft(a_x[:,None]*Z)
        T_Y_gradient_rho_prime_t = np.fft.ifft(a_y[None,:]*Z)

        print('np.sum(T_X_gradient_rho_prime_t-prev_T_X_gradient_rho_prime_t)',np.sum(T_X_gradient_rho_prime_t-prev_T_X_gradient_rho_prime_t))
        print('np.sum(T_Y_gradient_rho_prime_t-prev_T_Y_gradient_rho_prime_t)',np.sum(T_Y_gradient_rho_prime_t-prev_T_Y_gradient_rho_prime_t))
        residual = np.sum(np.abs(T_X_gradient_rho_prime_t-prev_T_X_gradient_rho_prime_t))+np.sum(np.abs(T_Y_gradient_rho_prime_t-prev_T_Y_gradient_rho_prime_t))
        print('residual',residual)
        condition = (residual>=epsilon)

    rho_k = np.fft.ifft(Z).real()

    return rho_k



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
	return np.concatenate([im_x[:,:,None],im_y[:,:,None]],axis=2)


def threshold_gradient(img,threshold):
    condition_1 = np.sqrt(img[:,:,0]**2+img[:,:,1]**2)<threshold
    img[condition_1]=0
    # print('img.shape',img.shape)
    # print('condition_1.shape',condition_1.shape)
    return img

def threshold_shadow(img,grey_img,threshold_1,threshold_2):
    condition_1 = np.sqrt(img[:,:,0]**2+img[:,:,1]**2)>=threshold_1
    condition_2 = np.sqrt(grey_img[:,:,0]**2+grey_img[:,:,1]**2)<=threshold_2
    # print('img.shape',img.shape)
    # print('condition_1.shape',condition_1.shape)
    # print('condition_2.shape',condition_2.shape)
    # print('condition_1 and condition_2.shape',(np.logical_and(condition_1, condition_2)).shape)

    output_url = './image/output'
    io.imshow(np.logical_and(condition_1, condition_2))
    plt.title('condition', loc = 'center')
    plt.savefig(os.path.join(output_url,'condition.png'))
    plt.close()

    img[np.logical_and(condition_1, condition_2)]=0
    return img


def shadow_remove_1(img):
    '''
    :param img: ndarray, Input image.
    '''
    output_url = './image/output'
    print('step_1')

    plt.hist(img[:,:,0].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_R_histogram.png'))
    plt.close()
    plt.hist(img[:,:,1].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_G_histogram.png'))
    plt.close()
    plt.hist(img[:,:,2].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_B_histogram.png'))
    plt.close()

    # grey_img = derive_1D_shadow_free_image(img)[0]

    out_img = compute_gray_img(compute_log_chromaticity(img),133)
    min_val = np.amin(out_img)
    max_val = np.amax(out_img)
    grey_img = (out_img - min_val) / ((max_val - min_val)/255)

    io.imshow(grey_img)
    plt.title('grey_img', loc = 'center')
    plt.savefig(os.path.join(output_url,'grey_img.png'))
    plt.close()


    print('step_2')
    const = 1
    img = np.where(img == 0, const, img).astype(np.float)
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]

    rho_prime_R = np.log(img_R)
    rho_prime_G = np.log(img_G)
    rho_prime_B = np.log(img_B)

    plt.hist(rho_prime_R.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_log_R_histogram.png'))
    plt.close()
    plt.hist(rho_prime_G.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_log_G_histogram.png'))
    plt.close()
    plt.hist(rho_prime_B.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_log_B_histogram.png'))
    plt.close()


    print('step_3')


    gradient_rho_prime_R = compute_derivatives(rho_prime_R)
    gradient_rho_prime_G = compute_derivatives(rho_prime_G)
    gradient_rho_prime_B = compute_derivatives(rho_prime_B)

    io.imshow(np.sqrt(gradient_rho_prime_R[:,:,0]**2+gradient_rho_prime_R[:,:,1]**2))
    plt.title('gradient_rho_prime_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_rho_prime_R.png'))
    plt.close()

    io.imshow(np.sqrt(gradient_rho_prime_G[:,:,0]**2+gradient_rho_prime_G[:,:,1]**2))
    plt.title('gradient_rho_prime_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_rho_prime_G.png'))
    plt.close()

    io.imshow(np.sqrt(gradient_rho_prime_B[:,:,0]**2+gradient_rho_prime_B[:,:,1]**2))
    plt.title('gradient_rho_prime_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_rho_prime_B.png'))
    plt.close()

    print('step_4')
    threshold = 1
    # T_gradient_rho_prime_R = gradient_rho_prime_R
    # T_gradient_rho_prime_G = gradient_rho_prime_G
    # T_gradient_rho_prime_B = gradient_rho_prime_B
    T_gradient_rho_prime_R = threshold_gradient(gradient_rho_prime_R,threshold)
    T_gradient_rho_prime_G = threshold_gradient(gradient_rho_prime_G,threshold)
    T_gradient_rho_prime_B = threshold_gradient(gradient_rho_prime_B,threshold)

    io.imshow(np.sqrt(T_gradient_rho_prime_R[:,:,0]**2+T_gradient_rho_prime_R[:,:,1]**2))
    plt.title('T_gradient_rho_prime_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'T_gradient_rho_prime_R.png'))
    plt.close()

    io.imshow(np.sqrt(T_gradient_rho_prime_G[:,:,0]**2+T_gradient_rho_prime_G[:,:,1]**2))
    plt.title('T_gradient_rho_prime_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'T_gradient_rho_prime_G.png'))
    plt.close()

    io.imshow(np.sqrt(T_gradient_rho_prime_B[:,:,0]**2+T_gradient_rho_prime_B[:,:,1]**2))
    plt.title('T_gradient_rho_prime_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'T_gradient_rho_prime_B.png'))
    plt.close()

    print('step_5')
    threshold_1 = 2
    threshold_2 = 55 # or 60

    print('grey_img.shape',grey_img.shape)
    gradient_grey_img = compute_derivatives(grey_img)

    io.imshow(np.sqrt(gradient_grey_img[:,:,0]**2+gradient_grey_img[:,:,1]**2))
    plt.title('gradient_grey_img', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_grey_img.png'))
    plt.close()

    # S_T_gradient_rho_prime_R = T_gradient_rho_prime_R
    # S_T_gradient_rho_prime_G = T_gradient_rho_prime_G
    # S_T_gradient_rho_prime_B = T_gradient_rho_prime_B
    S_T_gradient_rho_prime_R = threshold_shadow(T_gradient_rho_prime_R,gradient_grey_img,threshold_1,threshold_2)
    S_T_gradient_rho_prime_G = threshold_shadow(T_gradient_rho_prime_G,gradient_grey_img,threshold_1,threshold_2)
    S_T_gradient_rho_prime_B = threshold_shadow(T_gradient_rho_prime_B,gradient_grey_img,threshold_1,threshold_2)

    io.imshow(np.sqrt(S_T_gradient_rho_prime_R[:,:,0]**2+S_T_gradient_rho_prime_R[:,:,1]**2))
    plt.title('S_T_gradient_rho_prime_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'S_T_gradient_rho_prime_R.png'))
    plt.close()

    io.imshow(np.sqrt(S_T_gradient_rho_prime_G[:,:,0]**2+S_T_gradient_rho_prime_G[:,:,1]**2))
    plt.title('S_T_gradient_rho_prime_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'S_T_gradient_rho_prime_G.png'))
    plt.close()

    io.imshow(np.sqrt(S_T_gradient_rho_prime_B[:,:,0]**2+S_T_gradient_rho_prime_B[:,:,1]**2))
    plt.title('S_T_gradient_rho_prime_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'S_T_gradient_rho_prime_B.png'))
    plt.close()

    f_R = compute_derivatives(S_T_gradient_rho_prime_R[:,:,0])[:,:,0]+compute_derivatives(S_T_gradient_rho_prime_R[:,:,1])[:,:,1]
    f_G = compute_derivatives(S_T_gradient_rho_prime_G[:,:,0])[:,:,0]+compute_derivatives(S_T_gradient_rho_prime_G[:,:,1])[:,:,1]
    f_B = compute_derivatives(S_T_gradient_rho_prime_B[:,:,0])[:,:,0]+compute_derivatives(S_T_gradient_rho_prime_B[:,:,1])[:,:,1]

    print('step_6')
    reconstruct_R = solve_poission_equation_numerically(f_R)
    reconstruct_G = solve_poission_equation_numerically(f_G)
    reconstruct_B = solve_poission_equation_numerically(f_B)

    io.imshow(reconstruct_R)
    plt.title('Reconstruct_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'Reconstruct_R.png'))
    plt.close()

    io.imshow(reconstruct_G)
    plt.title('Reconstruct_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'Reconstruct_G.png'))
    plt.close()

    io.imshow(reconstruct_B)
    plt.title('Reconstruct_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'Reconstruct_B.png'))
    plt.close()


    min_val_R = np.min(reconstruct_R)
    max_val_R = np.max(reconstruct_R)
    reconstruct_R = (reconstruct_R - min_val_R) / ((max_val_R - min_val_R)/6)
    reconstruct_R = np.exp(reconstruct_R)
    min_val_R = np.min(reconstruct_R)
    max_val_R = np.max(reconstruct_R)
    reconstruct_R = (reconstruct_R - min_val_R) / ((max_val_R - min_val_R)/255)

    min_val_G = np.min(reconstruct_G)
    max_val_G = np.max(reconstruct_G)
    reconstruct_G = (reconstruct_G - min_val_G) / ((max_val_G - min_val_G)/6)
    reconstruct_G = np.exp(reconstruct_G)
    min_val_G = np.min(reconstruct_G)
    max_val_G = np.max(reconstruct_G)
    reconstruct_G = (reconstruct_G - min_val_G) / ((max_val_G - min_val_G)/255)

    min_val_B = np.min(reconstruct_B)
    max_val_B = np.max(reconstruct_B)
    reconstruct_B = (reconstruct_B - min_val_B) / ((max_val_B - min_val_B)/6)
    reconstruct_B = np.exp(reconstruct_B)
    min_val_B = np.min(reconstruct_B)
    max_val_B = np.max(reconstruct_B)
    reconstruct_B = (reconstruct_B - min_val_B) / ((max_val_B - min_val_B)/255)

    print('step_7')
    shadow_free_img = np.concatenate([reconstruct_R[:,:,None],reconstruct_G[:,:,None],reconstruct_B[:,:,None]],axis=2)

    plt.hist(shadow_free_img[:,:,0].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_log_R_histogram.png'))
    plt.close()
    plt.hist(shadow_free_img[:,:,1].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_log_G_histogram.png'))
    plt.close()
    plt.hist(shadow_free_img[:,:,2].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_log_B_histogram.png'))
    plt.close()

    print('max:',np.max(shadow_free_img))
    print('min:',np.min(shadow_free_img))

    hist, bins = np.histogram(shadow_free_img,300)
    plt.hist(hist, bins = bins)
    plt.savefig(os.path.join(output_url,'output_image_histogram.png'))
    plt.close()

    min_val = np.min(shadow_free_img)
    max_val = np.max(shadow_free_img)
    shadow_free_img = shadow_free_img.astype('uint8')

    plt.hist(shadow_free_img.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_normalize_histogram.png'))
    plt.close()


    print('max:',np.max(shadow_free_img))
    print('min:',np.min(shadow_free_img))

    return shadow_free_img


def shadow_remove_2(img):
    '''
    :param img: ndarray, Input image.
    '''
    output_url = './image/output'
    print('step_1')

    plt.hist(img[:,:,0].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_R_histogram.png'))
    plt.close()
    plt.hist(img[:,:,1].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_G_histogram.png'))
    plt.close()
    plt.hist(img[:,:,2].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_B_histogram.png'))
    plt.close()

    # grey_img = derive_1D_shadow_free_image(img)[0]

    out_img = compute_gray_img(compute_log_chromaticity(img),158)
    min_val = np.amin(out_img)
    max_val = np.amax(out_img)
    grey_img = (out_img - min_val) / ((max_val - min_val)/255)

    io.imshow(grey_img)
    plt.title('grey_img', loc = 'center')
    plt.savefig(os.path.join(output_url,'grey_img.png'))
    plt.close()


    print('step_2')
    const = 1
    img = np.where(img == 0, const, img).astype(np.float)
    img_R = img[:,:,0]
    img_G = img[:,:,1]
    img_B = img[:,:,2]

    rho_prime_R = np.log(img_R)
    rho_prime_G = np.log(img_G)
    rho_prime_B = np.log(img_B)

    plt.hist(rho_prime_R.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_log_R_histogram.png'))
    plt.close()
    plt.hist(rho_prime_G.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_log_G_histogram.png'))
    plt.close()
    plt.hist(rho_prime_B.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'input_image_log_B_histogram.png'))
    plt.close()


    print('step_3')


    gradient_rho_prime_R = compute_derivatives(rho_prime_R)
    gradient_rho_prime_G = compute_derivatives(rho_prime_G)
    gradient_rho_prime_B = compute_derivatives(rho_prime_B)

    io.imshow(np.sqrt(gradient_rho_prime_R[:,:,0]**2+gradient_rho_prime_R[:,:,1]**2))
    plt.title('gradient_rho_prime_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_rho_prime_R.png'))
    plt.close()

    io.imshow(np.sqrt(gradient_rho_prime_G[:,:,0]**2+gradient_rho_prime_G[:,:,1]**2))
    plt.title('gradient_rho_prime_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_rho_prime_G.png'))
    plt.close()

    io.imshow(np.sqrt(gradient_rho_prime_B[:,:,0]**2+gradient_rho_prime_B[:,:,1]**2))
    plt.title('gradient_rho_prime_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_rho_prime_B.png'))
    plt.close()

    print('step_4')
    threshold = 1
    # T_gradient_rho_prime_R = gradient_rho_prime_R
    # T_gradient_rho_prime_G = gradient_rho_prime_G
    # T_gradient_rho_prime_B = gradient_rho_prime_B
    T_gradient_rho_prime_R = threshold_gradient(gradient_rho_prime_R,threshold)
    T_gradient_rho_prime_G = threshold_gradient(gradient_rho_prime_G,threshold)
    T_gradient_rho_prime_B = threshold_gradient(gradient_rho_prime_B,threshold)

    io.imshow(np.sqrt(T_gradient_rho_prime_R[:,:,0]**2+T_gradient_rho_prime_R[:,:,1]**2))
    plt.title('T_gradient_rho_prime_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'T_gradient_rho_prime_R.png'))
    plt.close()

    io.imshow(np.sqrt(T_gradient_rho_prime_G[:,:,0]**2+T_gradient_rho_prime_G[:,:,1]**2))
    plt.title('T_gradient_rho_prime_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'T_gradient_rho_prime_G.png'))
    plt.close()

    io.imshow(np.sqrt(T_gradient_rho_prime_B[:,:,0]**2+T_gradient_rho_prime_B[:,:,1]**2))
    plt.title('T_gradient_rho_prime_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'T_gradient_rho_prime_B.png'))
    plt.close()

    print('step_5')
    threshold_1 = 2
    threshold_2 = 55 # or 60

    print('grey_img.shape',grey_img.shape)
    gradient_grey_img = compute_derivatives(grey_img)

    io.imshow(np.sqrt(gradient_grey_img[:,:,0]**2+gradient_grey_img[:,:,1]**2))
    plt.title('gradient_grey_img', loc = 'center')
    plt.savefig(os.path.join(output_url,'gradient_grey_img.png'))
    plt.close()

    # S_T_gradient_rho_prime_R = T_gradient_rho_prime_R
    # S_T_gradient_rho_prime_G = T_gradient_rho_prime_G
    # S_T_gradient_rho_prime_B = T_gradient_rho_prime_B
    S_T_gradient_rho_prime_R = threshold_shadow(T_gradient_rho_prime_R,gradient_grey_img,threshold_1,threshold_2)
    S_T_gradient_rho_prime_G = threshold_shadow(T_gradient_rho_prime_G,gradient_grey_img,threshold_1,threshold_2)
    S_T_gradient_rho_prime_B = threshold_shadow(T_gradient_rho_prime_B,gradient_grey_img,threshold_1,threshold_2)

    io.imshow(np.sqrt(S_T_gradient_rho_prime_R[:,:,0]**2+S_T_gradient_rho_prime_R[:,:,1]**2))
    plt.title('S_T_gradient_rho_prime_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'S_T_gradient_rho_prime_R.png'))
    plt.close()

    io.imshow(np.sqrt(S_T_gradient_rho_prime_G[:,:,0]**2+S_T_gradient_rho_prime_G[:,:,1]**2))
    plt.title('S_T_gradient_rho_prime_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'S_T_gradient_rho_prime_G.png'))
    plt.close()

    io.imshow(np.sqrt(S_T_gradient_rho_prime_B[:,:,0]**2+S_T_gradient_rho_prime_B[:,:,1]**2))
    plt.title('S_T_gradient_rho_prime_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'S_T_gradient_rho_prime_B.png'))
    plt.close()

    print('step_6')
    print('solve_poission_equation_numerically_2_R')
    reconstruct_R = solve_poission_equation_numerically_2(S_T_gradient_rho_prime_R)
    print('solve_poission_equation_numerically_2_G')
    reconstruct_G = solve_poission_equation_numerically_2(S_T_gradient_rho_prime_G)
    print('solve_poission_equation_numerically_2_B')
    reconstruct_B = solve_poission_equation_numerically_2(S_T_gradient_rho_prime_B)

    io.imshow(reconstruct_R)
    plt.title('Reconstruct_R', loc = 'center')
    plt.savefig(os.path.join(output_url,'Reconstruct_R.png'))
    plt.close()

    io.imshow(reconstruct_G)
    plt.title('Reconstruct_G', loc = 'center')
    plt.savefig(os.path.join(output_url,'Reconstruct_G.png'))
    plt.close()

    io.imshow(reconstruct_B)
    plt.title('Reconstruct_B', loc = 'center')
    plt.savefig(os.path.join(output_url,'Reconstruct_B.png'))
    plt.close()


    min_val_R = np.min(reconstruct_R)
    max_val_R = np.max(reconstruct_R)
    reconstruct_R = (reconstruct_R - min_val_R) / ((max_val_R - min_val_R)/6)
    reconstruct_R = np.exp(reconstruct_R)
    min_val_R = np.min(reconstruct_R)
    max_val_R = np.max(reconstruct_R)
    reconstruct_R = (reconstruct_R - min_val_R) / ((max_val_R - min_val_R)/255)

    min_val_G = np.min(reconstruct_G)
    max_val_G = np.max(reconstruct_G)
    reconstruct_G = (reconstruct_G - min_val_G) / ((max_val_G - min_val_G)/6)
    reconstruct_G = np.exp(reconstruct_G)
    min_val_G = np.min(reconstruct_G)
    max_val_G = np.max(reconstruct_G)
    reconstruct_G = (reconstruct_G - min_val_G) / ((max_val_G - min_val_G)/255)

    min_val_B = np.min(reconstruct_B)
    max_val_B = np.max(reconstruct_B)
    reconstruct_B = (reconstruct_B - min_val_B) / ((max_val_B - min_val_B)/6)
    reconstruct_B = np.exp(reconstruct_B)
    min_val_B = np.min(reconstruct_B)
    max_val_B = np.max(reconstruct_B)
    reconstruct_B = (reconstruct_B - min_val_B) / ((max_val_B - min_val_B)/255)

    print('step_7')
    shadow_free_img = np.concatenate([reconstruct_R[:,:,None],reconstruct_G[:,:,None],reconstruct_B[:,:,None]],axis=2)

    plt.hist(shadow_free_img[:,:,0].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_log_R_histogram.png'))
    plt.close()
    plt.hist(shadow_free_img[:,:,1].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_log_G_histogram.png'))
    plt.close()
    plt.hist(shadow_free_img[:,:,2].reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_log_B_histogram.png'))
    plt.close()

    print('max:',np.max(shadow_free_img))
    print('min:',np.min(shadow_free_img))

    hist, bins = np.histogram(shadow_free_img,300)
    plt.hist(hist, bins = bins)
    plt.savefig(os.path.join(output_url,'output_image_histogram.png'))
    plt.close()

    min_val = np.min(shadow_free_img)
    max_val = np.max(shadow_free_img)
    shadow_free_img = shadow_free_img.astype('uint8')

    plt.hist(shadow_free_img.reshape(-1), bins = 255)
    plt.savefig(os.path.join(output_url,'output_image_normalize_histogram.png'))
    plt.close()


    print('max:',np.max(shadow_free_img))
    print('min:',np.min(shadow_free_img))

    return shadow_free_img
