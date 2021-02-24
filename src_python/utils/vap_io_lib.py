import cv2
import os
import matplotlib.pyplot as plt
from skimage import io
import imageio

graph_count = 0

def vap_imnew():
    global graph_count
    plt.figure(graph_count)
    graph_count+=1

def vap_imshow(img, title):
    """
    img: ndarray (w,h,)
    title: String   title of image
    """
    io.imshow(img)
    plt.title(title, loc = 'center')

def vap_imshow_set(show_dx, show_dy, pos, img, title):
    """
    show_dx: int
    show_dy: int
    pos: int
    img: ndarray (w,h,)
    title: String   title of image
    """
    plt.subplot(show_dx, show_dy, pos)
    io.imshow(img)
    plt.title(title, loc = 'center')

def vap_imread(img_path, flag = cv2.IMREAD_COLOR):
    """
    img_path: string    path of image
    flag: int            
    """
    img = cv2.imread(img_path, flag)
    return img

def vap_imwrite(img_path, img):
    cv2.imwrite(img_path, img)
    # imageio.imwrite(img_path, img)
    

def vap_imsave(img_path, output_names):
    """
    img_path: string        path of image
    output_names: string    output name image    
    """
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    plt.savefig(os.path.join(img_path,'_'.join(output_names)+'.png'))
    vap_imnew()

def vap_impop_up():
    """
    show image to 
    """
    io.show()