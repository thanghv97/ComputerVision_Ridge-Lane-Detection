from remove_shadow import shadow_remove_1,shadow_remove_2
import cv2
import matplotlib.pyplot as plt
from skimage import io

def main():
    input_url = './image/input/resize_rawvideo_30_fov_006601.jpg'
    output_url = './image/output/resize_rawvideo_30_fov_006601.jpg'
    input_img = cv2.imread(input_url)
    output_img = shadow_remove_2(input_img)
    # cv2.imwrite(output_url,output_img)
    io.imshow(output_img)
    plt.title('reconstruct_image', loc = 'center')
    plt.savefig(output_url)

if __name__ == "__main__":
    main()
