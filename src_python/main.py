import cv2
import os
import rawpy
import imageio

from utils import vap_imshow, vap_imread, vap_imwrite
from utils import vap_time_decorator
from common import hough_based_detect_line, lane_decision
from common import img_illuminant_invariance
from ridgeness_detection import vap_ridgeness_detection

out_img_path_ridge_illu_invar = './image/output/ridgeness_illu_invar'
out_img_path_ridge = './image/output/ridgeness'
out_img_path_canny = './image/output/canny'


@vap_time_decorator
def lane_detection_by_ridge(img, out_name):
    global out_img_path_ridge
    path = out_img_path_ridge

    # =============================================
    # convert to gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get ridge image
    ridge_img = vap_ridgeness_detection(gray_img, path, out_name, False)

    # get hough line image
    # ridge_hough_lines = hough_based_detect_line(ridge_img, 15)
    # for line in ridge_hough_lines:
    #     leftLane, rightLane = lane_decision(ridge_img.shape[0], line)

    #     for x1,y1,x2,y2 in line:
    # 		# show Hough line by red color (BGR = (0,0,255))
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    vap_imwrite(path + '/'.join(out_name) + '/ridge_feature.jpg', ridge_img)
    # vap_imwrite(path + '/'.join(out_name) + '/ridge_houghlines.jpg', img)


@vap_time_decorator
def lane_detection_by_ridge_illu_invar(img, out_name):
    global out_img_path_ridge_illu_invar
    path = out_img_path_ridge_illu_invar

    # =============================================
    # convert to gray image
    gray_img, theta = img_illuminant_invariance(img, path, img_shw_step=True)
    """
    # get ridge image
    ridge_img = vap_ridgeness_detection(gray_img, path, out_name, show_step_result=True)

    # get hough line image
    ridge_hough_lines = hough_based_detect_line(ridge_img, 15)
    for line in ridge_hough_lines:
        leftLane, rightLane = lane_decision(ridge_img.shape[0], line)

        for x1,y1,x2,y2 in line:
			# show Hough line by red color (BGR = (0,0,255))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    vap_imwrite(path + '/'.join(out_name) + '/ridge_feature.jpg', ridge_img)
    vap_imwrite(path + '/'.join(out_name) + '/ridge_houghlines.jpg', img)
    """


def test_image():
    # =============================================
    # Read image
    img_path = '/home/thanghv7/Downloads/shadow26.png'
    # img_path = './image/input/rawvideo_30_fov_006601.jpg'
    # img_path = '../test/input/image/MKZ_LONGCAMERA/2190(US_001).jpg'
    img = vap_imread(img_path)

    # raw = rawpy.imread(img_path)
    # img = raw.postprocess(no_auto_bright=True, use_auto_wb=False, gamma=None)
    print(img.shape, end=" ")

    # =============================================
    # Resize image
    scale = 1
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    print(" => ", img.shape)

    # =============================================
    # Process
    # lane_detection_by_ridge(img, [])
    lane_detection_by_ridge_illu_invar(img, [])


if __name__ == "__main__":
    if not os.path.exists('./image/output'):
        os.mkdir('./image/output')

    test_image()
