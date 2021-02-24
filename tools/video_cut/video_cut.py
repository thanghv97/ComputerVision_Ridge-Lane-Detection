import cv2
import os

if __name__ == "__main__":
    # input 
    video_path_in = "/home/thanghv7/Autopilot/Video/MKZ_LONGCAMERA/2_VN_001.mp4"
    image_path_out = "/home/thanghv7/Autopilot/Video/MKZ_LONGCAMERA/1_image_cut/"

    if not os.path.exists(image_path_out):
        os.mkdir(image_path_out)

    # capture video input
    cap = cv2.VideoCapture(video_path_in)
    H, W, F = cap.get(4), cap.get(3), cap.get(5)

    print("Frame input:", "width", W, "height", H, "fps", F)
    i = 0
    j = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            # frame = frame[:int(H-200), :] # for test_3
            # frame = frame[:int(H-100), :] # for test_4
            # frame = frame[90:, :] # for test_6
      
            
            # frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            if i % 30 == 0:
                cv2.imwrite(image_path_out + "/" + str(j) + ".jpg", frame)
                j+=1
            i+=1
        else:
            break
    print("total image:", j )
    cap.release()
    cv2.destroyAllWindows()