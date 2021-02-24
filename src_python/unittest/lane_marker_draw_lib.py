import cv2
import numpy as np
import imutils

def draw_line_1_1(img,px,py,width,marker_length):
  row=img.shape[0]
  current=0
  while (current<row):
    img[current:current+marker_length,py:py+width,0]=0
    img[current:current+marker_length,py:py+width,1]=255
    img[current:current+marker_length,py:py+width,2]=255
    
    current=int(current+marker_length*3)
  return img

def draw_line_1_2(img,px,py,width):
  img[px:,py:py+width,0]=0
  img[px:,py:py+width,1]=255
  img[px:,py:py+width,2]=255
  
  return img

def draw_line_1_3(img,px,py,width,distance_between_edge):
  img=draw_line_1_2(img,px,py,width)
  img=draw_line_1_2(img,px,py+width+distance_between_edge,width)
  return img

def draw_line_1_4(img,px,py,width,distance_between_edge,line_1_1_length):
  img=draw_line_1_1(img,px,py,width,line_1_1_length)
  img=draw_line_1_2(img,px,py+width+distance_between_edge,width)
  return img

def draw_line_1_5(img,px,py,width,distance_between_edge,marker_length):
  img=draw_line_1_1(img,px,py,width,marker_length)
  img=draw_line_1_1(img,px,py+width+distance_between_edge,width,marker_length)
  return img

def draw_line_2_1(img,px,py,width,marker_length):
  row=img.shape[0]
  current=px
  while (current<row):
    img[current:current+marker_length,py:py+width,0]=255
    img[current:current+marker_length,py:py+width,1]=255
    img[current:current+marker_length,py:py+width,2]=255
    
    current=int(current+marker_length*3)
  return img

def draw_line_2_2(img,px,py,width):
  img[px:,py:py+width,0]=255
  img[px:,py:py+width,1]=255
  img[px:,py:py+width,2]=255
  
  return img

def draw_line_2_4(img,px,py,width,distance_between_edge,line_1_1_length):
  img=draw_line_2_1(img,px,py,width,line_1_1_length)
  img=draw_line_2_2(img,px,py+width+distance_between_edge,width)
  return img
  
def draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness):
  # Using cv2.ellipse() method 
  # Draw a ellipse with blue line borders of thickness of -1 px 
  img = cv2.ellipse(img, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

  start_point = (center_coordinates[0]-axesLength[0],center_coordinates[1]) 
  end_point =  (center_coordinates[0]-axesLength[0],img.shape[0]-1) 
  img = cv2.line(img, start_point, end_point, color, thickness) 

  return img

def draw_line_2_2_double_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness):
  # Using cv2.ellipse() method 
  # Draw a ellipse with blue line borders of thickness of -1 px 
  img = cv2.ellipse(img, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

  start_point = (center_coordinates[0]-axesLength[0],center_coordinates[1]) 
  end_point =  (center_coordinates[0]-axesLength[0],int(img.shape[0]*0.6-1))

  img = cv2.line(img, start_point, end_point, color, thickness) 

  center_coordinates_second=(center_coordinates[0]-2*axesLength[0],int(img.shape[0]*0.6)-1)
  img = cv2.ellipse(img, center_coordinates_second, (axesLength[0],int(axesLength[1]*1.5)), 0, 0, 90, color, thickness)

  return img

def draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness):
  # Using cv2.ellipse() method 
  # Draw a ellipse with blue line borders of thickness of -1 px 
  len_per_degree=np.pi*np.max(axesLength)/180
  increase_degree=round(marker_length/len_per_degree)
  currAngle=startAngle
  while currAngle<endAngle:
    img = cv2.ellipse(img, center_coordinates, axesLength, angle, currAngle, currAngle+increase_degree, color, thickness)
    currAngle=currAngle+3*increase_degree

  start_point = (center_coordinates[0]-axesLength[0]-int(thickness/2),center_coordinates[1]+2*marker_length) 
  img=draw_line_2_1(img,start_point[1],start_point[0],thickness,marker_length)

  return img

def draw_line_2_1_double_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness):
  # Using cv2.ellipse() method 
  # Draw a ellipse with blue line borders of thickness of -1 px 
  len_per_degree=np.pi*np.max(axesLength)/180
  increase_degree=round(marker_length/len_per_degree)
  currAngle=startAngle
  while currAngle<endAngle:
    img = cv2.ellipse(img, center_coordinates, axesLength, angle, currAngle, currAngle+increase_degree, color, thickness)
    currAngle=currAngle+3*increase_degree

  start_point = (center_coordinates[0]-axesLength[0]-int(thickness/2),center_coordinates[1]+2*marker_length) 
  
  row=img.shape[0]
  current=start_point[1]
  while (current<int(img.shape[0]*0.6)-1):
    img[current:current+marker_length,start_point[0]:start_point[0]+thickness,0]=255
    img[current:current+marker_length,start_point[0]:start_point[0]+thickness,1]=255
    img[current:current+marker_length,start_point[0]:start_point[0]+thickness,2]=255
    
    current=int(current+marker_length*3)

  center_coordinates_second=(center_coordinates[0]-2*axesLength[0],int(img.shape[0]*0.6)-1)
  axesLength=(axesLength[0],int(axesLength[1]*1.5))
  len_per_degree=np.pi*np.max(axesLength)/180
  increase_degree=round(marker_length/len_per_degree)
  currAngle=increase_degree
  while currAngle<90:
    img = cv2.ellipse(img, center_coordinates_second, axesLength, 0, currAngle, currAngle+increase_degree, color, thickness)
    currAngle=currAngle+3*increase_degree
  return img
  
def draw_arrow(img,px,py,w,h,is_upper_direction):
  triangle_w=w
  triangle_h=w
  if is_upper_direction:
    vertices=np.array([[int(py+triangle_w/2),px],[py,px+triangle_h],[py+triangle_w,px+triangle_h]])
    pts=vertices.reshape((-1,1,2))
    cv2.polylines(img,[pts],isClosed=True,color=(255,255,255),thickness=2)
    cv2.fillPoly(img,[pts],color=(255,255,255))
    cv2.rectangle(img,(int(py+w/4),px+triangle_h),(int(py+w/2+w/4),int(px+h-w)),(255,255,255),-1)
  else:
    vertices=np.array([[py,px+h-triangle_h],[py+triangle_w,px+h-triangle_h],[int(py+triangle_w/2),px+h]])
    pts=vertices.reshape((-1,1,2))
    cv2.polylines(img,[pts],isClosed=True,color=(255,255,255),thickness=2)
    cv2.fillPoly(img,[pts],color=(255,255,255))
    cv2.rectangle(img,(int(py+w/4),px),(int(py+w/2+w/4),int(px+h-triangle_h)),(255,255,255),-1)
  return img
def draw_lane_split(withArrow=True):
  barrier=cv2.imread("lane_split_merge_barrier.PNG",1)
  img=np.zeros((1200,500,3),np.uint8)
  img[:,:]=0
  row,col,_=img.shape

  scale_percent=0.4
  scale_width=int(barrier.shape[1]*scale_percent)
  scale_height=int(barrier.shape[0]*scale_percent)
  barrier=cv2.resize(barrier,(scale_width,scale_height),interpolation=cv2.INTER_AREA)
  img[0:barrier.shape[0],int(col/20+col/2):int(col/20+col/2)+barrier.shape[1]]=barrier
  img=draw_line_2_2(img,int(row*0.8),int(col/20+col/2),10)
  
  row=img.shape[0]
  current=int(row*0.4)
  py=int(col/20+col/2)
  img[0:current-50,py:py+5,0]=255
  img[0:current-50,py:py+5,1]=255
  img[0:current-50,py:py+5,2]=255
  
  while (current<row):
    img[current:current+40,py:py+10,0]=255
    img[current:current+40,py:py+10,1]=255
    img[current:current+40,py:py+10,2]=255
    current=int(current+90)+1


  start_point = (int(col/20+col/2)+5,int(row*0.8))
  end_point =  (int(col/20+col/2+col/10),int(row*0.5))
  img = cv2.line(img, start_point, end_point, (255,255,255), 8) 

    
  start_point =  (int(col/20+col/2+col/10),int(row*0.5))
  end_point = (int(col-col/20),0)
  img = cv2.line(img, start_point, end_point, (255,255,255), 8) 

  # start_point =  (int(col/20+col/2)+5,int(row*0.4)-50)
  # end_point = (int(col-col*0.25),0)
  # img = cv2.line(img, start_point, end_point, (255,255,255), 5) 

  # img = draw_arrow(img,int(row*0.8),int(col*0.15),20,120,True)
  # img = draw_arrow(img,int(row*0.8),int(col*0.15+col/4),20,120,True)
  if withArrow:
    rotate_arrow=np.zeros((80,40,3),np.uint8)
    rotate_arrow[:,:]=0
    rotate_arrow = draw_arrow(rotate_arrow,10,10,20,100,True)
    rotate_arrow=imutils.rotate(rotate_arrow,-13)
    img[int(row/4.5):int(row/4.5)+rotate_arrow.shape[0],int(col/2+col/6):int(col/2+col/6)+rotate_arrow.shape[1]]= rotate_arrow
  img=img[:,int(col/2):]
  return img
def draw_lane_merge():
  img=draw_lane_split(False)
  row,col,_=img.shape
  img=cv2.flip(img,0)
  rotate_arrow=np.zeros((80,40,3),np.uint8)
  rotate_arrow[:,:]=0
  rotate_arrow = draw_arrow(rotate_arrow,10,10,20,100,True)
  rotate_arrow=imutils.rotate(rotate_arrow,13)
  img[int(4*row/4.5):int(4*row/4.5)+rotate_arrow.shape[0],int(col/2+col/15):int(col/2+col/15)+rotate_arrow.shape[1]]= rotate_arrow
  return img
def convert2carview(img):
  pts1 = np.float32([[0,0],[img.shape[1]-1,0],[0,img.shape[0]-1],[img.shape[1]-1,img.shape[0]-1]])
  pts2 = np.float32([[img.shape[1]*0.25,img.shape[0]*0.6],[img.shape[1]*0.75,img.shape[0]*0.6],[0,img.shape[0]-1],[img.shape[1]-1,img.shape[0]-1]])

  M = cv2.getPerspectiveTransform(pts1,pts2)

  dst = cv2.warpPerspective(img,M,(img.shape[1],img.shape[0]))
  return dst
def run_test():
  img=np.zeros((640,480,3),np.uint8)
  img[:,:]=0
  img_ridge=np.zeros((640,480,3),np.uint8)

  row,col,_=img.shape
  img=draw_line_2_1_double_curve(img,60,(int(col),int(row*0.3)),(200,250),0,180,250,(255,255,255),3)
  # img=convert2carview(img)
  # img=draw_arrow(img,0,0,50,100,False)
  
  cv2.imshow("img",img)
  cv2.waitKey(0)
# run_test()