import lane_marker_draw_lib as lmdl
import numpy as np
import cv2
import unittest
import os
#import your ridgeness_detection module as below
#now I'm use ridgeness_detection.py in the master
import sys
sys.path.insert(1,'../')
import ridgeness_detection as rd

def write_md(output_path,test_case_name,test_case_header):
  files=[]
  filewriter=open(output_path+"/../"+test_case_name+".md","w")
  for r,d,f in os.walk(output_path):
    for file in f:
      if '.png' in file:
        files.append(file)
  files.sort()
  filewriter.write("# "+test_case_header+"  \n")
  filewriter.write("[Go back to Summary](../summary.md)  \n")
  for file in files:
    data="*"+file+"*  \n!["+file+"](./data/"+file+")  \n"
    filewriter.write(data)
  filewriter.close()
  
def process_algorithm(img,img_ridge,output_path,test_case_name):
  gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  rd_out=rd.vap_ridgeness_detection(gray,output_path,['unittest.png'],True)
  cv2.imwrite(output_path+"/0_input.png",img)
  cv2.imwrite(output_path+"/0_ridgeness_expect.png",img_ridge)
  cv2.imwrite(output_path+"/0_ridgeness_output.png",rd_out)

def init_img(test_case_name):
  output_path=os.getcwd()+"/"+test_case_name+"/data/"
  if not os.path.exists(os.getcwd()+"/"+test_case_name):
    os.mkdir(os.getcwd()+"/"+test_case_name)
    os.mkdir(output_path)
  img=np.zeros((640,480,3),np.uint8)
  img[:,:]=0
  img_ridge=np.zeros((640,480,3),np.uint8)
  return output_path,img,img_ridge

def run_test_white_dash_line(width,line_length,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  img=lmdl.draw_line_2_1(img,0,(int)(img.shape[1]/2-width/2),width,line_length)
  
  lmdl.draw_line_2_1(img_ridge,0,(int)(img.shape[1]/2)-1,2,line_length)
  process_algorithm(img,img_ridge,output_path,test_case_name)

  test_case_header = "test_white_dash_line width:" + str(width) + "px, line_length:"+str(line_length)+"px"
  write_md(output_path, test_case_name, test_case_header)

def run_test_white_solid_line(width,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  img=lmdl.draw_line_2_2(img,0,(int)(img.shape[1]/2-width/2),width)
  
  lmdl.draw_line_2_2(img_ridge,0,(int)(img.shape[1]/2)-1,2)
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_solid_line width:"+str(width)+"px"
  write_md(output_path,test_case_name,test_case_header)
  
def run_test_white_solid_line_two_lane(width,distance_between_edge,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  img=lmdl.draw_line_2_2(img,0,(int)(img.shape[1]/2-width/2-distance_between_edge/2),width)
  img=lmdl.draw_line_2_2(img,0,(int)(img.shape[1]/2-width/2+distance_between_edge/2),width)
  
  lmdl.draw_line_2_2(img_ridge,0,(int)(img.shape[1]/2-distance_between_edge/2)-1,2)
  lmdl.draw_line_2_2(img_ridge,0,(int)(img.shape[1]/2+distance_between_edge/2)-1,2)
  
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_solid_line_two_lane, line width:"+str(width)+"px distance between two line:"+str(distance_between_edge)+"px"
  write_md(output_path,test_case_name,test_case_header)

def run_test_white_dashed_line_two_lane(width,distance_between_edge,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  img=lmdl.draw_line_2_1(img,0,(int)(img.shape[1]/2-width/2-distance_between_edge/2),width,60)
  img=lmdl.draw_line_2_1(img,0,(int)(img.shape[1]/2-width/2+distance_between_edge/2),width,60)
  
  lmdl.draw_line_2_1(img_ridge,0,(int)(img.shape[1]/2-distance_between_edge/2)-1,2,60)
  lmdl.draw_line_2_1(img_ridge,0,(int)(img.shape[1]/2+distance_between_edge/2)-1,2,60)
  

  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_dashed_line_two_lane, line width:"+str(width)+"px distance between two line:"+str(distance_between_edge)+"px"
  write_md(output_path,test_case_name,test_case_header)

def run_test_white_solid_line_three_lane(width,distance_between_edge,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  img=lmdl.draw_line_2_2(img,0,(int)(img.shape[1]/2-width/2-distance_between_edge),width)
  img=lmdl.draw_line_2_2(img,0,(int)(img.shape[1]/2-width/2),width)
  img=lmdl.draw_line_2_2(img,0,(int)(img.shape[1]/2-width/2+distance_between_edge),width)
  
  lmdl.draw_line_2_2(img_ridge,0,(int)(img.shape[1]/2-distance_between_edge)-1,2)
  lmdl.draw_line_2_2(img_ridge,0,(int)(img.shape[1]/2)-1,2)
  lmdl.draw_line_2_2(img_ridge,0,(int)(img.shape[1]/2+distance_between_edge)-1,2)
  
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_solid_line_three_lane, line width:"+str(width)+"px distance between lines:"+str(distance_between_edge)+"px"
  write_md(output_path,test_case_name,test_case_header)

def run_test_white_dashed_line_three_lane(width,distance_between_edge,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  img=lmdl.draw_line_2_1(img,0,(int)(img.shape[1]/2-width/2-distance_between_edge),width,60)
  img=lmdl.draw_line_2_1(img,0,(int)(img.shape[1]/2-width/2),width,60)
  img=lmdl.draw_line_2_1(img,0,(int)(img.shape[1]/2-width/2+distance_between_edge),width,60)
  
  lmdl.draw_line_2_1(img_ridge,0,(int)(img.shape[1]/2-distance_between_edge)-1,2,60)
  lmdl.draw_line_2_1(img_ridge,0,(int)(img.shape[1]/2)-1,2,60) 
  lmdl.draw_line_2_1(img_ridge,0,(int)(img.shape[1]/2+distance_between_edge)-1,2,60)
  

  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_dashed_line_three_lane, line width:"+str(width)+"px distance between lines:"+str(distance_between_edge)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_white_solid_curve_line(width,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  center_coordinates = (int(img.shape[1]), int(2*img.shape[0]*0.3)) 
  axesLength = (240, 280) 
  angle = 0
  startAngle = 180
  endAngle = 280
  color = (255, 255, 255)
  thickness = width
  img=lmdl.draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_2_curve(img_ridge,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_solid_curve_line, line width:"+str(width)+"px "
  write_md(output_path,test_case_name,test_case_header)

def run_test_white_dashed_curve_line(width,marker_length,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  center_coordinates = (int(img.shape[1]), int(2*img.shape[0]*0.3)) 
  axesLength = (240, 280) 
  angle = 0
  startAngle = 180
  endAngle = 280
  color = (255, 255, 255)
  thickness = width
  img=lmdl.draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_1_curve(img_ridge,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_dashed_curve_line, line width:"+str(width)+"px marker length:"+str(marker_length)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_white_solid_curve_line_two_lane(width,distance_between_lane,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  center_coordinates = (int(img.shape[1]), int(2*img.shape[0]*0.3)) 
  axesLength = (140, 200) 
  angle = 0
  startAngle = 180
  endAngle = 270
  color = (255, 255, 255)
  thickness = width
  img=lmdl.draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_2_curve(img_ridge,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  axesLength=(140+distance_between_lane,160+distance_between_lane)

  img=lmdl.draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_2_curve(img_ridge,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_solid_curve_line_two_lane, line width:"+str(width)+"px distance_between_lane:"+str(distance_between_lane)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_white_dashed_curve_line_two_lane(width,distance_between_lane,marker_length,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  center_coordinates = (int(img.shape[1]), int(2*img.shape[0]*0.3)) 
  axesLength = (140, 200) 
  angle = 0
  startAngle = 180
  endAngle = 270
  color = (255, 255, 255)
  thickness = width
  img=lmdl.draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_1_curve(img_ridge,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  axesLength=(140+distance_between_lane,160+distance_between_lane)

  img=lmdl.draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_1_curve(img_ridge,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  process_algorithm(img,img_ridge,output_path,test_case_name)  

  test_case_header="test_white_solid_curve_line_two_lane, line width:"+str(width)+"px distance_between_lane:"+str(distance_between_lane)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_white_solid_curve_line_three_lane(width,distance_between_lane,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  center_coordinates = (int(img.shape[1]), int(2*img.shape[0]*0.3)) 
  axesLength = (50, 100) 
  angle = 0
  startAngle = 180
  endAngle = 270
  color = (255, 255, 255)
  thickness = width
  img=lmdl.draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_2_curve(img_ridge,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  axesLength=(50+distance_between_lane,100+distance_between_lane)

  img=lmdl.draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_2_curve(img_ridge,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)

  axesLength=(50+2*distance_between_lane,100+int(1.9*distance_between_lane))

  img=lmdl.draw_line_2_2_curve(img,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_2_curve(img_ridge,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  process_algorithm(img,img_ridge,output_path,test_case_name)

  test_case_header="test_white_solid_curve_line_three_lane, line width:"+str(width)+"px distance_between_lane:"+str(distance_between_lane)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_white_dashed_curve_line_three_lane(width,distance_between_lane,marker_length,test_case_name):
  output_path,img,img_ridge =init_img(test_case_name)

  center_coordinates = (int(img.shape[1]), int(2*img.shape[0]*0.3)) 
  axesLength = (50, 80) 
  angle = 0
  startAngle = 180
  endAngle = 270
  color = (255, 255, 255)
  thickness = width
  img=lmdl.draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_1_curve(img_ridge,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  axesLength=(50+distance_between_lane,100+distance_between_lane)

  img=lmdl.draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_1_curve(img_ridge,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  
  axesLength=(50+2*distance_between_lane,100+int(1.9*distance_between_lane))

  img=lmdl.draw_line_2_1_curve(img,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,thickness)
  img_ridge=lmdl.draw_line_2_1_curve(img_ridge,marker_length,center_coordinates,axesLength,angle,startAngle,endAngle,color,1)
  
  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_white_dashed_curve_line_three_lane, line width:"+str(width)+"px distance_between_lane:"+str(distance_between_lane)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_lane_split(test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape
  lane_split=lmdl.draw_lane_split()
  scale_percent=img.shape[0]/lane_split.shape[0]
  scale_width=int(lane_split.shape[1]*scale_percent)
  scale_height=int(lane_split.shape[0]*scale_percent)
  lane_split=cv2.resize(lane_split,(scale_width,scale_height),interpolation=cv2.INTER_AREA)
  img[0:lane_split.shape[0],img.shape[1]-lane_split.shape[1]:]=lane_split
  img=lmdl.draw_line_2_2(img,0,int(col/20),5)
  img=lmdl.draw_line_2_1(img,0,int(col/20+col/3),5,40)

  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_lane_split"
  write_md(output_path,test_case_name,test_case_header)
def run_test_lane_merge(test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape
  lane_split=lmdl.draw_lane_merge()
  scale_percent=img.shape[0]/lane_split.shape[0]
  scale_width=int(lane_split.shape[1]*scale_percent)
  scale_height=int(lane_split.shape[0]*scale_percent)
  lane_split=cv2.resize(lane_split,(scale_width,scale_height),interpolation=cv2.INTER_AREA)
  img[0:lane_split.shape[0],img.shape[1]-lane_split.shape[1]:]=lane_split
  img=lmdl.draw_line_2_2(img,0,int(col/20),5)
  img=lmdl.draw_line_2_1(img,0,int(col/20+col/3),5,40)

  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_lane_split"
  write_md(output_path,test_case_name,test_case_header)
def run_test_double_curve_single_solid_line(thickness,test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape

  row,col,_=img.shape
  img=lmdl.draw_line_2_2_double_curve(img,(int(col),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  img_ridge=lmdl.draw_line_2_2_double_curve(img_ridge,(int(col),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  img=lmdl.convert2carview(img)
  img_ridge=lmdl.convert2carview(img_ridge)
  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_double_curve_lane_single_line, width:"+str(thickness)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_double_curve_lane_dashed_line(thickness,test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape

  row,col,_=img.shape
  img=lmdl.draw_line_2_1_double_curve(img,60,(int(col),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  img_ridge=lmdl.draw_line_2_1_double_curve(img_ridge,60,(int(col),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  img=lmdl.convert2carview(img)
  img_ridge=lmdl.convert2carview(img_ridge)
  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_double_curve_lane_dashed_line, width:"+str(thickness)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_double_curve_solid_two_line(thickness,test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape

  row,col,_=img.shape
  img=lmdl.draw_line_2_2_double_curve(img,(int(col*1.2),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  img=lmdl.draw_line_2_2_double_curve(img,(int(col*0.6),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  
  img_ridge=lmdl.draw_line_2_2_double_curve(img_ridge,(int(col*1.2),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  img_ridge=lmdl.draw_line_2_2_double_curve(img_ridge,(int(col*0.6),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  
  
  img_ridge=lmdl.convert2carview(img_ridge)
  img=lmdl.convert2carview(img)
  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_double_curve_lane_single_line, width:"+str(thickness)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_double_curve_lane_dashed_two_line(thickness,test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape

  row,col,_=img.shape
  img=lmdl.draw_line_2_1_double_curve(img,60,(int(col*1.2),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  img=lmdl.draw_line_2_1_double_curve(img,60,(int(col*0.6),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  
  img_ridge=lmdl.draw_line_2_1_double_curve(img_ridge,60,(int(col*1.2),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  img_ridge=lmdl.draw_line_2_1_double_curve(img_ridge,60,(int(col*0.6),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  
  img=lmdl.convert2carview(img)
  img_ridge=lmdl.convert2carview(img_ridge)
  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_double_curve_lane_dashed_two_line, width:"+str(thickness)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_double_curve_solid_three_line(thickness,test_case_name):
  output_path, img, img_ridge =init_img(test_case_name)
  row,col,_=img.shape

  row,col,_=img.shape
  img=lmdl.draw_line_2_2_double_curve(img,(int(col*1.2),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  img=lmdl.draw_line_2_1_double_curve(img,60,(int(col*0.9),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  img=lmdl.draw_line_2_2_double_curve(img,(int(col*0.6),int(row*0.3)),(200,250),0,180,250,(255,255,255),thickness)
  
  img_ridge=lmdl.draw_line_2_2_double_curve(img_ridge,(int(col*1.2),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  img_ridge=lmdl.draw_line_2_1_double_curve(img_ridge,60,(int(col*0.9),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  img_ridge=lmdl.draw_line_2_2_double_curve(img_ridge,(int(col*0.6),int(row*0.3)),(200,250),0,180,250,(255,255,255),1)
  
  
  img_ridge=lmdl.convert2carview(img_ridge)
  img=lmdl.convert2carview(img)
  process_algorithm(img,img_ridge,output_path,test_case_name)  
  test_case_header="test_double_curve_solid_three_line, width:"+str(thickness)+"px"
  write_md(output_path,test_case_name,test_case_header)
def run_test_from_real_image():
  test_case_name="out20"
  img_path="/home/chinhnt24/ridge-lane-detection/src_python/image/input/long/merge1/"+test_case_name+".png"
  output_path="/home/chinhnt24/ridge-lane-detection/src_python/image/input/long/merge1/"+test_case_name
  if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir(output_path+"/data/")

  img=cv2.imread(img_path,1)
  img_ridge=np.array(img.shape)
  process_algorithm(img,img_ridge,output_path+"/data/",test_case_name)
  test_case_header=test_case_name
  write_md(output_path+"/data/",test_case_name,test_case_header)

def summary_test_from_real_image():
  output_path="/home/chinhnt24/ridge-lane-detection/src_python/image/input/long/exit1/"
  print("output_path",output_path)
  files=[]
  # filewriter=open(output_path+"/summary.md","w")
  for r,d,f in os.walk(output_path):
    for file in f:
      if '0_input.png' in file:
        input_path=os.path.join(r,file)
        output_ridge=os.path.join(r,"0_ridgeness_output.png")
        output_vis=os.path.join(r,"summary view.png")
        input_img=cv2.imread(input_path,1)
        output_img=cv2.imread(output_ridge,1)
        vis = np.concatenate((input_img, output_img), axis=1)
        cv2.imwrite(output_vis,vis)
        # cv2.imshow("vis",vis)
        # cv2.waitKey(0)
        # print(input_path,output_ridge)
        # print("\n\n")
  files.sort()
  # print(files)
  # filewriter.write("# Summary  \n")
  # for file in files:
  #   data="* Test Case"+file+"  \n["+file+"]("+file+")  \n"
  #   filewriter.write(data)
  # filewriter.close()
def summary_md_test_cases():
  output_path=os.getcwd()+"/"
  print("output_path",output_path)
  files=[]
  filewriter=open(output_path+"/summary.md","w")
  for r,d,f in os.walk(output_path):
    for file in f:
      if '.md' in file and "summary.md" not in file:
        files.append(os.path.join(r,file).replace(output_path,""))
  files.sort()
  # print(files)
  filewriter.write("# Summary  \n")
  for file in files:
    data="* Test Case"+file+"  \n["+file+"]("+file+")  \n"
    filewriter.write(data)
  filewriter.close()
# run_test_from_real_image()
summary_test_from_real_image()
# run_test_white_solid_line(1,"TC01")
# run_test_white_solid_line(2,"TC02")
# run_test_white_solid_line(5,"TC03")
# run_test_white_solid_line(10,"TC04")
# run_test_white_solid_line(20,"TC05")
run_test_white_solid_line(50,"TC06")
# run_test_white_dash_line(1,60,"TC07")
# run_test_white_dash_line(2,60,"TC08")
# run_test_white_dash_line(5,60,"TC09")
# run_test_white_dash_line(10,60,"TC10")
# run_test_white_dash_line(20,60,"TC11")
# run_test_white_dash_line(50,60,"TC12")
# run_test_white_solid_line_two_lane(1,300,"TC13")
# run_test_white_solid_line_two_lane(2,300,"TC14")
# run_test_white_solid_line_two_lane(5,300,"TC15")
# run_test_white_solid_line_two_lane(10,300,"TC16")
# run_test_white_solid_line_two_lane(20,300,"TC17")
# run_test_white_solid_line_two_lane(50,300,"TC18")
# run_test_white_dashed_line_two_lane(1,300,"TC19")
# run_test_white_dashed_line_two_lane(2,300,"TC20")
# run_test_white_dashed_line_two_lane(5,300,"TC21")
# run_test_white_dashed_line_two_lane(10,300,"TC22")
# run_test_white_dashed_line_two_lane(20,300,"TC23")
# run_test_white_dashed_line_two_lane(50,300,"TC24")
# run_test_white_solid_line_three_lane(1,200,"TC25")
# run_test_white_solid_line_three_lane(2,200,"TC26")
# run_test_white_solid_line_three_lane(5,200,"TC27")
# run_test_white_solid_line_three_lane(10,200,"TC28")
# run_test_white_solid_line_three_lane(20,200,"TC29")
# run_test_white_dashed_line_three_lane(1,200,"TC30")
# run_test_white_dashed_line_three_lane(2,200,"TC31")
# run_test_white_dashed_line_three_lane(5,200,"TC32")
# run_test_white_dashed_line_three_lane(10,200,"TC33")
# run_test_white_dashed_line_three_lane(20,200,"TC34")
# run_test_white_solid_curve_line(1,"TC35")
# run_test_white_solid_curve_line(2,"TC36")
# run_test_white_solid_curve_line(5,"TC37")
# run_test_white_solid_curve_line(10,"TC38")
# run_test_white_solid_curve_line(20,"TC39")
# run_test_white_solid_curve_line(50,"TC40")
# run_test_white_dashed_curve_line(1,40,"TC41")
# run_test_white_dashed_curve_line(2,40,"TC42")
# run_test_white_dashed_curve_line(5,40,"TC43")
# run_test_white_dashed_curve_line(10,40,"TC44")
# run_test_white_dashed_curve_line(20,40,"TC45")
# run_test_white_dashed_curve_line(50,80,"TC46")
# run_test_white_solid_curve_line_two_lane(1,300,"TC47")
# run_test_white_solid_curve_line_two_lane(2,300,"TC48")
# run_test_white_solid_curve_line_two_lane(5,300,"TC49")
# run_test_white_solid_curve_line_two_lane(10,300,"TC50")
# run_test_white_solid_curve_line_two_lane(20,300,"TC51")
# run_test_white_solid_curve_line_two_lane(50,300,"TC52")
# run_test_white_dashed_curve_line_two_lane(1,300,60,"TC53")
# run_test_white_dashed_curve_line_two_lane(2,300,60,"TC54")
# run_test_white_dashed_curve_line_two_lane(5,300,60,"TC55")
# run_test_white_dashed_curve_line_two_lane(10,300,60,"TC56")
# run_test_white_dashed_curve_line_two_lane(20,300,60,"TC57")
# run_test_white_dashed_curve_line_two_lane(50,300,60,"TC58")
# run_test_white_solid_curve_line_three_lane(1,200,"TC59")
# run_test_white_solid_curve_line_three_lane(2,200,"TC60")
# run_test_white_solid_curve_line_three_lane(5,200,"TC61")
# run_test_white_solid_curve_line_three_lane(10,200,"TC62")
# run_test_white_solid_curve_line_three_lane(20,200,"TC63")
# run_test_white_dashed_curve_line_three_lane(1,200,60,"TC64")
# run_test_white_dashed_curve_line_three_lane(2,200,60,"TC65")
# run_test_white_dashed_curve_line_three_lane(5,200,60,"TC66")
# run_test_white_dashed_curve_line_three_lane(10,200,60,"TC67")
# run_test_white_dashed_curve_line_three_lane(20,200,60,"TC68")
# run_test_lane_split("TC69")
# run_test_lane_merge("TC70")
# run_test_double_curve_lane_single_line(1,"TC71")
# run_test_double_curve_lane_single_line(2,"TC72")
# run_test_double_curve_lane_single_line(5,"TC73")
# run_test_double_curve_lane_single_line(10,"TC74")
# run_test_double_curve_lane_single_line(20,"TC75")
# run_test_double_curve_lane_dashed_line(1,"TC76")
# run_test_double_curve_lane_dashed_line(2,"TC77")
# run_test_double_curve_lane_dashed_line(5,"TC78")
# run_test_double_curve_lane_dashed_line(10,"TC79")
# run_test_double_curve_lane_dashed_line(20,"TC80")
# run_test_double_curve_solid_two_line(1,"TC81")
# run_test_double_curve_solid_two_line(2,"TC82")
# run_test_double_curve_solid_two_line(5,"TC83")
# run_test_double_curve_solid_two_line(10,"TC84")
# run_test_double_curve_solid_two_line(20,"TC85")
# run_test_double_curve_lane_dashed_two_line(1,"TC86")
# run_test_double_curve_lane_dashed_two_line(2,"TC87")
# run_test_double_curve_lane_dashed_two_line(5,"TC88")
# run_test_double_curve_lane_dashed_two_line(10,"TC89")
# run_test_double_curve_lane_dashed_two_line(20,"TC90")
# run_test_double_curve_solid_three_line(1,"TC91")
# run_test_double_curve_solid_three_line(2,"TC92")
# run_test_double_curve_solid_three_line(5,"TC93")
# run_test_double_curve_solid_three_line(10,"TC94")
# run_test_double_curve_solid_three_line(20,"TC95")



# summary_md_test_cases()





















