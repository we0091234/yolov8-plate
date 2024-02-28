import onnxruntime
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import time

def allFilePath(rootPath,allFIleList):  #遍历文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)


def my_letter_box(img,size=(640,640)):  #对应yolo中的letterbox前处理
    h,w,c = img.shape
    r = min(size[0]/h,size[1]/w)
    new_h,new_w = int(h*r),int(w*r)
    top = int((size[0]-new_h)/2)
    left = int((size[1]-new_w)/2)
    
    bottom = size[0]-new_h-top
    right = size[1]-new_w-left
    img_resize = cv2.resize(img,(new_w,new_h))
    img = cv2.copyMakeBorder(img_resize,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def xywh2xyxy(boxes):   #xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    xywh =copy.deepcopy(boxes)
    xywh[:,0]=boxes[:,0]-boxes[:,2]/2
    xywh[:,1]=boxes[:,1]-boxes[:,3]/2
    xywh[:,2]=boxes[:,0]+boxes[:,2]/2
    xywh[:,3]=boxes[:,1]+boxes[:,3]/2
    return xywh
 
def my_nms(boxes,iou_thresh):         #自己实现的nms
    index = np.argsort(boxes[:,4])[::-1]
    keep = []
    while index.size >0:
        i = index[0]
        keep.append(i)
        x1=np.maximum(boxes[i,0],boxes[index[1:],0])
        y1=np.maximum(boxes[i,1],boxes[index[1:],1])
        x2=np.minimum(boxes[i,2],boxes[index[1:],2])
        y2=np.minimum(boxes[i,3],boxes[index[1:],3])
        
        w = np.maximum(0,x2-x1)
        h = np.maximum(0,y2-y1)

        inter_area = w*h
        union_area = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])+(boxes[index[1:],2]-boxes[index[1:],0])*(boxes[index[1:],3]-boxes[index[1:],1])
        iou = inter_area/(union_area-inter_area)
        idx = np.where(iou<=iou_thresh)[0]
        index = index[idx+1]
    return keep

def restore_box(boxes,r,left,top):  #返回原图上面的坐标
    boxes[:,[0,2]]-=left
    boxes[:,[1,3]]-=top

    boxes[:,[0,2]]/=r
    boxes[:,[1,3]]/=r
    return boxes

def detect_pre_precessing(img,img_size):  #检测前处理
    img,r,left,top=my_letter_box(img,img_size)
    # cv2.imwrite("1.jpg",img)
    img =img[:,:,::-1].transpose(2,0,1).copy().astype(np.float32)
    img=img/255
    img=img.reshape(1,*img.shape)
    return img,r,left,top

def post_precessing(dets,r,left,top,conf_thresh=0.4,iou_thresh=0.5,num_class=8):#检测后处理
    score= np.max(dets[:,:,4:4+num_class],axis=-1,keepdims=True)
    new_dets = np.concatenate((dets,score),axis=-1)
    choice = new_dets[:,:,-1]>conf_thresh
    new_dets=new_dets[choice]
    score=score[choice]
    # dets[:,5:5+num_class]*=dets[:,4:5]
    box = new_dets[:,:4]
    boxes = xywh2xyxy(box)
    # score= np.max(dets[:,5:5+num_class],axis=-1,keepdims=True)
    index = np.argmax(new_dets[:,4:4+num_class],axis=-1).reshape(-1,1)
    output = np.concatenate((boxes,score,index),axis=1) 
    reserve_=my_nms(output,iou_thresh) 
    output=output[reserve_] 
    output = restore_box(output,r,left,top)
    return output

def Detect(img,session_detect,r,left,top,num_class=2):
    y_onnx = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img})[0]
    y_onnx = y_onnx.transpose((0,2,1))
    outputs = post_precessing(y_onnx,r,left,top,num_class=num_class)
    detect_dict =[]
    for result in outputs:
        result_dict = {}
        rect=[int(result[0]),int(result[1]),int(result[2]),int(result[3])]
        label = int(result[5])
        obj_score=result[4]
        result_dict["rect"]=rect
        result_dict["label"]=label
        result_dict["obj_score"]=round(obj_score,4)
        detect_dict.append(result_dict)
        
    return detect_dict

def  init_detect_model(model_path):
     session_detect = onnxruntime.InferenceSession(model_path, providers=providers )
     return session_detect





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    first_stage_detect_model_path =r"runs/detect/train2/weights/best.onnx"
    fiest_state_detect_image_size = 640
    image_path = "imgs"
    output='result1'
    opt = parser.parse_args()
    file_list = []
    allFilePath(image_path,file_list)
    providers =  ['CUDAExecutionProvider']
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(0,0,0),(255,255,255),(255,0,255)]
    img_size = (640,640)
    session_detect = init_detect_model(first_stage_detect_model_path)
  
    if not os.path.exists(output):
        os.mkdir(output)
    save_path = output
    count = 0
    begin = time.time()
    for pic_ in file_list:
        count+=1
        print(count,pic_,end=" ")
        img=cv2.imread(pic_)
        img0 = copy.deepcopy(img)
        img,r,left,top = detect_pre_precessing(img,img_size) #检测前处理
        outputs = Detect(img,session_detect,r,left,top,num_class=2) #得到检测结果
        for output in outputs:
            # label =int(output[5])
            rect = output['rect']
            cv2.rectangle(img0,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),clors[output['label']],2)
            
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path,img_name)
        cv2.imwrite(save_img_path,img0)
        # cv2.imwrite('haha.jpg',img0)
        print(len(outputs))
       
    print(f"总共耗时{time.time()-begin} s")
    

        