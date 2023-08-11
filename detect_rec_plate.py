import torch
import cv2
import numpy as np
import argparse
import copy
import time
import os
from ultralytics.nn.tasks import  attempt_load_weights
from plate_recognition.plate_rec import get_plate_result,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from fonts.cv_puttext import cv2ImgAddText

def allFilePath(rootPath,allFIleList):# 读取文件夹内的文件，放到list
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
            
def four_point_transform(image, pts):                       #透视变换得到车牌小图
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
            

def letter_box(img,size=(640,640)):  #yolo 前处理 letter_box操作
    h,w,_=img.shape
    r=min(size[0]/h,size[1]/w)
    new_h,new_w=int(h*r),int(w*r)
    new_img = cv2.resize(img,(new_w,new_h))
    left= int((size[1]-new_w)/2)
    top=int((size[0]-new_h)/2)   
    right = size[1]-left-new_w
    bottom=size[0]-top-new_h 
    img =cv2.copyMakeBorder(new_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def load_model(weights, device):  #加载yolov8 模型
    model = attempt_load_weights(weights,device=device)  # load FP32 model
    return model    

def xywh2xyxy(det):     #xywh转化为xyxy
    y = det.clone()
    y[:,0]=det[:,0]-det[0:,2]/2
    y[:,1]=det[:,1]-det[0:,3]/2
    y[:,2]=det[:,0]+det[0:,2]/2
    y[:,3]=det[:,1]+det[0:,3]/2
    return y

def my_nums(dets,iou_thresh):  #nms操作
    y = dets.clone()
    y_box_score = y[:,:5]
    index = torch.argsort(y_box_score[:,-1],descending=True)
    keep = []
    while index.size()[0]>0:
        i = index[0].item()
        keep.append(i)
        x1=torch.maximum(y_box_score[i,0],y_box_score[index[1:],0])
        y1=torch.maximum(y_box_score[i,1],y_box_score[index[1:],1])
        x2=torch.minimum(y_box_score[i,2],y_box_score[index[1:],2])
        y2=torch.minimum(y_box_score[i,3],y_box_score[index[1:],3])
        zero_=torch.tensor(0).to(device)
        w=torch.maximum(zero_,x2-x1)
        h=torch.maximum(zero_,y2-y1)
        inter_area = w*h
        nuion_area1 =(y_box_score[i,2]-y_box_score[i,0])*(y_box_score[i,3]-y_box_score[i,1]) #计算交集
        union_area2 =(y_box_score[index[1:],2]-y_box_score[index[1:],0])*(y_box_score[index[1:],3]-y_box_score[index[1:],1])#计算并集

        iou = inter_area/(nuion_area1+union_area2-inter_area)#计算iou
        
        idx = torch.where(iou<=iou_thresh)[0]   #保留iou小于iou_thresh的
        index=index[idx+1]
    return keep


def restore_box(dets,r,left,top):  #坐标还原到原图上

    dets[:,[0,2,5,7,9,11]]=dets[:,[0,2,5,7,9,11]]-left
    dets[:,[1,3,6,8,10,12]]= dets[:,[1,3,6,8,10,12]]-top
    dets[:,:4]/=r
    dets[:,5:13]/=r

    return dets
    # pass

def post_processing(prediction,conf,iou_thresh,r,left,top):  #后处理

    prediction = prediction.permute(0,2,1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf  #过滤掉小于conf的框         
    x = prediction[xc]
    if not len(x):
        return []
    boxes = x[:,:4]  #框
    boxes = xywh2xyxy(boxes)  #中心点 宽高 变为 左上 右下两个点
    score,index = torch.max(x[:,4:6],dim=-1,keepdim=True) #找出得分和所属类别
    x = torch.cat((boxes,score,x[:,6:14],index),dim=1)      #重新组合
    
    score = x[:,4]
    keep =my_nums(x,iou_thresh)
    x=x[keep]
    x=restore_box(x,r,left,top)
    return x

def pre_processing(img,opt,device):  #前处理
    img, r,left,top= letter_box(img,(opt.img_size,opt.img_size))
    # print(img.shape)
    img=img[:,:,::-1].transpose((2,0,1)).copy()  #bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img/255.0
    img =img.unsqueeze(0)
    return img ,r,left,top

def det_rec_plate(img,img_ori,detect_model,plate_rec_model):
    result_list=[]
    img,r,left,top = pre_processing(img,opt,device)  #前处理
    predict = detect_model(img)[0]                   
    outputs=post_processing(predict,0.3,0.5,r,left,top) #后处理
    for output in outputs:
        result_dict={}
        output = output.squeeze().cpu().numpy().tolist()
        rect=output[:4]
        rect = [int(x) for x in rect]
        label = output[-1]
        land_marks=np.array(output[5:13],dtype='int64').reshape(4,2)
        roi_img = four_point_transform(img_ori,land_marks)   #透视变换得到车牌小图
        if int(label):        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img=get_split_merge(roi_img)
        plate_number,rec_prob,plate_color,color_conf=get_plate_result(roi_img,device,plate_rec_model,is_color=True)
        
        result_dict['plate_no']=plate_number   #车牌号
        result_dict['plate_color']=plate_color   #车牌颜色
        result_dict['rect']=rect                      #车牌roi区域
        result_dict['detect_conf']=output[4]              #检测区域得分
        result_dict['landmarks']=land_marks.tolist() #车牌角点坐标
        # result_dict['rec_conf']=rec_prob   #每个字符的概率
        result_dict['roi_height']=roi_img.shape[0]  #车牌高度
        # result_dict['plate_color']=plate_color
        # if is_color:
        result_dict['color_conf']=color_conf    #颜色得分
        result_dict['plate_type']=int(label)   #单双层 0单层 1双层
        result_list.append(result_dict)
    return result_list


def draw_result(orgimg,dict_list,is_color=False):   # 车牌结果画出来
    result_str =""
    for result in dict_list:
        rect_area = result['rect']
        
        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=max(0,int(y-padding_h))
        rect_area[2]=min(orgimg.shape[1],int(rect_area[2]+padding_w))
        rect_area[3]=min(orgimg.shape[0],int(rect_area[3]+padding_h))

        height_area = result['roi_height']
        landmarks=result['landmarks']
        result_p = result['plate_no']
        if result['plate_type']==0:#单层
            result_p+=" "+result['plate_color']
        else:                             #双层
            result_p+=" "+result['plate_color']+"双层"
        result_str+=result_p+" "
        for i in range(4):  #关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        
        labelSize = cv2.getTextSize(result_p,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
        if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
            rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.2*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)#画文字框,背景白色
        
        if len(result)>=6:
            orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
            # orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
               
    print(result_str)
    return orgimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights/yolov8-lite-t-plate.pt', help='model.pt path(s)')  #yolov8检测模型
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec_color.pth', help='model.pt path(s)')#车牌字符识别模型  
    parser.add_argument('--image_path', type=str, default=r'imgs', help='source')   #待识别图片路径
    parser.add_argument('--img_size', type=int, default=320, help='inference size (pixels)')  #yolov8 网络模型输入大小
    parser.add_argument('--output', type=str, default='result', help='source')     #结果保存的文件夹
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    opt = parser.parse_args()
    save_path = opt.output                

    if not os.path.exists(save_path): 
        os.mkdir(save_path)
        
    detect_model = load_model(opt.detect_model, device)  #初始化yolov8识别模型
    plate_rec_model=init_model(device,opt.rec_model,is_color=True)      #初始化识别模型
    #算参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("yolov8 detect params: %.2fM,rec params: %.2fM" % (total/1e6,total_1/1e6))
    
    detect_model.eval() 
    # print(detect_model)
    file_list = []
    allFilePath(opt.image_path,file_list)
    count=0
    time_all = 0
    time_begin=time.time()
    for pic_ in file_list:
        print(count,pic_,end=" ")
        time_b = time.time()               #开始时间
        img = cv2.imread(pic_)
        img_ori = copy.deepcopy(img)
        result_list=det_rec_plate(img,img_ori,detect_model,plate_rec_model)
        time_e=time.time()
        ori_img=draw_result(img,result_list)  #将结果画在图上
        img_name = os.path.basename(pic_)  
        save_img_path = os.path.join(save_path,img_name)  #图片保存的路径
        time_gap = time_e-time_b                         #计算单个图片识别耗时
        if count:
            time_all+=time_gap 
        count+=1
        cv2.imwrite(save_img_path,ori_img)               #op
        # print(result_list)
    print(f"sumTime time is {time.time()-time_begin} s, average pic time is {time_all/(len(file_list)-1)}")
     