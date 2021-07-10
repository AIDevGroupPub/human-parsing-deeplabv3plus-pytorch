# -*- coding: utf-8 -*-
"""
@author: lhnows

"""
from PIL.Image import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torch.autograd import Variable
import warnings
import time
import cv2

import glob
import os
import sys
import segmentation_models_pytorch as smp
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss

import time

localtime = time.asctime( time.localtime(time.time()) )



# 忽略警告信息
warnings.filterwarnings('ignore')
# cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题        
torch.backends.cudnn.enabled = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)


    
    CLASSES = 2
    model_path = "./model/deeplabv3+.pth"
    Image_Dir = './data/test/'
    
    re_img_size = 512


    Save_Dir = './'+localtime
    infer_save = Save_Dir +'/infer_save'


    os.mkdir(Save_Dir)
    os.mkdir(infer_save)
    


    # 准备数据
    full_img_list = glob.glob(Image_Dir+'*.jpg')
    
   

   

    print(full_img_list[0])
 


   

    # 定义网络模型
    model = smp.DeepLabV3Plus(
             encoder_name="resnet18",# efficient net  b7
             encoder_weights="imagenet",
             in_channels=3,
             classes=2,#2
    )
    model.to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
    


    as_tensor = T.Compose([
            T.ToTensor(),
        ])
    #for index in tqdm(range(2)):
    for index in tqdm(range(len(full_img_list))):
        
        image_in=cv2.imread(full_img_list[index], cv2.IMREAD_UNCHANGED)
       
        height = image_in.shape[0]
        width = image_in.shape[1]
        image_in = cv2.resize(image_in, (re_img_size, re_img_size))
        if image_in.ndim == 2:    #2维度表示长宽
            image_in = cv2.cvtColor(image_in, cv2.COLOR_GRAY2BGR)
            
        
        #print(image_in)
        rgb_image = np.ascontiguousarray(image_in)
       

        rgb_image =  as_tensor(rgb_image)
       
        #print(image_in)

        rgb_image = Variable(torch.unsqueeze(rgb_image, dim=0).float(), requires_grad=False)
       
       
        #rgb_image = rgb_image.to(device=device, dtype=torch.float32)
        rgb_image =rgb_image.cuda()

        output = model(rgb_image )
        
        
        output = output.argmax(1)
        output = output.byte() #.numpy()
        outputarray = output.cpu().numpy().squeeze(0) #.transpose((1,2,0))
        # print(full_img_list[index])
        


        save_img_name = infer_save + '/' + os.path.splitext(os.path.basename(full_img_list[index]))[0] +'.png'

        outputarray = outputarray *255
        # print("heiht width = " + str(height) +','+str(width))
        outputarray = cv2.resize(outputarray, (width,height))


        cv2.imwrite(save_img_name,outputarray)
        #torchvision.utils.save_image(output, save_img_name)
        
    