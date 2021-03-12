#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
# 資料處理套件
import cv2
import csv
import random
import time
import numpy as np
import pandas as pd
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from keras.preprocessing import image
#import seaborn as sns
import os 
import random
import math

# 設定顯示中文字體
from matplotlib.font_manager import FontProperties
#import FontProperties from matplotlib.font_manager 
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

# Keras深度學習模組套件
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import RMSprop
from keras.models import Sequential 
from keras import utils as np_utils 
from keras import backend as K 
from keras import optimizers

# tensorflow深度學習模組套件
from tensorflow.keras import models, layers 
from tensorflow import keras 
import tensorflow as tf


# In[2]:


#圖片尺寸
re_row = 224
re_col = 224

# 設定超參數HyperParameters 
batch_size = 32
epochs = 100

# 檔名設定
file_name = str(epochs)+'_'+str(batch_size)


# In[3]:

#讀取模型
model = tf.keras.models.load_model('./'+file_name+'.h5')
model.summary()


# In[4]:


#讀取測試csv
url = './AI_mango/Test_UploadSheet.csv'
result = pd.read_csv(url)


# In[7]:

######預測######

#讀取每張測試圖片並預測
count=0
for index in range(len(result)):
    img_path = "./AI_mango/Test/"+result["image_id"][index]
    count+=1
    print(str(count)+" "+img_path)
    # 讀入待測試圖像
    img = cv2.imread(img_path)
    res = cv2.resize(img,(re_row,re_col),interpolation=cv2.INTER_LINEAR)
    
    labels = ["不良-乳汁吸附","不良-機械傷害","不良-炭疽病","不良-著色不佳","不良-黑斑病"]

    # 將圖像轉成模型可分析格式(row*col*3, float32)
    x = img_to_array(res)  # Numpy array with shape (row, col, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, row, col, 3)
    x /= 255 # Rescale by 1/255

    #start = time.time() # 啟動計時器
    result2 = model.predict(x) # 對輸入圖像進行推論(預測)
    #finish = time.time() # 結束計時器
    
    #取得預測機率
    pred = result2.argmax(axis=1)[0]
    print(result2)
    for index2 in range(5):
        temp = "D"+str(index2+1)
        #如果機率大於0.15則判斷為真
        if result2[0][index2] >= 0.15:
            #result[temp][index] = 1
            result.loc[index,temp] = 1
        else:
            #result[temp][index] = 0
            result.loc[index,temp] = 0


# In[48]:


#儲存結果csv
result.to_csv('./AI_mango/Test_UploadSheet.csv', index=False)

