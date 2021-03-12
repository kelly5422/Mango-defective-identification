#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


#圖片尺寸
re_row = 224
re_col = 224
#限制張數 (-1未限制
total=-1


# In[39]:


csvfile = open('./AI_mango/train.csv',"r",encoding="utf-8")
reader = csv.reader(csvfile)

label = {"不良-乳汁吸附":"0","不良-機械傷害":"1","不良-炭疽病":"2","不良-著色不佳":"3","不良-黑斑病":"4"}

# 轉換圖片的標籤
X = []
Y = []

num=0 #計數
flag=0 

# 讀取csv標籤
for line in reader:
    #排除第一筆亂碼
    if flag==0:
        picname = line[0][1:]
        flag=1
    else:
        picname = line[0]
    #讀取每一筆的標籤與位置
    for index in range(1,len(line),5):
        if(len(line[index]) == 0):
            break
        tmp = []
        x=line[index].replace('.0','')
        y=line[index+1].replace('.0','')
        x2=line[index+2].replace('.0','')
        y2=line[index+3].replace('.0','')
        tmp.append(picname)
        tmp.append(x)
        tmp.append(y)
        tmp.append(x2)
        tmp.append(y2)
        X.append(tmp)    
        Y.append(label[line[index+4]])
        num+=1
    if num>=total and total != -1:
        break
        
csvfile.close()

picnum = len(X)
print("芒果圖片數量: ",picnum)


# In[40]:

######資料處理######

#圖片資料增強
datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
#獲取訓練圖片與標籤
def get_img(x_train,y_train,start,pic_num):
    batch_x = []
    batch_y = []
    for i in range(start,start+pic_num): 
        batch_list = x_train[start : start+pic_num]
        batch_label = y_train[start : start+pic_num]
        #讀取圖片與標籤
        for j in range(len(batch_list)):
            img = cv2.imread("./AI_mango/Train/" + batch_list[j][0])
            x=int(batch_list[j][1])
            y=int(batch_list[j][2])
            x2=x+int(batch_list[j][3])
            y2=y+int(batch_list[j][4])
            img_roi = img[y:y2,x:x2,:].copy()
            res = cv2.resize(img,(re_row,re_col),interpolation=cv2.INTER_LINEAR)
            res = img_to_array(res)

            batch_x.append(res)
            batch_y.append(batch_label[j])

        print("Done")
        return batch_x, batch_y


# In[41]:


#圖片開始編號
start=0
#一次讀取之數量
pic_num = 500


# In[42]:


#獲取讀片
X2, Y2 = get_img(X,Y,start,pic_num)


# In[43]:


#保留原本標籤
Y_label_org = Y2

#轉換numpy及浮點數格式
X2 = np.array(X2)
for j in range(len(X2)):
    X2[j] = X2[j].astype('float32')
#轉換numpy與浮點數格式
Y2 = np.array(Y2)
Y2 = tf.strings.to_number(Y2, out_type=tf.float32)
#將標籤[1]轉換成[1,0,0,0,0]
Y2 = np_utils.to_categorical(Y2, num_classes = 5)

# 分配訓練集及測試集比例
x_test = X2[int((start+pic_num)*0.8):]
y_test = Y2[int((start+pic_num)*0.8):]
x_train = X2[:int((start+pic_num)*0.8)]
y_train = Y2[:int((start+pic_num)*0.8)]

y_train_label = [0.,0.,0.,0.,0.]
y_test_label = [0.,0.,0.,0.,0.]

#統計標籤數量
for i in range(0,len(y_train)):
    y_train_label = y_train[i] + y_train_label
for i in range(0,len(y_test)):
    y_test_label = y_test[i] + y_test_label

print(y_train_label)
print(y_test_label)


# In[44]:


# 導入圖像增強參數
datagen.fit(x_train)
x_train = x_train/255
x_test = x_test/255


# In[45]:

######訓練流程######

# 建立深度學習CNN Model
model = tf.keras.Sequential()

model.add(layers.Conv2D(8,(3,3),
                 strides=(1,1),
                 input_shape=(re_row, re_col, 3),
                 padding='valid',
                 activation='relu',
                 ))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
model.add(layers.Conv2D(16,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 ))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
model.add(layers.Conv2D(32,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 ))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
model.add(layers.Conv2D(64,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 ))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))
model.add(layers.Conv2D(128,(3,3),
                 strides=(1,1),
                 padding='valid',
                 activation='relu',
                 ))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=None))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dense(32,activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(5,activation='softmax'))

model.summary()


# In[46]:


"""history = model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
"""

adam = tf.keras.optimizers.Adam(lr=5)
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])

# 設定超參數HyperParameters 
batch_size = 32
epochs = 100

# 檔名設定
file_name = str(epochs)+'_'+str(batch_size)

# 加入EarlyStopping以及Tensorboard等回調函數
CB = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
TB = keras.callbacks.TensorBoard(log_dir='./log'+"_"+file_name, histogram_freq=1)


# In[47]:

#訓練模型
history = model.fit(
    x = x_train , y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2,
    callbacks = [CB]
)


# In[48]:


#顯示訓練成果
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)


# In[49]:

######預測######

# 測試集標籤預測
y_pred = model.predict(x_test)

# 整體準確度
count = 0
for i in range(len(y_pred)):
    if(np.argmax(y_pred[i]) == np.argmax(y_test[i])): #argmax找出機率最大值
        count += 1
score = count/len(y_pred)
print('正確率為:%.2f%s' % (score*100,'%'))

# 模型預測後的標籤
predict_label = np.argmax(y_pred,axis=1)


# 模型原標籤
true_label = Y_label_org[int(pic_num*0.8):]
true_label = np.array(true_label)


# 模型預測後的標籤
predictions = model.predict_classes(x_test)
print(predict_label)
print(true_label)

pd.crosstab(true_label,predict_label,rownames=['實際值'],colnames=['預測值'])


# In[50]:


# 儲存模型相關參數
model.save('./'+file_name+'.h5')


# In[ ]:




