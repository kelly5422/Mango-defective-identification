# Deep Learning - Speech Recognition報告
## Deep Learning@NTUT, 2020 Fall報告
Taiwanese Speech Recognition using End-to-End Approach


- 學生: 郭靜
- 學號: 108598068

---

## 做法說明
1. 讀取csv標籤
2. 圖片資料增強
3. 獲取訓練圖片與標籤
4. 分配訓練集及測試集比例
5. 定義模型
6. 訓練模型
7. 測試模型

---

## 程式方塊圖與寫法

![](https://i.imgur.com/erB8G2Q.png)





#### 讀取csv標籤
```
for line in reader:
    if flag==0:
        picname = line[0][1:]
        flag=1
    else:
        picname = line[0]
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
```


#### 圖片資料增強參數
```
datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
```

#### 獲取訓練圖片與標籤
```
def get_img(x_train,y_train,start,pic_num):
    batch_x = []
    batch_y = []
    for i in range(start,start+pic_num): 
        batch_list = x_train[start : start+pic_num]
        batch_label = y_train[start : start+pic_num]
        
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
```


#### 分配訓練集及測試集比例
```
def get_a_cell():
    return tf.nn.rnn_cell.GRUCell(num_hidden)

stack = tf.contrib.rnn.MultiRNNCell([get_a_cell() for _ in range(1)], state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
```

#### 定義模型，epochs = 100，batch_size = 12
```
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
```

#### 訓練模型
```
history = model.fit(
    x = x_train , y = y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2,
    callbacks = [CB]
)
```

#### 測試模型
```
y_pred = model.predict(x_test)
predict_label = np.argmax(y_pred,axis=1)
predictions = model.predict_classes(x_test)
print(predict_label)
print(true_label)
```
---

## 討論預測值誤差很大的，是怎麼回事？
1. 模型參數不知道是否符合需求

---

## 如何改進？
1. 多調整幾組參數，保留預測結果較好的參數

---# Mango-defective-identification
