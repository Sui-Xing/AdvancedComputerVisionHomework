#-*- codeing = utf-8 -*-
#@Time : 2020/12/30 11:43
#@Auther : 李昊冉
#@File : realOne.py
#@Software : PyCharm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# plt显示灰度图片
def plt_show(img):
    plt.imshow(img,cmap='gray')
    plt.show

# 读取一个文件夹下的所有图片，输入参数是文件名，返回文件地址列表

def read_directory(directory_name):
    faces_addr=[]
    for filename in os.listdir(directory_name):
        faces_addr.append(directory_name+"/"+filename)
    return faces_addr

# 读取所有人脸文件夹，保存图像地址在列表中
faces=[]
numm=0
for i in range(1,44):
    # numm=numm+1
    # print("="*5,numm,"="*5)
    faces_addr=read_directory('./att_faces/s'+str(i))
    for addr in faces_addr:
        faces.append(addr)

# 读取图片数据，生成列表标签
images=[]
labels=[]
for index,face in enumerate(faces):
    image=cv2.imread(face,0)
    images.append(image)
    labels.append(int(index/10+1))
# print(len(labels))
# print(len(images))
# print(type(images[0]))
# print(labels)

# 画出最后两组人脸图像
# #创建画布和子图对象
# fig, axes = plt.subplots(4,10
#                        ,figsize=(15,4)
#                        ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
#                        )
# #填充图像
# for i, ax in enumerate(axes.flat):
#     ax.imshow(images[i+390],cmap="gray") #选择色彩的模式
# -----------------PCA降低维度------------------
# 图像数据转换成特征矩阵
image_data=[]
for image in images:
    data=image.flatten()
    image_data.append(data)
# print(image_data[0].shape)

# 转换为numpy数组
X = np.array(image_data)
y = np.array(labels)



# 导入sklearn的pca模块
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 画出特征矩阵
import pandas as pd
data = pd.DataFrame(X)
data.head()

# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2)

# 训练PCA模型
pca=PCA(n_components=100)
pca.fit(x_train)
PCA(copy=True, iterated_power='auto', n_components=100, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)

# 返回测试集和训练集降维后的数据集
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
# print(x_train_pca.shape)
# print(x_test_pca.shape)

V = pca.components_
V.shape


# # 100个特征脸
# #创建画布和子图对象
# fig, axes = plt.subplots(10,10
#                        ,figsize=(15,15)
#                        ,subplot_kw = {"xticks":[],"yticks":[]} #不要显示坐标轴
#                        )
# #填充图像
# for i, ax in enumerate(axes.flat):
#     ax.imshow(V[i,:].reshape(112,92),cmap="gray") #选择色彩的模式


# 模型创建与训练
model = cv2.face.EigenFaceRecognizer_create()
model.train(x_train,y_train)

# 预测
res = model.predict(x_test[0])
# print(res)

# 测试数据集的准确率
ress = []
true = 0
for i in range(len(y_test)):
    res = model.predict(x_test[i])
#     print(res[0])
    if y_test[i] == res[0]:
        true = true+1
    else:
        print(i)

print('测试集识别准确率：%.2f'% (true/len(y_test)))


# 降维
pca=PCA(n_components=120)
pca.fit(X)
X = pca.transform(X)

# 将所有数据都用作训练集
# 模型创建与训练
model = cv2.face.EigenFaceRecognizer_create()
model.train(X,y)

# plt显示彩色图片
def plt_show0(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()

# 输入图片识别
img = cv2.imread('./att_faces/test6.jpg')
# plt_show0(img)
# print(img.shape)


def getImg(dir):
    # 加载人脸检测模型
    face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml ')

    print('./att_faces/%s'%(dir))
    img = cv2.imread('./att_faces/%s'%(dir))
    plt_show0(img)
    # 复制图像灰度处理
    img_ = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(np.shape(img))
    # 检测人脸获取人脸区域

    faces = face_engine.detectMultiScale(gray)
    print("faces.len=",len(faces))
    # 将检测出的人脸可视化
    for (x, y, w, h) in faces:

        cv2.rectangle(img_, (x, y), (x + w, y + h), (0, 0, 255), 3)
        plt_show0(img_)

        w2=int(w*1.2)
        h2 = int(w2 * 1.217)
        print(w,h,h2)

        # 图像像素不能太低
        face = img[y-int(h/4):y-int(h/4) + h2, x-int(w/10):x-int(w/10) + w2]

        face=cv2.resize(face,(92,112))
        plt_show0(face)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)


        print(np.shape(face))

        return face


testTT=getImg('tt7.jpg')
# 灰度处理
img = cv2.imread('./att_faces/test6.jpg',0)
# print(img)
# plt_show(img)


imgs = []
imgs.append(testTT)
# 特征矩阵
image_data = []
for img in imgs:
    data = img.flatten()
    image_data.append(data)

test = np.array(image_data)


test.shape



# 用训练好的pca模型给图片降维
test = pca.transform(test)

res = model.predict(test)

print('人脸识别结果：',res[0])

