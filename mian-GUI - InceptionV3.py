# coding=utf-8
import cv2
#opencv的库
import os, shutil
#
import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.models import load_model
import numpy as np
import sys

font = cv2.FONT_HERSHEY_SIMPLEX
from keras.optimizers import Adam
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import utils
from scipy import misc

#上面的是导入相关的依赖库，包含numpy opencv tensorflow等依赖库

CLASSES = (
'one','ding', 'ugly', 'zhuan', 'qie', 'shi','qiu','bing','ye','cong','dong','si','qi','wang','zhang','san','up','down','no','yu')
#class 类别
model = load_model('model-ResNet50-final.h5')
#加载训练好的模型，修改该模型的名称可以加载不同的模型，model文件夹下面有两个模型

path='test_dataset/10/216859.png'

img = misc.imread(path, mode='RGB')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# self.detection = self.img
#缩放图片到指定的大小
img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

#图片预处理
code = utils.ImageEncode(path)
#图片预测
ret = model.predict(code)
print(ret)
#输入最大相似度的类别
res1 = np.argmax(ret[0, :])
#打印最大相似度的类别
print('result:', CLASSES[res1])
#在图片上绘制出类别相似度
cv2.putText(img, str(float('%.4f' % np.max(ret[0, :])) * 100) + '%', (1, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
            thickness=2, lineType=2)
#在图片上绘制出类别
cv2.putText(img, str(CLASSES[res1]), (1, 160),
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                          thickness=2, lineType=2)
cv2.imshow('out',img)
cv2.waitKey(0)