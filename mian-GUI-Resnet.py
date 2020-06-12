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


#可视化界面初始化
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        #初始化界面的相关参数
        # self.face_recognition = face.Recognition()
        self.timer_camera = QtCore.QTimer()#定时器
        self.timer_camera_capture = QtCore.QTimer()#定时器
        self.cap = cv2.VideoCapture()#打开摄像头参数，不过这个没有用到
        self.CAM_NUM = 0#摄像头的num   也没有用到
        self.set_ui()#ui初始化，就是界面
        self.slot_init()#槽函数初始化
        self.__flag_work = 0#标志位
        self.x = 0

    def set_ui(self):
        #界面上的按钮初始化
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        #打开图片按钮
        self.pushButton = QtWidgets.QPushButton(u'打开图片')
        # self.addface = QtWidgets.QPushButton(u'建库')
        # self.captureface = QtWidgets.QPushButton(u'采集人脸')
        # self.saveface = QtWidgets.QPushButton(u'保存人脸')
        #打开图片的大小
        self.pushButton.setMinimumHeight(50)
        # self.addface.setMinimumHeight(50)
        # self.captureface.setMinimumHeight(50)
        # self.saveface.setMinimumHeight(50)
        #编辑框的位置
        self.lineEdit = QtWidgets.QLineEdit(self)  # 创建 QLineEdit
        # self.lineEdit.textChanged.connect(self.text_changed)
        #编辑框的大小
        self.lineEdit.setMinimumHeight(50)
        #编辑框的位置
        # self.opencamera.move(10, 30)
        # self.captureface.move(10, 50)
        self.lineEdit.move(15, 350)

        # 信息显示
        #显示加载的图片的控件
        self.label = QtWidgets.QLabel()
        # self.label_move = QtWidgets.QLabel()
        #设置edit控件的大小
        self.lineEdit.setFixedSize(100, 30)
        #设置显示图片控件的大小
        self.label.setFixedSize(641, 481)
        self.label.setAutoFillBackground(False)

        self.__layout_fun_button.addWidget(self.pushButton)
        # self.__layout_fun_button.addWidget(self.addface)
        # self.__layout_fun_button.addWidget(self.captureface)
        # self.__layout_fun_button.addWidget(self.saveface)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label)

        self.setLayout(self.__layout_main)
        # self.label_move.raise_()
        self.setWindowTitle(u'汉字分类')

    def slot_init(self):
        #槽函数初始化按钮的链接
        self.pushButton.clicked.connect(self.button_open_image_click)
        # self.addface.clicked.connect(self.button_add_face_click)
        # self.timer_camera.timeout.connect(self.show_camera)
        # self.timer_camera_capture.timeout.connect(self.capture_camera)
        # self.captureface.clicked.connect(self.button_capture_face_click)
        # self.saveface.clicked.connect(self.save_face_click)
        #打开图片按钮响应事件
    def button_open_image_click(self):
        #清空显示界面
        self.label.clear()
        #清空编辑框的内容
        self.lineEdit.clear()
        #打开图片
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        #获取图片的路径
        self.img = misc.imread(os.path.expanduser(imgName), mode='RGB')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # self.detection = self.img
        #缩放图片到指定的大小
        self.img = cv2.resize(self.img, (640, 480), interpolation=cv2.INTER_AREA)
        #判断图片是否是空
        if self.img is None:
            return None
        #图片预处理
        code = utils.ImageEncode(imgName)
        #图片预测
        ret = model.predict(code)
        print(ret)
        #输入最大相似度的类别
        res1 = np.argmax(ret[0, :])
        #打印最大相似度的类别
        print('result:', CLASSES[res1])
        #在图片上绘制出类别相似度
        cv2.putText(self.img, str(float('%.4f' % np.max(ret[0, :])) * 100) + '%', (1, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                    thickness=2, lineType=2)
        #在图片上绘制出类别
        cv2.putText(self.img, str(CLASSES[res1]), (1, 160),
                                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                  thickness=2, lineType=2)
        #颜色通道变换
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2BGRA)
        #图片格式转换成界面接受的格式
        self.QtImg = QtGui.QImage(self.img_rgb.data, self.img_rgb.shape[1], self.img_rgb.shape[0],
                                  QtGui.QImage.Format_RGB32)
        # 显示图片到label中;
        # self.label.resize(QtCore.QSize(self.img_rgb.shape[1], self.img_rgb.shape[0]))
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        print(CLASSES[res1])
        #编辑框输出类别
        self.lineEdit.setText(CLASSES[res1])

    def closeEvent(self, event):
        #关闭程序的按钮
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        #提示是否关闭
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
        #点击确认后关闭程序
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    #程序入口
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
