# Chinese-Character-Recognition-Demo


A Chinese characters recognition demo repository based on convolutional recurrent networks, specifically ResNet50.

Download the dataset here:  http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

Due to personal reasons, the project trains a model on a quite small dataset (20 classes with 60 pictures for each cluster).

# For environment setting:
 
1.Install Anaconda，run Anaconda Prompt

2.conda create -n tensorflow python=3.5

3.activate tensorflow 

4.run

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.9

5.run

    pip install scipy
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple h5py
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Pillow
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple psutil
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple easydict
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras==2.1.0

# For training, run

    python train-resnet.py
    pyhton train-inception.py

# For testing, run

    python main-GUI.py
    
    
References:

https://zhuanlan.zhihu.com/p/24698483?refer=burness-DL

https://github.com/soloice/Chinese-Character-Recognition

https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec

https://github.com/AmemiyaYuko/HandwrittenChineseCharacterRecognition
