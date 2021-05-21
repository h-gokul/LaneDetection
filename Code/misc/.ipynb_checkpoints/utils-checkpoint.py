import cv2
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def StitchImages(K,kC, datapath = "./Project2_Dataset2/Project2_Dataset2/data_1/data/*png"):
    im_paths = sorted(glob.glob(datapath))
    imgs = []
    for im_path in im_paths:
        frame = cv2.imread(im_path)
        frame =  cv2.undistort(frame, K, kC, None, K)
        imgs.append(rgb(frame))
    return imgs

def vidRead(K,kC, path ="./Project2_Dataset1/Night Drive - 2689.mp4" ):
    imgs = []
    cap = cv2.VideoCapture(path)
    while(True):
        ret, frame = cap.read()
        if ret:
            frame =  cv2.undistort(frame, K, kC, None, K)
            imgs.append(rgb(frame))
        else:
            break
    cap.release()
    return imgs

def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

def rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class MovingAverage:
    """
    This code was referenced from the FaceSwap project submission for course CMSC733 done by me and my team-mate Sakshi kakde(also in ENPM673-2021)    
    """
    def __init__(self, window_size):

        self.window_size_ = window_size
        self.values_ = []
        self.average_ = 0
#         self.weight_ = weight

    def add_values(self, points):

        if len(self.values_) < self.window_size_:
            self.values_.append(points)

        else:
            self.values_.pop(0)
            self.values_.append(points)

    def getmean(self):
        values = self.values_
        values = np.array(values)
        sum = np.sum(values, axis = 0)
        # sum = 0
        # for i in range(len(self.values_)):
        #     sum = sum + self.weight_[i] * markers[i,:]

        # self.average_ = (sum / np.sum(self.weight_)).astype(int)
        
        self.average_ = (sum / len(self.values_)).astype(np.int32)
        
        if len(self.values_) < self.window_size_:
            return values[-1]
        else:
            return self.average_

    def __len__(self): 
        return len(self.values_)
