import cv2
from misc.utils import *
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

def vidRead(path = '../Data/Project2_Dataset1/Night Drive - 2689.mp4'):
    imgs = []
    cap = cv2.VideoCapture(path)
    while(True):
        ret, frame = cap.read()
        if ret:
#             frame =  cv2.undistort(frame, mtx, dist, None, mtx)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs.append(frame)
        else:
            break
    cap.release()
    
    return imgs

def histogram(im_flat):
    bins = 256
    h = np.zeros(bins)
    for i in im_flat:
        h[i] +=1
    return h

def cumulate_distribution(A):
    
    sum_ = A[0]
    c = [sum_]
    for i in range(0,len(A)-1):
        sum_+=A[i+1]
        c.append(sum_)
    return np.array(c)

def normalize(c):
    c = np.array(c)
    return ((c - c.min()) * 255) / (c.max() - c.min())

def HistogramEqualization(im):
    """
    reference: https://github.com/torywalker/histogram-equalizer/blob/master/HistogramEqualization.ipynb
    """
    for i in range(im.shape[2]):
        im_ = im[:,:,i]
        im_flat =  im_.flatten()
        h = histogram(im_flat)
        c = cumulate_distribution(h)
#         c_norm = np.int32(cv2.normalize(c,None, 0,255,cv2.NORM_MINMAX))
        c_norm  = np.int32(normalize(c))
        im_eq = c_norm[im_flat]
        im_eq = im_eq.reshape(-1,im.shape[1])
        
        if i==0:
            im_eqs = np.array(im_eq)
        else:
            im_eqs = np.dstack((im_eqs,im_eq))
            
    return im_eqs


def AdjustGamma(im_, gamma=1.0):
    """
    Buid a lookup table of
    
    """
    im = im_.copy()
    gamma_inv = 1.0 / gamma
    gammatable = []
    for i in np.arange(0, 256):
        i = (i / 255.0) ** gamma_inv
        gammatable.append(i*255)
    
    gammatable = np.array(gammatable).astype("uint8")
    
    # apply gamma correction using the lookup table
    return cv2.LUT(im, gammatable)

def main():

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="../Data/Project2_Dataset1/Night Drive - 2689.mp4", help='Data path of images, Default: ../Data/Project2_Dataset1/Night Drive - 2689.mp4')
    Parser.add_argument('--SavePath', default="../Outputs/Problem1/", help='Save file name, Default: ../Outputs/Problem1.mp4')    
    Parser.add_argument('--Mode', default=1, help='1 for histogram_equalization, 2 gamma correction for , Default: 2')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    SavePath = Args.SavePath
    Mode = int(Args.Mode)
    
    if(not (os.path.isdir(SavePath))):
        print(SavePath, "  was not present, creating the folder...")
        os.makedirs(SavePath) 
    if Mode ==1:
        SaveFileName = SavePath + 'hist_eq.mp4'
    else:
        SaveFileName = SavePath + 'gamma_corr.mp4'
           
    cap = cv2.VideoCapture(DataPath)    
    w = int(cap.get(3)) 
    h = int(cap.get(4))
#     print(cap, w, h)
    if Mode ==1:
        w = w//3
        h = h//3
    ################### Load video writer ##############
    result = cv2.VideoWriter(SaveFileName,  
                            cv2.VideoWriter_fourcc(*'DIVX'), 
                            30, (w, h)) 
    ################### Run the code ##############  
    i = 0
    while(True):
        ret, im = cap.read()
        if ret:
            i+=1
            if Mode ==1:
                im  = cv2.resize(im, (w,h))
                out = HistogramEqualization(im)
                print('running equalization on frame :', i)
#                 cv2.imshow('Histogram Equalization', out)
            elif Mode ==2:
                out = AdjustGamma(im, gamma = 2.0)
                print('running gamma correction on frame :', i)
                cv2.imshow('Gamma Correction', out)
            result.write(np.uint8(out))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()            

if __name__ == '__main__':
    main()
    
