import cv2
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from misc.utils import gray

def GradientThresholding(im):
    sx = cv2.Sobel(gray(im),cv2.CV_64F,1,0,ksize=3) 
    sy = cv2.Sobel(gray(im),cv2.CV_64F,0,1,ksize=3) 

    ### set  sobel edges
    sobelx = np.uint8(np.absolute(sx))
    t1,t2= 25,255
    _,sobelx = cv2.threshold(sobelx,t1,t2,cv2.THRESH_BINARY)
    kernel = np.ones((1,2),np.uint8)
    sobelx = cv2.dilate(sobelx,kernel,iterations = 1)
    sobelx = sobelx//255
  
    return sobelx
#     ### set  gradient directions
#     dir_output = grad_direction(sy,sx)

#     sobel_output = np.zeros_like(dir_output, np.uint8)  
#     sobel_output[((sobelx == 1) | (dir_output == 1))] = 1
    
#     return sobel_output

def ColorThreshold(im, mode):
    imHLS = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)

    # White-ish areas in image
    # H value can be arbitrary, thus within [0 ... 360] (OpenCV: [0 ... 180])
    # L value must be relatively high (we want high brightness), e.g. within [0.7 ... 1.0] (OpenCV: [0 ... 255])
    # S value must be relatively low (we want low saturation), e.g. within [0.0 ... 0.3] (OpenCV: [0 ... 255])
    white_lower = np.array([np.round(  0 / 2), np.round(0.7 * 255), np.round(0.00 * 255)])
    white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    white_mask = cv2.inRange(imHLS, white_lower, white_upper)

    # Yellow-ish areas in image
#     lower = 20,95,55
#     upper 45, 200,255
    # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
    # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
    # S value must be above some threshold (we want at least some saturation), e.g. within [0.1 ... 1.0]
#     yellow_lower = np.array([np.round( 40 / 2), np.round(0.5 * 255), np.round(0.5 * 255)])
    if mode ==1:
        yellow_lower = np.array([np.round( 40 / 2), np.round(0.37 * 255), np.round(0.22 * 255)])
    else:
        yellow_lower = np.array([np.round( 40 / 2), np.round(0.3 * 255), np.round(0.1 * 255)])
    yellow_upper = np.array([np.round( 90 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    yellow_mask = cv2.inRange(imHLS, yellow_lower, yellow_upper)

    # Calculate combined mask, and masked image
    colorMask = cv2.bitwise_or(yellow_mask, white_mask)
    ret,colorMask = cv2.threshold(colorMask,0.5,1,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    colorMask = cv2.dilate(colorMask,kernel,iterations = 2)
    colorMask = cv2.morphologyEx(colorMask, cv2.MORPH_CLOSE, kernel)

    return colorMask


def mask(im,mode, grad = False):
    colorMask = ColorThreshold(im, mode)
    if grad==True:
        sobel_output = GradientThresholding(im)
        binary = cv2.bitwise_or(colorMask, sobel_output)
        return binary
    else:
        return colorMask
    
    
########################################################################################################################
######################################## ROI from hough transform  ########################################
########################################################################################################################

def getROI_points(im, horizon = 460, minLineLength = 300,d_bot1 = 100 , d_bot2 = 100, d_top1 = 20,d_top2=20): 
    height, width = im.shape[:2]
    
    im_crop = im[horizon:,:, :]
    out = mask(im_crop, 2, True)

    distribution  = np.sum(out[out.shape[0]//2:,:],axis=0)
    mdpt = int(len(distribution)//2) # find midpoint of the distribution alonf x axis    
    Xcleft = np.argmax(distribution[:mdpt]) # find left lane peak as p1
    Xcright = np.argmax(distribution[mdpt:]) + mdpt # find right lane peak as p2 

    vanishROI = np.zeros_like(out)
    vanishROI[:,Xcleft-100:Xcright+100] = out[:,Xcleft-100:Xcright+100]


    lines =np.squeeze(cv2.HoughLinesP(vanishROI, 2, np.pi / 180, 100,  
                                np.array([]), minLineLength = minLineLength,  
                                maxLineGap = 10))

    v_l,v_r =  vanishingLineEndPoints(lines)

    b_left, b_right = Xcleft- d_bot1, Xcright+d_bot2 ## edit to change ROI
    t_left, t_right = v_l[0] -d_top1, v_r[0] +d_top2 ## edit to change ROI

    pts = np.array([[b_left,height],[t_left,horizon],[t_right,horizon], [b_right, height]], np.float32)

    return pts


def get_points(image, line_data): 
#     print(line_data)
    m,c = line_data 
    y1 = image.shape[0] 
    y2 = int(y1 * (3 / 5))
    
    x1 = int((y1 - c) / m) 
    x2 = int((y2 - c) / m)
    
    return np.int32(np.array([x1, y1, x2, y2]) )
def vanishingLineEndPoints(lines):

    left_line = []
    right_line = []
    for l in lines:
        x1,y1,x2,y2 = l
        m,c = np.polyfit((x1, x2), (y1, y2), 1)  
        if m < 0: 
            left_line.append(l) 
        else: 
            right_line.append(l)

    right_line = np.int32(np.mean(right_line, axis=0))
    left_line = np.int32(np.mean(left_line, axis=0))

    v_l =left_line[2:]
    v_r =right_line[:2]
    return v_l, v_r


def vanishingPoints(lines, line_image):  
    left_line , right_line = [],[]
    for l in lines:
        x1,y1,x2,y2 = l
        m,c = np.polyfit((x1, x2), (y1, y2), 1)  
        if m < 0: 
            left_line.append((m,c)) 
        else: 
            right_line.append((m,c))

    if left_line:
        left_avg = np.mean(left_line, axis = 0)
        left_pts = get_points(line_image, left_avg) 
    else:
        left_pts = [0,0,0,0]

    if right_line:
        right_avg = np.mean(right_line, axis = 0)
        right_pts = get_points(line_image, right_avg) 
    else:
        left_pts = [0,0,0,0]

    return left_pts[2:], right_pts[:2]

