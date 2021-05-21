import cv2
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from misc.utils import *
from misc.preprocess import *
from misc.LaneDetector import *

import yaml
import argparse

def main():

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="../Data/Project2_Dataset2/", help='Data path of images, Default: ./Project2_Dataset2/')
    Parser.add_argument('--Mode', default=1, help='Video type 1 or 2, Default: type 1')
    Parser.add_argument('--SavePath', default='../Outputs/Problem2/', help='Path to save Results, Default: ./Outputs/')
    
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    SavePath = Args.SavePath
    Mode = int(Args.Mode)
    
    if(not (os.path.isdir(SavePath))):
        print(SavePath, "  was not present, creating the folder...")
        os.makedirs(SavePath) 
        
    if Mode == 1:
        print('entered mode 1')
        ################### Set up file names ##############
        VideoPath = DataPath+'data_1/data/'
        yaml_path = DataPath+ 'data_1/camera_params.yaml'        
        SaveFileName = SavePath+'data1_output.mp4'
        
        ################# Assign model parameters ###########
        with open(yaml_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        K = np.array([np.float32(value) for value in data['K'].split()]).reshape(3,3)
        kC = np.array([np.float32(value) for value in data['D'].split()]).reshape(1,-1)
        
        imgs =  StitchImages(K,kC, datapath = VideoPath+"*png")
 
        boundaries = (150,950,610,710) 
        horizon = 265
        window_size = 200
        mva_filter = True
        mva_win = 5
        lanecenter = 600
        h, w = imgs[0].shape[:2]
        
        ################# Load Lane detector class ###########
        detector = LaneDetector(horizon, boundaries, window_size, mva_filter, mva_win, lanecenter, Mode)
        
        ################### Load video writer ##############
        result = cv2.VideoWriter(SaveFileName,  
                                cv2.VideoWriter_fourcc(*'DIVX'), 
                                20, (w, h)) 
        for i, im in enumerate(imgs):
            print(i, '####################################')
            out = detector.processLane(im)

            cv2.imshow('Lanes', rgb(out))
            result.write(rgb(np.uint8(out)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        result.release()
        cv2.destroyAllWindows()            
        
    if Mode == 2:
        ################### Set up file names ##############
        VideoFilePath = DataPath + "/data_2/challenge_video.mp4"
        SaveFileName = SavePath + 'data2_output.mp4'
        
        ################### Assign model parameters ##############
        ### hard coded these parameters since the yaml file had wrong format to do a yaml.load
        K = np.array([9.037596e+02, 0.000000e+00, 6.957519e+02,
              0.000000e+00, 9.019653e+02, 2.242509e+02,
              0.000000e+00, 0.000000e+00, 1.000000e+00]).reshape(3,3)

        kC = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]).reshape(1,-1)
        

        boundaries = (200,1200,600,730) 
        horizon = 480
        window_size = 250
        mva_filter = True
        mva_win = 10
        lanecenter = 600
        ################### Load Lane detector model ##############
        detector = LaneDetector(horizon, boundaries, window_size, mva_filter, mva_win, lanecenter, Mode)

        cap = cv2.VideoCapture(VideoFilePath)
        w = int(cap.get(3)) 
        h = int(cap.get(4))

        ################### Load video writer ##############
        result = cv2.VideoWriter(SaveFileName,  
                                cv2.VideoWriter_fourcc(*'DIVX'), 
                                30, (w, h)) 
        ################### Run the code ##############
        i = 0
        while(True):
            i+=1
            ret, im = cap.read()
            if not ret:
                print("Stream ended..")
                break
            else:
                im =  cv2.undistort(rgb(im), K, kC, None, K)
                print(i, '####################################')
                out = detector.processLane(im)
    
                cv2.imshow('Lanes', rgb(out))
                result.write(rgb(np.uint8(out)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        result.release()
        cv2.destroyAllWindows()            
        
if __name__ == '__main__':
    main()
