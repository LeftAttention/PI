import cv2
import json
import torch
import Assistant
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util
from tqdm import tqdm
import csaps

p = Parameters()




def Testing():
    print('Testing')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_assistant = Assistant.Assistant()
    else:
        lane_assistant = Assistant.Assistant()
        lane_assistant.load_weights(804, "tensor(0.5786)")
	
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_assistant.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_assistant.evaluate_mode()

    if p.mode == 0 : # check model with test data 
        for _, _, _, test_image in loader.Generate():
            _, _, ti = test(lane_assistant, np.array([test_image]))
            cv2.imshow("test", ti[0])
            cv2.waitKey(0)
	
    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("abc.mp4")
        while(cap.isOpened()):
            ret, frame = cap.read()
            torch.cuda.synchronize()
            prevTime = time.time()
            frame = cv2.resize(frame, (512,256))/255.0
            frame = np.rollaxis(frame, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([frame])) 
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            ti[0] = cv2.resize(ti[0], (1280,800))
            cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame',ti[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
