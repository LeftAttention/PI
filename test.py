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
