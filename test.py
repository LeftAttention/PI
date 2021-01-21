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
