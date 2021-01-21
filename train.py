import cv2
import torch
import visdom
#import sys

import Assistant
import numpy as np
from data_loader import Generator
from parameters import Parameters
import test
import evaluation
import util
import copy

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')

    vis = visdom.Visdom()
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))
    
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
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(1912, "tensor(0.9420)")
        
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()
        #torch.backends.cudnn.benchmark=True

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    
    for epoch in range(p.n_epoch):
        lane_agent.training_mode()
        for inputs, target_lanes, target_h, test_image, data_list in loader.Generate(sampling_list):
            #training
            #util.visualize_points(inputs[0], target_lanes[0], target_h[0])
            print("epoch : " + str(epoch))
            print("step : " + str(step))
            loss_p = lane_agent.train(inputs, target_lanes, target_h, epoch, lane_agent, data_list)
            torch.cuda.synchronize()
            loss_p = loss_p.cpu().data
            
            if step%50 == 0:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * int(step/50),
                    Y=torch.Tensor([loss_p]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')
                
            if step%100 == 0:
                lane_agent.save_model(int(step/100), loss_p)
                testing(lane_agent, test_image, step, loss_p)
            step += 1

        sampling_list = copy.deepcopy(lane_agent.get_data_list())
        lane_agent.sample_reset()
