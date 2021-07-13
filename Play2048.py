import os

from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
import math
import sys
import cv2
from itertools import count
from My2048Matrix import M2048
from MyDQN import DQN

from tensorboardX import SummaryWriter
# from MyUtilities import *


saved_path = './trained_models/2048_Ep_10000_Terminal'

model = torch.load(saved_path)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)    
# criterion = nn.MSELoss()
env = M2048()
if torch.cuda.is_available():
    model.cuda()
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)

def play(idx):
    step = 0
    env.Reset()
    while True:
        state = np.copy(env.mat)
        expand_state = state[None,:,:] 
        px = env.MatTransfer(expand_state).cuda()
        nx = env.MatNormalize(expand_state).cuda()            
        prediction = model(px,nx).view(5).cpu()
        
        possibleAction = env.CheckPossibleAction()    
        for a in range(5):
            if not possibleAction[a]:
                prediction[a] = prediction[a] - float("Inf")    

        action = (torch.argmax(prediction)).item()
        # print state.astype('int'), 'prediction = ', prediction
        # tmp = raw_input()

        next_state, reward, terminal = env.OneMove(action)
        step = step + 1

        # if step % 10 == 0:                
        #     tensorImg, npImg = env.GenerateImg()            
        #     cv2.imwrite('./Visualization/Intermediate.jpg',npImg)

        possibleAction = env.CheckPossibleAction()
        terminal = 0
        if np.sum(possibleAction) == 0:
            terminal = 1
        if np.sum(possibleAction) == 1 and possibleAction[0] == 1:
            if env.maxScore < 1024:
                terminal = 1
                print "Terminated by no other actions than undo."
        curSum = np.sum(env.mat)
        undoStepThres = max( 100, curSum/64*10 )        
        if env.undoAcc > undoStepThres:
            terminal = 1
            print "Terminated by undo threshold."
        print 'step = ', step, 'action = ', action, 'max = ', np.max(env.mat), 'score = ', env.score
        if terminal:
            break

    maxValue = np.max(env.mat)
    tensorImg, npImg = env.GenerateImg()   
    if maxValue >= 512:   
        cv2.imwrite('./Results/Final_'+str(idx)+'_'+str(int(maxValue))+'.jpg',npImg)

    print env.mat.astype('int')
    print 'step = ', step
    print 'score = ', env.score
    print 'max_value = ', maxValue

    return maxValue


if __name__ == "__main__":    
    maxValueList = []
    for i in range(1000):
        maxValue = play(i)
        maxValueList.append(maxValue)

    maxValueList = np.array(maxValueList)
    np.savetxt( './Results/maxValueList.txt', maxValueList, fmt='%d' )

