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

batch_size = 160
sub_batch_size = [160,80,50,40]
lr = 1e-6
gamma = 0.99
initial_epsilon = 0.1
final_epsilon = 1e-4
num_iters = 200000
num_episodes = 10000
replay_memory_size = 20000
saved_path = 'trained_models'

tb_log_path = 'tensorboard/train11081750'
os.makedirs(tb_log_path)
writer = SummaryWriter(tb_log_path)

model = DQN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)    
criterion = nn.MSELoss()
env = M2048()
if torch.cuda.is_available():
    model.cuda()
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)



def PredictionFilter(prediction, state, flag, possibleAction = None):
    if possibleAction is None:    
        possibleAction, impossibleAction, possibleActionList, impossibleActionList = env.CheckPossibleActionUI( state, flag )
    else:
        possibleActionList = []
        for a in range(5):
            if possibleAction[a] == 1:
                possibleActionList.append(a)
    for a in range(5):
        if not possibleAction[a]:
            prediction[a] = prediction[a] - float("Inf")
    # prediction[impossibleActionList] = -100
    return possibleActionList


def train():
    all_iter = 0   
    replay_memory = []
    replay_memory_list = [[],[],[],[]]
    for i_episode in range(num_episodes):
        env.Reset()   
        state = env.mat     

        iter = 0
        while True:            
            expand_state = state[None,:,:] 
            px = env.MatTransfer(expand_state).cuda()
            nx = env.MatNormalize(expand_state).cuda()            
            prediction = model(px,nx).view(5).cpu()

            possibleActionList = PredictionFilter(prediction, state, 1 )
            if possibleActionList == []:
                print state
                print env.mat
                print type(state)                                
                print possibleActionList
                tmp = raw_input()

            epsilon = final_epsilon + ( (num_episodes - i_episode ) * ( initial_epsilon - final_epsilon ) / num_episodes )
            u = random()
            random_action = u <= epsilon            
            if random_action:
                # print("Perform a random action")
                action = sample(possibleActionList,1)
                action = action[0]
                act = 'rand'
            else:
                action = (torch.argmax(prediction)).item()
                act = 'max'

            next_state, reward, terminal = env.OneMove(action)

            next_possibleAction, next_impossibleAction, next_possibleActionList, next_impossibleActionList = env.CheckPossibleActionUI( next_state, 0 )
            next_possibleAction = next_possibleAction.view(1,5)
            expand_next_state = next_state[None,:,:]
            next_px = env.MatTransfer(expand_next_state)
            next_nx = env.MatNormalize(expand_next_state)
            px = px.cpu()
            nx = nx.cpu()

            m_state = torch.from_numpy(expand_state)
            m_next_state = torch.from_numpy(expand_next_state)
            m_action = np.zeros((1,5))
            m_action[0][action] = 1
            m_action = torch.from_numpy(m_action)

            mIdx = int( math.log( np.max(state), 2 ) - 1 ) / 4
            replay_memory_list[mIdx].append([m_state, m_action, reward, m_next_state, terminal, px, nx, next_px, next_nx, next_possibleAction])
            # replay_memory.append([m_state, m_action, reward, m_next_state, terminal, px, nx, next_px, next_nx, next_possibleAction])
            if len(replay_memory_list[mIdx]) > replay_memory_size:
                del replay_memory_list[mIdx][0]

            nMemList = 0
            for m in range(4):
                if len(replay_memory_list[3-m]) > 0:
                    nMemList = 4-m
                    break            
            batch = []
            for m in range(nMemList):
                batch = batch + sample( replay_memory_list[m], min(sub_batch_size[nMemList-1],len(replay_memory_list[m]) ) )

            real_batch_size = len(batch)
            # real_batch_size = min(len(replay_memory), batch_size)
            # batch = sample(replay_memory, real_batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, px_batch, nx_batch, next_px_batch, next_nx_batch, next_possible_action_batch = zip(*batch)

            state_batch = torch.cat(tuple(state for state in state_batch))
            action_batch = torch.cat(tuple(action for action in action_batch))
            next_state_batch = torch.cat(tuple(next_state for next_state in next_state_batch))            
            reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
            px_batch = torch.cat(tuple(px for px in px_batch))       
            nx_batch = torch.cat(tuple(nx for nx in nx_batch))       
            next_px_batch = torch.cat(tuple(next_px for next_px in next_px_batch))       
            next_nx_batch = torch.cat(tuple(next_nx for next_nx in next_nx_batch))           
            next_possible_action_batch = torch.cat(tuple(next_pa for next_pa in next_possible_action_batch))                        

            if torch.cuda.is_available():
                px_batch = px_batch.cuda()
                nx_batch = nx_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_px_batch = next_px_batch.cuda()
                next_nx_batch = next_nx_batch.cuda()

            current_prediction_batch = model(px_batch,nx_batch).view(real_batch_size,5)
            next_prediction_batch = model(next_px_batch,next_nx_batch).view(real_batch_size,5)
            possibleActionList_batch = []
            for b in range(real_batch_size):
                possibleActionList = PredictionFilter(next_prediction_batch[b,:], next_state_batch[b,:,:].numpy(), 0, possibleAction=next_possible_action_batch[b,:])
                possibleActionList_batch.append(possibleActionList)

            y_batch = torch.cat(tuple(reward if terminal else reward + gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))
            
            q_value = torch.sum(current_prediction_batch * action_batch.float(), dim=1)
            optimizer.zero_grad()

            loss = criterion(q_value, y_batch)
            loss.backward()
            optimizer.step()

            state = next_state
            # env.mat = state

            iter = iter + 1
            env.step = iter
            all_iter = all_iter + 1
            if (iter+1) % 1000 == 0:
                torch.save(model, "{}/2048_Ep_{}_Iter_{}".format(saved_path, i_episode, iter+1))
            
            print("Ep-Iter: {}-{}, Action: {}-{}, CurMax: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            i_episode+1, iter, action, act, int(np.max(state)), loss, epsilon, reward, torch.max(prediction)))
            

            if ( all_iter - 1 ) % 10 == 0:
                maxParam, minParam = model.CheckMaxMinParam()            
                tensorImg, npImg = env.GenerateImg()
                writer.add_scalar('Train/Loss', loss/real_batch_size, all_iter-1)
                writer.add_scalar('Train/Epsilon', epsilon, all_iter-1)
                writer.add_scalar('Train/Reward', reward, all_iter-1)
                writer.add_scalar('Train/Q-value', torch.max(prediction), all_iter-1)
                writer.add_scalar('Train/maxParam', maxParam, all_iter-1)
                writer.add_scalar('Train/minParam', minParam, all_iter-1)
                writer.add_scalar('Train/logMaxScore', math.log(env.maxScore,2), all_iter-1 )
                writer.add_scalar('Train/nCorrNeighbor', env.numPrevCorrNeighbor, all_iter-1 )
                
                writer.add_image('CurrentGameUI', tensorImg, all_iter-1)

            if terminal:
                tensorImg, npImg = env.GenerateImg()
                imgIdx = i_episode % 10
                imgName = './Visualization/TerminalState' + str(imgIdx) + '.jpg'
                cv2.imwrite(imgName,npImg)

                break

        if (i_episode+1) % 100 == 0 or iter > 1000:
            torch.save(model, "{}/2048_Ep_{}_Terminal".format(saved_path, i_episode+1))


    writer.export_scalars_to_json("./tensorboard/test.json")
    writer.close()

        
if __name__ == "__main__":    
    train()