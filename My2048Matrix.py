import numpy as np
import torch
import math
import os
import sys
import cv2

class M2048():    
    def __init__(self, cfgFile=None):
        self.mat = np.zeros((4,4))
        self.step = 0
        self.score = 0
        self.maxScore = 0
        self.matSeq = []
        self.opSeq = []
        self.scoreSeq = []
        self.sumScoreSeq = []
        self.undoAcc = 0
        self.doAcc = 0
        self.undoSteps = 3
        self.undoAccThres = 0
        self.prev_reward_state = [0,0,0,0]
        self.prevSumScore = 0
        self.prevNeighborScore = 0
        self.accStuck = 0
        self.numPrevCorrNeighbor = 0
        # self.order shape (4,2,16)
        self.corner = np.array([0,3,15,12])
        self.orderList = np.array([[[0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12],[0,4,8,12,13,9,5,1,2,6,10,14,15,11,7,3]], \
                                   [[3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15],[3,7,11,15,14,10,6,2,1,5,9,13,12,8,4,0]], \
                                   [[15,14,13,12,8,9,10,11,7,6,5,4,0,1,2,3],[15,11,7,3,2,6,10,14,13,9,5,1,0,4,8,12]], \
                                   [[12,13,14,15,11,10,9,8,4,5,6,7,3,2,1,0],[12,8,4,0,1,5,9,13,14,10,6,2,3,7,11,15]] ])
        self.colorList = [(180,191,205),(217,227,237),(194,225,234),(118,181,242),(110,152,246),(98,125,245),  \
                          (57,99,242),  (127,208,232),(113,205,232),(100,201,232),(89,198,230), (79,195,230),  \
                          (109,106,239),(88,78,235),  (61,65,243),  (216,183,113),(221,162,92) ]

        if cfgFile is not None:
            cfg = np.loadtxt( cfgFile )   
            self.score = cfg[0][0]
            self.maxScore = cfg[0][1]
            self.mat = cfg[1:5,:]
        else:
            # Initialization of n numbers
            initNumPos = 1
            for i in range(initNumPos):
                row, col = self.AddPoint()
                self.opSeq.append( [ -1, row, col ] )
            self.matSeq.append(np.copy(self.mat))
            self.scoreSeq.append(self.score)
            self.sumScoreSeq.append(np.sum(self.mat) )
            self.maxScore = np.max(self.mat)        


    def Reset(self):
        self.mat = np.zeros((4,4))
        self.step = 0
        self.score = 0
        self.maxScore = 0
        self.matSeq = []
        self.opSeq = []
        self.scoreSeq = []
        self.sumScoreSeq = []
        self.undoAcc = 0
        self.doAcc = 0
        self.undoSteps = 3
        self.undoAccThres = 0
        self.prevSumScore = 0
        self.accStuck = 0
        self.numPrevCorrNeighbor = 0
        initNumPos = 1
        for i in range(initNumPos):
            row, col = self.AddPoint()
            self.opSeq.append( [ -1, row, col ] )
        self.matSeq.append(np.copy(self.mat))
        self.scoreSeq.append(self.score)

        self.maxScore = np.max(self.mat)            
                
    def MatTransfer(self,x):
        nB = np.shape(x)[0]
        powerMat = torch.zeros(nB,17,4,4)
        for b in range(nB):
            for i in range(4):
                for j in range(4):
                    if x[b][i][j] == 0:
                        powerMat[b][0][i][j] = 1.0
                    else:
                        power = int(math.log(x[b][i][j],2))
                        powerMat[b][power][i][j] = 1.0
        if torch.cuda.is_available():
            powerMat = powerMat.cuda()

        return powerMat      

    def MatNormalize(self,x):
        nB = np.shape(x)[0]
        NormMat = torch.zeros(nB,1,4,4)
        for b in range(nB):
            for i in range(4):
                for j in range(4):
                    if x[b][i][j] == 0:
                        NormMat[b][0][i][j] = 0.0
                    else:
                        power = int(math.log(x[b][i][j],2))
                        NormMat[b][0][i][j] = float(power) / 16.0
        if torch.cuda.is_available():
            NormMat = NormMat.cuda()

        return NormMat    

    def RandomPos(self,thres,op):
        assert op == '>' or op == '=' or op == '<' or op == '>=' or op == '!=' or op == '<=', \
        'Operation should be one from ">", "=", "<", ">=", "!=", "<=".'
        prob = np.random.rand(4,4)
        if op == '=':
            prob[self.mat!=thres] = 0
        elif op == '!=':
            prob[self.mat==thres] = 0
        elif op == '<':
            prob[self.mat>=thres] = 0
        elif op == '>':
            prob[self.mat<=thres] = 0
        elif op == '<=':
            prob[self.mat>thres] = 0
        elif op == '>=':
            prob[self.mat<thres] = 0            
        (row,col) = np.unravel_index( np.argmax(prob, axis=None), prob.shape )

        return row, col

    def AddPoint(self):
        row, col = self.RandomPos(0,'=')
        self.mat[row][col] = 2
        return row, col

    def VecSum(self, inputVec0):
        inputVec = np.zeros(4)
        tmp = inputVec0[inputVec0!=0]
        inputVec[0:len(tmp)] = tmp
        extraScore = 0

        outputVec = np.zeros(4)
        if inputVec[0] == inputVec[1]:
            extraScore = extraScore + inputVec[0]
            outputVec[0] = inputVec[0] + inputVec[1]            
            if inputVec[2] == inputVec[3]:
                extraScore = extraScore + inputVec[2]
                outputVec[1] = inputVec[2] + inputVec[3]            
            else:
                outputVec[1] = inputVec[2]
                outputVec[2] = inputVec[3]
        else:
            outputVec[0] = inputVec[0]
            outputVec[1] = inputVec[1]
            if inputVec[1] == inputVec[2]:
                extraScore = extraScore + inputVec[1] 
                outputVec[1] = inputVec[1] + inputVec[2]
                outputVec[2] = inputVec[3]                  
            else:
                if inputVec[2] == inputVec[3]:
                    extraScore = extraScore + inputVec[2]
                    outputVec[2] = inputVec[2] + inputVec[3]                    
                else:
                    outputVec[2] = inputVec[2]
                    outputVec[3] = inputVec[3]     
        # print extraScore   
        return outputVec, extraScore

    def VecCheck(self, inputVec):
        outputVec = np.zeros(4)        
        inputNZ = inputVec[inputVec!=0]

        outputVec[0:len(inputNZ)] = inputNZ
        if np.sum( outputVec == inputVec ) == len(inputVec):
            flagZero = 0
        else:
            flagZero = 1
        flagEqual = 0
        for i in range(len(inputNZ)-1):
            if inputNZ[i] == inputNZ[i+1]:
                flagEqual = 1

        return flagEqual or flagZero

    def MatSum(self, action):          
        if action == 0: # action 0: undo
            extraMatScore = self.Undo()           
        elif action == 1: # action 1: up
            extraMatScore = self.MatSumUp()
        elif action == 2: # action 2: down
            extraMatScore = self.MatSumDown()
        elif action == 3: # action 3: left
            extraMatScore = self.MatSumLeft()
        elif action == 4: # action 4: right
            extraMatScore = self.MatSumRight()  
        else:
            print "Action", action, "is illegal!"
            tmp = raw_input()

        return extraMatScore

    def Undo(self):
        # if len(matSeq) > 1:
        prevScore = self.scoreSeq[-1]
        for i in range(self.undoSteps):
            # extraMatScore = extraMatScore - self.scoreSeq[-1]
            self.matSeq.pop()
            self.scoreSeq.pop()
        self.mat = np.copy(self.matSeq[-1])
        self.score = np.copy(self.scoreSeq[-1])   
        extraMatScore = self.score - prevScore
        return extraMatScore         

    def MatSumUp(self):        
        extraMatScore = 0
        for col in range(4):
            newVec, extraScore = self.VecSum(self.mat[:,col])
            self.mat[:,col] = newVec
            extraMatScore = extraMatScore + extraScore            
        return extraMatScore
        
    def MatSumDown(self):        
        extraMatScore = 0
        for col in range(4):
            newVec, extraScore = self.VecSum( np.flip( self.mat[:,col], 0 ) )
            self.mat[:,col] = np.flip( newVec, 0 )            
            extraMatScore = extraMatScore + extraScore            
        return extraMatScore
    
    def MatSumLeft(self):        
        extraMatScore = 0
        for row in range(4):
            newVec, extraScore = self.VecSum(self.mat[row,:])
            self.mat[row,:] = newVec
            extraMatScore = extraMatScore + extraScore
        return extraMatScore
    
    def MatSumRight(self):        
        extraMatScore = 0
        for row in range(4):
            newVec, extraScore = self.VecSum( np.flip( self.mat[row,:], 0 ) )
            self.mat[row,:] = np.flip( newVec, 0 )
            extraMatScore = extraMatScore + extraScore
        return extraMatScore
   
    def OneMoveUI(self, prevMat, action):
        # possibleAction = self.CheckPossibleAction()
        # if possibleAction[action] != 0:
        if action != 0:            
            newMat, extraMatScore = self.MatSum(prevMat, action)                        
            self.score = self.score + extraMatScore
            
            row, col = self.AddPoint()    
            self.opSeq.append( [ action, row, col ] )
            self.matSeq.append(np.copy(self.mat))
            self.scoreSeq.append(self.score)
            self.doAcc = self.doAcc + 1
            self.undoAcc = 0            
        else: # undo
            newMat, extraMatScore = self.MatSum(prevMat, action)
            # if self.undoAcc % 3 == 0:
            #     row, col = self.RandomPos(8,'<=')                    
            #     self.mat[row,col] = 0
            #     self.opSeq.append( [ action, row, col ] )                    
            # else:
            self.opSeq.append( [ action, -1, -1 ] )
            self.undoAcc = self.undoAcc + 1
            self.doAcc = 0
            
        reward = self.RewardFunction(action, extraMatScore, self.maxScore)
        terminal = self.CheckTerminal()
        self.maxScore = np.max(self.mat)

        return newMat, reward, terminal

    def OneMove(self,action):
        possibleAction = self.CheckPossibleAction()
        # print possibleAction
        if possibleAction[action] != 0:
            if action != 0:            
                extraMatScore = self.MatSum(action)                        
                self.score = self.score + extraMatScore
                
                row, col = self.AddPoint()    
                self.opSeq.append( [ action, row, col ] )
                self.matSeq.append(np.copy(self.mat))
                self.scoreSeq.append(self.score)
                self.doAcc = self.doAcc + 1
                self.undoAcc = 0            
            else: # undo
                extraMatScore = self.MatSum(action)
                # if self.undoAcc % 3 == 0:
                #     row, col = self.RandomPos(8,'<=')                    
                #     self.mat[row,col] = 0
                #     self.opSeq.append( [ action, row, col ] )                    
                # else:
                self.opSeq.append( [ action, -1, -1 ] )
                self.undoAcc = self.undoAcc + 1
                self.doAcc = 0
                
            # print extraMatScore
            reward = self.RewardFunctionV3(action, extraMatScore, self.maxScore)
            terminal = self.CheckTerminalV3()
            self.maxScore = np.max(self.mat)
            self.sumScoreSeq.append(np.sum(self.mat))

        return self.mat, reward, terminal

    def CheckPossibleAction(self):
        matClone = np.copy(self.mat)        
        # if self.doAcc >= self.undoAccThres:
        if len(self.matSeq) > 0:# and np.max(self.mat) >= 2048:
            flagUndo = 1
        else:
            flagUndo = 0
        flagUp = 0 # Check Up 
        for col in range(4):
            flagVec = self.VecCheck( matClone[:,col] )
            flagUp = flagUp or flagVec
        flagDown = 0 # Check Down
        for col in range(4):
            flagVec = self.VecCheck( np.flip( matClone[:,col], 0 ) )
            flagDown = flagDown or flagVec
        flagLeft = 0 # Check Left
        for row in range(4):
            flagVec = self.VecCheck(matClone[row,:])
            flagLeft = flagLeft or flagVec
        flagRight = 0 # Check Right
        for row in range(4):
            flagVec = self.VecCheck( np.flip( matClone[row,:], 0 ) )
            flagRight = flagRight or flagVec

        return np.array( [ flagUndo, flagUp, flagDown, flagLeft, flagRight ] )

    def CheckPossibleActionUI(self, state, flag ):
        # matClone = np.copy(self.mat)        
        # if self.doAcc >= self.undoAccThres:
        if len(self.matSeq) > 0: # and np.max(self.mat) >= 2048:
            flagUndo = 1
        else:
            flagUndo = 0
        flagUp = 0 # Check Up 
        for col in range(4):
            flagVec = self.VecCheck( state[:,col] )
            flagUp = flagUp or flagVec
        flagDown = 0 # Check Down
        for col in range(4):
            flagVec = self.VecCheck( np.flip( state[:,col], 0 ) )
            flagDown = flagDown or flagVec
        flagLeft = 0 # Check Left
        for row in range(4):
            flagVec = self.VecCheck(state[row,:])
            flagLeft = flagLeft or flagVec
        flagRight = 0 # Check Right
        for row in range(4):
            flagVec = self.VecCheck( np.flip( state[row,:], 0 ) )
            flagRight = flagRight or flagVec

        possibleAction = torch.tensor( [ flagUndo, flagUp, flagDown, flagLeft, flagRight ] )

        impossibleAction = 1 - possibleAction
        possibleActionList = []
        impossibleActionList = []
        for i in range(5):
            if possibleAction[i] == 1:
                possibleActionList.append(i)
            if possibleAction[i] == 0:
                impossibleActionList.append(i)

        return possibleAction, impossibleAction, possibleActionList, impossibleActionList

    def CheckMaxPossibleMerge(self):
        # check maximum possible single etra score
        maxMergeScore = 0
        for i in range(4):
            for j in range(3):
                if self.mat[i,j] == self.mat[i,j+1]:
                    maxMergeScore = max(maxMergeScore,self.mat[i,j])
                if self.mat[j,i] == self.mat[j+1,i]:
                    maxMergeScore = max(maxMergeScore,self.mat[j,i])
                    
        return maxMergeScore

    def CheckBestMerge(self):
        argsortMat = -np.sort(-self.mat.reshape(16))    
        maxMergeScore = 0
        for p in range(15):
            v0 = argsortMat[p]
            v1 = argsortMat[p+1]
            if v0 == v1:
                maxMergeScore = v0
                break
        return maxMergeScore
    
    def CheckMaxValue(self, thres ):
        matClone = np.copy( self.mat.reshape(16) )   
        maxValue = np.max( self.mat.reshape(16) )
        maxCnId = -1
        maxFlag = -1
        if maxValue >= thres:
            for cn in range(4):
                if matClone[self.corner[cn]] == maxValue:
                    maxCnId = cn
                    maxFlag = 1
            return maxFlag, maxCnId
        else:
            return 0, maxCnId

    def FindNeighborSet(self, i, j):
        neighborSet = []
        if i-1 >= 0:
            neighborSet.append(self.mat[i-1,j])
        if i+1 <= 3:
            neighborSet.append(self.mat[i+1,j])
        if j-1 >= 0:
            neighborSet.append(self.mat[i,j-1])
        if j+1 <= 3:
            neighborSet.append(self.mat[i,j+1])

        return np.array(neighborSet)

    def CheckNeighbor(self, thres, maxCnId):
        matClone = np.copy( self.mat.reshape(16) )   
        # self.order[maxCnId,:,:] 
        argsortMat = -np.sort(-self.mat.reshape(16)) 
        vSeq = np.copy(argsortMat).tolist()
        for p in range(15):
            pos = 14 - p
            if vSeq[pos] == vSeq[pos+1]:
                del vSeq[pos+1]

        numCorrNeighbor = 0
        for orderId in range(2):
            order = self.orderList[maxCnId,orderId,:]
            numCorrNeighborInOneOrder = 0
            curPos = 0
            for p in range(15):
                if matClone[order[p+1]] >= thres:
                    if matClone[order[p+1]] == vSeq[curPos]: # mat[p] == mat[p+1] 
                        numCorrNeighborInOneOrder = numCorrNeighborInOneOrder + 1
                    elif matClone[order[p+1]] == vSeq[curPos+1]: # mat[p] != mat[p+1] but in correct order
                        numCorrNeighborInOneOrder = numCorrNeighborInOneOrder + 1
                        curPos = curPos + 1
                    else:
                        break
            numCorrNeighbor = max( numCorrNeighbor, numCorrNeighborInOneOrder )

        nCorr = numCorrNeighbor + 1
        nWrong = np.sum(self.mat>=thres) - nCorr
        return nCorr, nWrong

    def CheckNeighborV2(self,thres):
        matClone = np.copy( self.mat.reshape(16) )   
        # self.order[maxCnId,:,:] 
        argsortMat = -np.sort(-self.mat.reshape(16)) 
        vSeq = np.copy(argsortMat).tolist()
        rSeq = np.ones(16).tolist()
        for p in range(15):
            pos = 14 - p
            if vSeq[pos] == vSeq[pos+1]:
                rSeq[pos] = rSeq[pos] + rSeq[pos+1]
                del vSeq[pos+1]
                del rSeq[pos+1]

        score = np.zeros((4,4))

        for i in range(4):
            for j in range(4):
                v = self.mat[i,j]                
                if v >= thres:
                    weight = float(math.log(v,2)) / 4
                    neighborSet = self.FindNeighborSet(i,j)
                    seqId = vSeq.index(v)
                    if rSeq[seqId] == 1:
                        flag = -1
                        if np.sum( np.isin(neighborSet,vSeq[seqId+1]) ): 
                        # next number is in the neighbor set
                            if seqId == 0:
                                flag = 1
                            else:
                                if np.sum( np.isin(neighborSet,vSeq[seqId-1]) ): 
                                # prev number is in the neighbor set
                                    flag = 1

                    if rSeq[seqId] > 1:
                        flag = -1
                        numSame =  np.sum( np.isin(neighborSet,v) )
                        if numSame >= 2: 
                        # there is more than 1 same numbers in the neighbor set
                            flag = 1
                        if numSame == 1:
                            if seqId == 0:
                            # v is the largest
                                if len(neighborSet) == 2 or np.sum( np.isin(neighborSet,vSeq[seqId+1]) ): 
                                # v is in corner or next number is in the neibhgorset
                                    flag = 1
                            else:
                                if np.sum( np.isin(neighborSet,vSeq[seqId-1]) ) or np.sum( np.isin(neighborSet,vSeq[seqId+1]) ):
                                # next number or prev number is in the neibhgorset
                                    flag = 1

                    score[i,j] = flag * weight

        return score

    def CheckNeighborV3(self,thres):
        matClone = np.copy( self.mat.reshape(16) )   
        # self.order[maxCnId,:,:] 
        argsortMat = -np.sort(-self.mat.reshape(16)) 
        argsortIdx = np.argsort(-self.mat.reshape(16))
        vSeq = np.copy(argsortMat).tolist()
        rSeq = np.ones(16).tolist()
        for p in range(15):
            pos = 14 - p
            if vSeq[pos] == vSeq[pos+1]:
                rSeq[pos] = rSeq[pos] + rSeq[pos+1]
                del vSeq[pos+1]
                del rSeq[pos+1]

        score = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                v = self.mat[i,j]                
                if v >= thres:
                    weight = float(math.log(v,2)) / 4
                    neighborSet = self.FindNeighborSet(i,j)
                    seqId = vSeq.index(v)
                    if rSeq[seqId] == 1:
                        flag = -1
                        if np.sum( np.isin(neighborSet,vSeq[seqId+1]) ): 
                        # next number is in the neighbor set
                            if seqId == 0:
                                flag = 1
                            else:
                                if np.sum( np.isin(neighborSet,vSeq[seqId-1]) ): 
                                # prev number is in the neighbor set
                                    flag = 1
                    if rSeq[seqId] > 1:
                        flag = -1
                        numSame =  np.sum( np.isin(neighborSet,v) )
                        if numSame >= 2: 
                        # there is more than 1 same numbers in the neighbor set
                            flag = 1
                        if numSame == 1:
                            if seqId == 0:
                            # v is the largest
                                if len(neighborSet) == 2 or np.sum( np.isin(neighborSet,vSeq[seqId+1]) ): 
                                # v is in corner or next number is in the neibhgorset
                                    flag = 1
                            else:
                                if np.sum( np.isin(neighborSet,vSeq[seqId-1]) ) or np.sum( np.isin(neighborSet,vSeq[seqId+1]) ):
                                # next number or prev number is in the neibhgorset
                                    flag = 1
                    score[i,j] = flag * weight

        orderedScore = score.reshape(16)[argsortIdx]
        corrList = (orderedScore > 0).tolist()
        nCorr = corrList.index(False) + 1
        nWrong = np.sum( orderedScore!=0 ) - nCorr
        
        return score, nCorr, nWrong


    def RewardFunctionV0(self, action, extraMatScore, prevMax):
        maxThres = 8
        neighborThres = 64
        mergeThres = 32        
        maxFlag, maxCnId = self.CheckMaxValue(maxThres)
        bestMerge = self.CheckMaxPossibleMerge()
        numNeighbor = np.sum(self.mat>=neighborThres) - 1
        if maxFlag == 1:
            numCorrNeighbor = self.CheckNeighbor(neighborThres,maxCnId)
        else: 
            numCorrNeighbor = 0

        numDiffNeighbor = numCorrNeighbor - self.numPrevCorrNeighbor
        self.numPrevCorrNeighbor = numCorrNeighbor
        curMax = np.max(self.mat)

        reward = 0.

        if extraMatScore > 0:
            reward = reward + 0.25

        if extraMatScore < 0:
            reward = reward - 0.25
        
        if curMax > prevMax:
            reward = reward + int( math.log(curMax,2) ) / 4
        
        reward = reward 
        return reward

    def RewardFunctionV1(self, action, extraMatScore, prevMax):
        maxThres = 8
        neighborThres = 64
        mergeThres = 32        
        maxFlag, maxCnId = self.CheckMaxValue(maxThres)
        bestMerge = self.CheckMaxPossibleMerge()
        numNeighbor = np.sum(self.mat>=neighborThres) - 1
        if maxFlag == 1:
            numCorrNeighbor = self.CheckNeighbor(neighborThres,maxCnId)
        else: 
            numCorrNeighbor = 0

        numDiffNeighbor = numCorrNeighbor - self.numPrevCorrNeighbor
        self.numPrevCorrNeighbor = numCorrNeighbor
        curMax = np.max(self.mat)

        reward = 0.
        reward = reward + maxFlag

        if numCorrNeighbor == numNeighbor:
            reward = reward + 1 + max(0,numDiffNeighbor)
        else:
            reward = reward - 1 + numDiffNeighbor

        if extraMatScore > 0:
            reward = reward + 0.25

        if extraMatScore < 0:
            reward = reward - 0.25
        
        if curMax > prevMax:
            reward = reward + int( math.log(curMax,2) ) / 4
        # if extraMatScore >= bestMerge and bestMerge >= mergeThres:
        #     reward = reward + int( math.log(bestMerge,2) )

        # reward = reward + ( numCorrNeighbor - self.numPrevCorrNeighbor )
        # if self.maxScore >= neighborThres and numCorrNeighbor == 0:
        #     reward = reward - 1
        # if action == 0:
        #     reward = reward - 0.5
        
        reward = reward 
        return reward

    def RewardFunctionV2(self, action, extraMatScore, prevMax):    	
        maxThres = 8
        neighborThres = 8
        mergeThres = 32        
        maxFlag, maxCnId = self.CheckMaxValue(maxThres)
        bestMerge = self.CheckMaxPossibleMerge()
        curMax = np.max(self.mat)
        neighborScoreMat = self.CheckNeighborV2(8)
        curNeighborScore = np.sum( neighborScoreMat )
        
        reward = 0.
        reward = reward + maxFlag * 5

        if extraMatScore > 0:
            reward = reward + 0.25

        if extraMatScore < 0:
            reward = reward - 0.25
        
        if curMax > prevMax:
            reward = reward + math.log(curMax,2) / 4

        if extraMatScore >= bestMerge and bestMerge >= mergeThres:
            reward = reward + math.log(bestMerge,2) / 4

        reward = reward + ( curNeighborScore - self.prevNeighborScore ) / 4
        
        reward = reward 

        self.prevNeighborScore = curNeighborScore
        return reward        

    def RewardFunctionV3(self, action, extraMatScore, prevMax):
        # prev_reward_state: prevOrderState, prevMaxState, prevNeighborScore, nPrevCorrNbr, nPrevWrongNbr
        # Reward State
        #   4 - Perfect: The position of max is correct, all NBRs are correct
        #   3 - Perfect: The position of max is correct, bad NBRs are non larger than 4
        #   2 - Perfect: The position of max is correct, bad NBRs are larger than 4
        #   1 - Perfect: The position of max is wrong

        maxThres = 8
        neighborThres = 8
        mergeThres = 32        
        alpha = 0.5
        beta = 0.25

        # Compute Order State  
        curMaxFlag, maxCnId = self.CheckMaxValue(maxThres)
        nCorr, nWrong = self.CheckNeighbor(8,maxCnId)
        curOrderState = 0
        prevOrderState = self.prev_reward_state[0]
        if curMaxFlag == 1:
            if nWrong == 0:
                curOrderState = 2
            elif nWrong <= 4:
                curOrderState = 1
            else:
                curOrderState = -1
        if curMaxFlag == -1:
            curOrderState = -2

        rewardOrder = 0.
        rewardGain = 0.
        # reward based on order state        
        if curOrderState != prevOrderState: 
        # state change
            rewardGain = alpha * ( curOrderState - prevOrderState )
            if action == 0:
                if rewardGain > 0:
                    rewardGain = rewardGain / 2
                if rewardGain == 0:
                    rewardGain = rewardGain - 1
                if rewardGain < 0:
                    rewardGain = rewardGain * 2            
        else:
        # state not change
            if action != 0:
                rewardGain = beta * curOrderState
            if action == 0:
                if curOrderState >= 0:
                    rewardGain = -1 * beta
                else:
                    rewardGain = 2 * beta * curOrderState

        reward = rewardGain

        self.prev_reward_state[0] = curOrderState
        self.prev_reward_state[1] = curMaxFlag
        # self.prev_reward_state[2] = scoreMat
        self.prev_reward_state[2] = nCorr
        self.prev_reward_state[3] = nWrong

        return reward    

    def CheckTerminal(self):
        possibleAction = self.CheckPossibleAction()
        terminal = 0
        if np.sum(possibleAction) == 0:
            terminal = 1
        if np.sum(possibleAction) == 1 and possibleAction[0] == 1:
            if self.maxScore < 8192:
                terminal = 1
            else:
                if len(self.sumScoreSeq) > 100 and self.sumScoreSeq[-1] - self.sumScoreSeq[-100] <= 64:
                    terminal = 1            
            terminal = 1

        return terminal

    def CheckTerminalV3(self):
        possibleAction = self.CheckPossibleAction()
        terminal = 0
        if np.sum(possibleAction) == 0:
            terminal = 1
        if np.sum(possibleAction) == 1 and possibleAction[0] == 1:
            if self.maxScore < 8192:
                terminal = 1

        curSum = np.sum(self.max)
        if curSum >= 32 and curSum > self.step:
            terminal = 1

        return terminal

    def GenerateImg(self):
        img = np.ones( (1125,1125,3), np.uint8 ) * 173
        margin = 25
        edge = 250
        for i in range(4):
            for j in range(4):
                v = int(self.mat[i][j])
                if v == 0:
                    idx = 0
                else:
                    idx = int(math.log(v,2))
                # draw rectangle
                img[(i+1)*margin+i*edge:(i+1)*margin+(i+1)*edge,(j+1)*margin+j*edge:(j+1)*margin+(j+1)*edge,0] = self.colorList[idx][2]
                img[(i+1)*margin+i*edge:(i+1)*margin+(i+1)*edge,(j+1)*margin+j*edge:(j+1)*margin+(j+1)*edge,1] = self.colorList[idx][1]
                img[(i+1)*margin+i*edge:(i+1)*margin+(i+1)*edge,(j+1)*margin+j*edge:(j+1)*margin+(j+1)*edge,2] = self.colorList[idx][0]
                # put text
                if v is not 0:
                    cv2.putText(img, str(v), ((j+1)*margin+j*edge+25,(i+1)*margin+i*edge+150), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 5, 8, 0)
        
        npImg = np.copy(img)
        npImg[:,:,0] = img[:,:,2]
        npImg[:,:,2] = img[:,:,0]
        tensorImg = torch.from_numpy(img.astype('float')) / 255
        tensorImg = tensorImg.permute(2,0,1)
        return tensorImg, npImg
