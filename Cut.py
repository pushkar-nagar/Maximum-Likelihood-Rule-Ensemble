import pandas as pd
import numpy as np
import math
from Rule import Rule
class Cut(Rule):
    def __init__(self, K = None, N = None):
        Rule.__init__(self)
        self.EPSILON = 1e-8
        self.R = 0.5
        self.rp = .5
        self.lineSearchMax = 4.0
        self.lineSearchPrecision = 1e-4
        self.N = N
        self.K = K
        self.maxk = 0
        self.hessian = self.R
        self.gradient = 0.0
        self.lineSearchIterations = 10
        self.lineSearchMax = 4.0
        self.EPSILON = 1e-8
        '''
        Cut Parameter
        '''
        self.decision = np.zeros(self.K)
        self.position = -1
        self.direction = 0
        self.value = 0
        self.empiricalRisk = 0
        self.exists = False

    def initializeCut(self, RuleBuilderObject):
        RuleBuilderObject.hessian = RuleBuilderObject.R
        RuleBuilderObject.gradient = 0
        RuleBuilderObject.hessians.fill(self.R)
        RuleBuilderObject.gradients.fill(0)

    def copyCut(self, bestcut):
        for i in range(self.K):
            self.decision[i] = bestcut.decision[i]
        self.position = bestcut.position
        self.direction = bestcut.direction
        self.value = bestcut.value
        self.exists = bestcut.exists
        self.empiricalRisk = bestcut.empiricalRisk

    def saveCut(self, cutDirection, currentValue, value, tempEmpiricalRisk):
        self.direction = cutDirection
        self.value = (currentValue + value) / 2
        self.empiricalRisk = tempEmpiricalRisk
        self.exists = True

    def computeCurrentEmpiricalRisk(self, position, weight, RuleBuilderObject=None):
        if(RuleBuilderObject.prechosenK):
            if RuleBuilderObject.traindf.label.iloc[position] == RuleBuilderObject.maxk:
                RuleBuilderObject.gradient += RuleBuilderObject.traindf.iloc[position].weight * weight
            RuleBuilderObject.gradient -= RuleBuilderObject.traindf.iloc[position].weight * weight * RuleBuilderObject.proba[position][RuleBuilderObject.maxk]

            if(RuleBuilderObject.method == 0):
                return -RuleBuilderObject.gradient
            else:
                RuleBuilderObject.hessian += RuleBuilderObject.traindf.iloc[position].weight* (RuleBuilderObject.rp + RuleBuilderObject.proba[position][RuleBuilderObject.maxk] * (1 - RuleBuilderObject.proba[position][RuleBuilderObject.maxk])) * weight
                return -RuleBuilderObject.gradient * (math.fabs(RuleBuilderObject.gradient) / RuleBuilderObject.hessian)
        else:
            y = RuleBuilderObject.traindf.label.iloc[position]
            for k in range(RuleBuilderObject.K):
                if y == k:
                    RuleBuilderObject.gradients[k] += RuleBuilderObject.traindf.iloc[position].weight * weight
                RuleBuilderObject.gradients[k] -= RuleBuilderObject.traindf.iloc[position].weight * weight * RuleBuilderObject.proba[position][k]
                if (RuleBuilderObject.method == 1):
                    RuleBuilderObject.hessians[k] += RuleBuilderObject.traindf.iloc[position].weight * (RuleBuilderObject.rp + RuleBuilderObject.proba[position][k] * (1 - RuleBuilderObject.proba[position][k])) * weight
            if(RuleBuilderObject.method == 0):
                return -np.max(RuleBuilderObject.gradients)
            else:
                highest = RuleBuilderObject.gradients[0] * (math.fabs(RuleBuilderObject.gradients[0]) / RuleBuilderObject.hessians[0])
                for k in range(1,RuleBuilderObject.K):
                    if (RuleBuilderObject.gradients[k] * (math.fabs(RuleBuilderObject.gradients[k]) / RuleBuilderObject.hessians[k])) > highest:
                        highest = RuleBuilderObject.gradients[k] * (math.fabs(RuleBuilderObject.gradients[k]) / RuleBuilderObject.hessians[k])

            return -highest

    def findBestCut(self, attributeIdx, coveredInstances, RuleBuilderObject = None):

        '''intialze another object of Cut Class , for object comparison
         how to get intialize reference again here for Cut class
         tricky part to understand the code
        '''
        tempEmpiricalRisk = 0
        for cutDirection in range(-1,3,2):
            self.initializeCut(RuleBuilderObject)
            currentPosition = 0
            i = 0
            if cutDirection == self.GREATER_EQUAL:
                i = self.N - 1
            else:
                i = 0

            while((cutDirection == self.GREATER_EQUAL and i >= 0) or (cutDirection != self.GREATER_EQUAL and i < RuleBuilderObject.N)):
                currentPosition = RuleBuilderObject.invertedList[attributeIdx][i]
                #import pdb; pdb.set_trace()
                if (coveredInstances[currentPosition] > 0 and RuleBuilderObject.traindf.iloc[currentPosition,attributeIdx] != None):
                    break
                if cutDirection == self.GREATER_EQUAL:
                    i -= 1
                else:
                    i += 1
            currentValue = RuleBuilderObject.traindf.iloc[currentPosition,attributeIdx]
            while((cutDirection == self.GREATER_EQUAL and i >= 0) or (cutDirection != self.GREATER_EQUAL and i < RuleBuilderObject.N)):
                nextPosition = RuleBuilderObject.invertedList[attributeIdx][i]
                #import pdb; pdb.set_trace()
                if coveredInstances[nextPosition] > 0 and RuleBuilderObject.traindf.iloc[nextPosition,attributeIdx] != None:
                    value = RuleBuilderObject.traindf.iloc[nextPosition,attributeIdx]
                    if currentValue != value:
                        if (tempEmpiricalRisk < self.empiricalRisk + RuleBuilderObject.EPSILON):
                            self.saveCut(cutDirection, currentValue, value, tempEmpiricalRisk)

                    tempEmpiricalRisk = self.computeCurrentEmpiricalRisk(nextPosition, coveredInstances[nextPosition], RuleBuilderObject)
                    currentValue =  RuleBuilderObject.traindf.iloc[nextPosition,attributeIdx]

                if cutDirection == self.GREATER_EQUAL:
                    i -= 1
                else:
                    i += 1

        ''' Here we have to return cut object reference which save best cut 
            again tricky part to handle
        '''
        return self













