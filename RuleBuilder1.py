
import pandas as pd
import numpy as np
import math
from Cut import Cut
from Rule import Rule
class RuleBuilder:
    def __init__(self, subsample,  nrules, traindf = None, testdf = None, shrink = .5, useLineSearch = False, method = 0, prechosenK = False):
        self.nrules = nrules
        self.shrink = shrink
        self.subsample = subsample
        self.method = method

        self.traindf = traindf
        self.textdf = testdf
        '''self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY'''
        self.EPSILON = 1e-8
        self.R = 0.5
        self.rp = .5
        self.lineSearchMax = 4.0
        self.lineSearchPrecision = 1e-4
        self.Rules = list()
        self.N = len(traindf)
        self.D =  len(traindf.columns) - 1  
        # -1 since, dataframe contains label column also

        self.K = len(traindf["label"].unique())
        #self.coveredInstances = np.zeros(N)
        self.defaultRule = None
        self.func = None
        self.resample = True
        self.useLineSearch = useLineSearch

        self.invertedList = np.zeros(self.D,self.K)
        self.attributeNames = list(self.traindf.columns.values)
        self.attributeNames.remove("label")
        for attribute,attidx in enumerate(self.attributeNames):
            self.traindf = self.traindf.sort_values(by=[attribute], ascending=True)
            self.invertedList[attidx,:] = self.traindf.index

        self.traindf = self.traindf.sort_index(inplace=True)

        self.proba = np.zeros(self.N,self.K)
        self.gradients = np.zeros(self.K)
        self.hessians = np.zeros(self.K)
        self.maxk = 0
        self.hessian = self.R
        self.gradient = 0.0
        self.prechosenK = prechosenK
        self.lineSearchIterations = 10
        self.lineSearchMax = 4.0
        self.EPSILON = 1e-8

    def initializeForRule(self, func, coveredInstances):
        self.func = func

        if(self.prechosenK):
            self.gradients = self.gradients.fill(0.0)
            self.hessians = self.hessians.fill(self.R)

        for i in range(self.N):
            if coveredInstances[i] >= 0:
                norm = 0
                for k in range(self.K):
                    self.proba[i][k] = np.exp(self.func[i][k])  
                    norm = norm + self.proba[i][k]
                for k in range(self.K):
                    self.proba[i][k] = self.proba[i][k] / norm
                    if(self.prechosenK):
                        self.gradients[k] -= self.traindf.iloc[i].weight * self.proba[i][k]
                        self.hessians[k] += self.traindf.iloc[i].weight* (self.rp + self.proba[i][k] * (1 - self.proba[i][k]))
                if(self.prechosenK):
                    self.gradients[self.traindf.iloc[i].label] += self.traindf.iloc[i].weight

        if(self.prechosenK):
            maxk = 0
            if(self.method == 0):
                maxk = np.argmax(self.gradients)
            else:
                for k in range(self.K):
                    if (self.gradients[k] / math.sqrt(self.hessians[k])) > (self.gradients[maxk] / math.sqrt(self.hessians[maxk])):
                        maxk = k


    def computeDecision(self, coveredInstances):
        
        if(self.prechosenK):
            self.hessian = self.R
            self.gradient = 0.0
            for i in range(self.N):
                if coveredInstances[i] >= 0:
                    if self.traindf.iloc[i].label == self.maxk: 
                        self.gradient += self.traindf.iloc[i].weight

                    self.gradient -= self.traindf.iloc[i].weight * self.proba[i][self.maxk]
                    self.hessian += self.traindf.iloc[i].weight* (self.rp + self.proba[i][self.maxk] * (1 - self.proba[i][self.maxk]))
            if self.gradient <= 0:
                return None

            alphaNR = self.gradient / self.hessian
            decision = np.zeros(self.K)
            decision = decision.fill(-(alphaNR / self.K))
            decision[self.maxk] = alphaNR * ((self.K - 1) / self.K)
            return decision
        else:
            self.gradients = self.gradients.fill(0.0)
            self.hessians = self.hessians.fill(self.R)
            chosenK = 0
            origgGradients = np.zeros(self.K)
            for i in range(self.N):
                if coveredInstances[i] >= 0:
                    for k in range(self.K):
                        if self.traindf.iloc[i].label == k:
                            self.gradients[k] += self.traindf.iloc[i].weight
                            origgGradients[k] += self.traindf.iloc[i].weight * coveredInstances[i]

                        self.gradients[k] -= self.traindf.iloc[i].weight * self.proba[i][k]
                        origgGradients[k] -= self.traindf.iloc[i].weight * self.proba[i][k]
                        self.hessians[k] += self.traindf.iloc[i].weight* (self.rp + self.proba[i][k] * (1 - self.proba[i][k]))

            for i in range(self.K):
                if origgGradients[k] > origgGradients[chosenK]:
                    chosenK = k
            if self.gradients[chosenK] <= 0:
                return None
            alphaNR = self.gradients[chosenK] / self.hessians[chosenK]
            decision = np.zeros(self.K)
            decision = decision.fill(-(alphaNR / self.K))
            decision[chosenK] = alphaNR * ((self.K - 1) / self.K)   

            return decision    
            

    def markedCoveredInstances(self, bestAttribute, coveredInstances, bestCut = None):
        for i in range(self.N):
            if coveredInstances[i] != -1:
                if self.traindf.iloc[i, bestAttribute] == -1:
                    coveredInstances[i] = -1
                else:
                    value = self.traindf.iloc[i,bestAttribute] 
                    if (value < bestCut.value and bestCut.direction == 1) or (value > bestCut.value and bestCut.direction == -1):
                        coveredInstances[i] = -1

        return coveredInstances            
    
    def computeRuleGradient(self, func, coveredInstances, point):
        size = 0
        gradient = 0
        for i in range(self.N):
            if coveredInstances[i] >= 0:
                size += 1
                self.proba[i] = np.exp(func[i])
                self.proba[i][self.maxk] = np.exp(func[i][self.maxk] + point)
                if self.traindf.iloc[i].label == self.maxk:
                    gradient += self.traindf.iloc[i].weight
                gradient -= self.traindf.iloc[i].weight * (self.proba[i][self.maxk] / np.sum(self.proba[i])) 
        return (gradient / size)

    def getLineSearchDecision2(self, func, coveredInstances, left, right, depth):
        middle = (left + right) / 2
        gradient = self.computeRuleGradient(func, coveredInstances, middle)
        if (abs(gradient) <= self.lineSearchPrecision)  or (depth == self.lineSearchIterations):
            decision = np.zeros(self.K, dtype = np.float)
            decision = decision.fill(-middle / self.K)
            decision[self.maxk] = -middle * (self.K - 1)  / self.K
            return decision
        else:
            if gradient > 0:
                return self.getLineSearchDecision2(func, coveredInstances, middle, right, depth + 1)
            else:
                return self.getLineSearchDecision2(func, coveredInstances, left, middle, depth + 1)  





    def getLineSearchDecision(self, func, coveredInstances):
        gradient = self.computeRuleGradient(func, coveredInstances, self.lineSearchMax)
        if gradient >= 0:
            decision = np.zeros(self.K, dtype = np.float)
            decision =decision.fill(-self.lineSearchMax / self.K)
            decision[self.maxk] = self.lineSearchMax * (self.K - 1 ) / self.K
            return decision
        else:
            self.getLineSearchDecision2(func, coveredInstances, 0, self.lineSearchMax, 1)    

    ''' write full fucntion '''
    def createDefaultRule1(self):
        priors = np.zeros(self.K)
        for i in range(len(self.N)):
            priors[int(self.traindf.iloc[i].label)] += 1
        emptyClasses = 0
        for k in range(len(self.K)):
            priors[k] /= self.N
            if priors[k] == 0:
                emptyClasses += 1

        logPriors = 0
        for k in range(len(self.K)):
            if priors[k] != 0:
                logPriors += math.log(priors[k])
        logPriors /= (self.K - emptyClasses)

        decision = np.zeros(self.K)
        decision = decision.fill(-logPriors)
        for k in range(len(self.K)):
            if priors[k] != 0:
                decision[k] += math.log(priors[k])
            else:
                decision[k] = 0

        return decision        







    def createDefaultRule(self, func, coveredInstances):
        
        self.maxk = self.initializeForRule(func, coveredInstances)
        decision = self.computeDecision(coveredInstances)
        for i in range(len(decision)):
            decision[i] *= self.shrink

        return decision


    def createRule(self, func, coveredInstances):

        self.maxk = self.initializeForRule(func, coveredInstances)


        ''' Rule class for saving the generated rules '''
 
        rule = Rule()    

        bestCut = Cut(self.K)
        bestCut.empiricalRisk = 0
        creating = True
        while(creating):
            bestAttribute = -1
            cut = Cut(self.K)
            for j in range(self.D):
                getCut = cut.findBestCut(j, coveredInstances, self)
                if getCut.empiricalRisk < (bestCut.empiricalRisk - self.EPSILON):
                    bestCut.copyCut(getCut)
                    bestAttribute = j
                if bestAttribute == -1 and bestCut.exists == False:
                    creating = False    
                else:
                    ''' bestCut.attribute name of best attribute '''    ''' adding attribute name''' 
                    rule.addSelector(bestAttribute, bestCut.value, bestCut.direction, self.attributeNames[bestAttribute])            
                    coveredInstances = self.markedCoveredInstances(bestAttribute, coveredInstances, bestCut)

        if bestCut.exists == True:
            decision = None
            if self.useLineSearch == True:
                decision = self.getLineSearchDecision(func, coveredInstances)
            else:
                decision = self.computeDecision(coveredInstances)
            if decision == None:
                return None
            else:
                for i in range(self.K):
                    decision[i] = decision[i] * self.shrink
                rule.decision = decision
                return rule
        else:
            return None

if __name__ == "__main__":