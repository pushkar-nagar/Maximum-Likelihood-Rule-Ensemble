import pandas as pd
import numpy as np
from RuleBuilder import RuleBuilder
from numpy.random import seed
from numpy import random
import math
seed(99)

class MLRules():  
    ''' 
    MLRules for generating interpretable decision rules, Only numerical features has been considered in generating rules 
    TO DO : considering categorical features
    Input: 
        traindf = train data in pandas dataframe format with coloumns names, class label column name as "label" 
        testdf = test data in pandas dataframe format with coloumns names, class label column name as "label"
        nrules = Number of rules to generate for classifier
        subsample = resampling of samples from train data for learning rules {0 to 1}
        method = '0' specifies first order optimization method, '1' specifies second order optimization method

    '''

    def __init__(self, nrules = 100, shrink = .5, subsample = .5, method = 0, traindf = None, testdf = None, resample = True):
        self.nrules = nrules
        self.shrink = shrink
        self.subsample = subsample
        self.method = method
        self.traindf = traindf
        self.testdf = testdf
        self.traindf = traindf.infer_objects()
        self.testdf = testdf.infer_objects()
        '''self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY'''
        self.R = 0.5
        self.rp = 0.5
        self.rules = list()
        self.N = len(traindf)
        self.D =  len(traindf.columns) - 1
        self.K = len(traindf["label"].unique())
        self.attributeNames = list(self.traindf.columns.values)
        self.coveredInstances = np.zeros(self.N)
        self.defaultRule = None
        self.func = np.zeros((self.N, self.K))
        self.resample = resample
        self.chooseClass = True
        self.uselineSearch = False
        self.modelBuilt = False     
        self.traindf["weight"] =  random.uniform(0,1,len(self.traindf))
        self.testdf["weight"] = random.uniform(0,1,len(self.testdf))
        self.ruleBuilder = RuleBuilder(subsample = self.subsample, nrules = self.nrules, traindf = self.traindf, testdf = self.testdf, shrink = self.shrink, useLineSearch = self.uselineSearch, method = self.method,  prechosenK = self.chooseClass)
   
        # To do , Create Rule class and make array of objects to save rules
        self.rules = list()
        self.MINUS_INFINITY = -1e40
        self.PLUS_INFINITY = 1e40


    def reSample(self):
        return (self.traindf.sample(self.subsample, replace=False, random_state=1))


    def updateFunction(self, decision):
        for i in range(self.N):
            if self.coveredInstances[i] >= 0:
                for k in range(self.K):
                    self.func[i][k] += decision[k]


    def buildClassifier(self):

        self.coveredInstances.fill(1)
        if(self.uselineSearch):
            self.defaultRule = self.ruleBuilder.createDefaultRule1()
        else:
            self.defaultRule = self.ruleBuilder.createDefaultRule(self.func, self.coveredInstances)
            self.updateFunction(self.defaultRule)

        for rule in range(self.nrules):
            # debugging
            print("Rule Number ", rule)
            if self.resample == True:
                random_idx = np.random.choice(self.N, int(self.N * self.subsample), replace=False)
                for i in random_idx:
                    self.coveredInstances[i] = 1; 
            else:
                self.coveredInstances.fill(1)

            self.rules.append(self.ruleBuilder.createRule(self.func, self.coveredInstances))
            if self.rules[rule] != None:
                self.updateFunction(self.rules[rule].decision)
            else:
                rule = rule - 1
            ##why rule-- has to be perforemd  yet to cleared from code

        self.modelBuilt = True
        # return classifier in compact form
        return self        


    def evaluateF(self, instance):
        evalF = np.zeros(self.K)
        '''
        convert nominal attributes to binary attributes 
        Originial implementation considering only string values as categorical values
        '''

        for k in range(self.K):
            evalF[k] = self.defaultRule[k]
        
        for m in range(self.nrules):
            currentValues = self.rules[m].classifyInstance(instance)
            if currentValues != None:
                for k in range(self.K):
                    evalF[k] += currentValues[k]

        return evalF


        
    def classifyInstance(self, instance):
        evalF = self.evaluateF(instance)
        classIndex = 0
        for k in range(self.K):
            if evalF[k] > evalF[classIndex]:
                classIndex = k
        return classIndex        

    ''' Evaluation has not been done for multiple classes, 
        TO DO Later
        '''        
        
    ''' Calculating Empirical Risk Minimization '''

    def EmpiricalRiskMinimization(self):

        empirical_risk = 0
        for i in range(self.N):
            total = 0
            for k in range(self.K):
                total += np.exp(self.func[i][k])
            empirical_risk -= self.traindf.iloc[i].weight * math.log(math.exp(self.func[i][self.traindf.label.iloc[i]]) / total)

        return empirical_risk / self.N

    def classifyData(self, dataset = None):
        if dataset == None:
            dataset = self.testdf
        pred_labels = list()
        for i in range(self.N):
            pred_labels.append(self.classifyInstance(dataset.iloc[i]))
        return(np.asarray(pred_labels))

    def printDefaultRule(self):
        for i in range(len(self.defaultRule)):
            print("vote for class " , i , " with weight " , self.defaultRule[i] , "\n")

    def describe_classifier(self):
        if self.modelBuilt != True:
            print("Maximum Likelihood Rule Ensembles (MLRules): No model built yet.")
        else:
            print("Maximum Likelihood Rule Ensembles (MLRules)...\n\n" ,
					                               self.nrules , " rules generated.\n" ,
					                               "Default rule:\n" , self.printDefaultRule() , "\n" ,
					                               "List of rules:\n")
        
        for idx,rule in enumerate(self.rules):
            print("Rule ", idx)
            for tuple in rule.expressionRule:   
            
                sign = ""
                if tuple[1] == self.MINUS_INFINITY:
                    sign = " <= " + str("{:.3f}".format(tuple[2]))
                elif tuple[2] == self.PLUS_INFINITY:
                    sign = " >= " + str("{:.3f}".format(tuple[1]))
                else:
                    sign = " in [" + str("{:.3f}".format(tuple[1])) + "," + str("{:.3f}".format(tuple[2]))

                print(" " + tuple[3] + sign )
            print("\n")    

            decision_idx = 0
            while(rule.decision[decision_idx] < 0):
                decision_idx += 1
            print("=> vote for class " + " class attribute ", str(decision_idx), " with weight " + str(rule.decision[decision_idx]))    



                               






