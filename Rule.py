import pandas as pd
import numpy as np

class Rule():
    def __init__(self):
        self.GREATER_EQUAL = 1
        self.LESS_EQUAL = -1
        self.MINUS_INFINITY = -1e40
        self.PLUS_INFINITY = 1e40
        self.decision = None
        self.expressionRule = list()

    ''' need to check attribute is attribute index or attribute name '''

    def addSelector(self, attributeIndex, cutValue, direction, attribute):
        flag = 0
        for tuple in self.expressionRule:
            if tuple[0] == attributeIndex:
                flag = 1
                if direction == self.GREATER_EQUAL:
                    tuple[1] = max(cutValue, tuple[1])
                else:
                    tuple[2] = max(cutValue, tuple[2])

        if flag == 0:
            temp_list = list()
            if direction == self.GREATER_EQUAL:
                temp_list = [attributeIndex, cutValue, self.PLUS_INFINITY, attribute]
                self.expressionRule.append(temp_list)
            else:
                temp_list = [attributeIndex, self.MINUS_INFINITY, cutValue, attribute]
                self.expressionRule.append(temp_list)
                                  
    def classifyInstance(self, instance):
        covered = True
        for rule in self.expressionRule:
            if instance[rule[0]] == -1:
                covered = False 
                break
            if (rule[1] > instance[rule[0]]) and (rule[2] < instance[rule[0]]):
                covered = False
                break
        if covered == True:
            return self.decision
        else:
            return None

