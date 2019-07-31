# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#import matplotlib.pyplot as plt

def loadDataSet(data_dir):
    dataMat = np.zeros((0,5))
    labelMat =np.zeros((0,1))
    fr = open(data_dir)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat=np.append(dataMat,[[1.0,float(lineArr[0]), float(lineArr[1]),float(lineArr[2]),float(lineArr[3])]],0)
        labelMat=np.append(labelMat,[[int(lineArr[4])]],0)
    return dataMat, labelMat

def sigmoid(x):
    return 1.0/(1+np.exp(-x)) #定义sigmoid函数
 
def grad_desc(dataMat,labelMat):
    m,n = np.shape(dataMat)
    a=0.001
    times=40
    weight = np.ones((n,1))
    for i in range(times):
        h= sigmoid(np.dot(dataMat,weight))
        error = h-labelMat
        weight = weight-a*np.dot(dataMat.T,error)
    return weight

def  random_grad_desc(dataMat,labelMat):
    a=0.001
    times = 90
    m,n = np.shape(dataMat)
    weight=np.ones((n,1))
    for i in range(times):
        for j in range(m):
            h=sigmoid(sum(np.dot(dataMat[j,:],weight)))
            error=h-labelMat[j]
            weight = weight-a*dataMat[j,:]*error
    return weight
      
def LR_classify(x,weight):
    prob = sigmoid(sum(np.dot(x,weight)))       
    if prob > 0.5:
        return 1
    else:
        return 0

def LR_test(dataMat,labelMat,weight):
    errorCount = 0
    numTest = 0
    for i in range(dataMat.shape[0]):
        numTest += 1
        if LR_classify(dataMat[i,:],weight) != labelMat[i]:
            errorCount += 1
    errorRate = (float(errorCount)/numTest)
    return errorRate
   
if __name__ == '__main__':
    data_dir='data_banknote_authentication.txt'
    dataMat, labelMat = loadDataSet(data_dir)
    #print(dataMat, labelMat)
    #dataMat=np.array(dataMat)
    #labelMat=np.array(labelMat)
    dataMat_train = dataMat[0:int(.8*dataMat.shape[0]),:]
    dataMat_test = dataMat[int(.8*dataMat.shape[0]):,:]
    labelMat_train = labelMat[0:int(.8*labelMat.shape[0])]
    labelMat_test = labelMat[int(.8*labelMat.shape[0]):]
    
    weights=grad_desc(dataMat_train,labelMat_train)
##    weights=random_grad_desc(dataMat_train,labelMat_train)
    errorRate=LR_test(dataMat_test,labelMat_test,weights)
    print ("the error rate of this test is :",errorRate)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
