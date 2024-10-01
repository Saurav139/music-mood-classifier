# -*- coding: utf-8 -*-
import numpy as np
import argparse
import warnings
from time import time
from SparseVector import SparseVector


def readBeta(input):
    """ Read a vector β from file input. Each line contains pairs of the form:
                (feature,value)
    """
    beta = SparseVector({})
    with open(input,'r') as fh:
        for  line in fh:
            (feat,val) = eval(line.strip())
            beta[feat] = val
    return beta

def writeBeta(output,beta):
    """ Write a vector β to a file output.  Each line contains pairs of the form:
                (feature,value)
 
    """
    with open(output,'w') as fh:
        for key in beta:
            fh.write('(%s,%f)\n' % (key,beta[key]))

def readData(input_file):
    """  Read data from an input file. Each line of the file contains tuples of the form

                    (x,y)  

         x is a dictionary of the form:                 

           { "feature1": value, "feature2":value, ...}

         and y is a binary value +1 or -1.

         The result is stored in a list containing tuples of the form
                 (SparseVector(x),y)             

    """ 
    listSoFar = []
    with open(input_file,'r') as fh:
        for line in fh:
            (x,y) = eval(line)
            x = SparseVector(x)
            listSoFar.append((x,y))

    return listSoFar

def getAllFeatures(data):
    """ Get all the features present in dataset data.
    """
    features = SparseVector({})
    for (x,y) in data:
        features = features + x
    return features.keys() 

def logisticLoss(beta,x,y):
    """
        Given sparse vector beta, a sparse vector x, and a binary value y in {-1,+1}, compute the logistic loss
               
                l(β;x,y) = log( 1.0 + exp(-y * <β,x>) )

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.log(1. + np.exp (-y * beta.dot(x)))

def gradLogisticLoss(beta,x,y):
    """
        Given a sparse vector beta, a sparse vector x, and 
        a binary value y in {-1,+1}, compute the gradient of the logistic loss 

              ∇l(B;x,y) = -y / (1.0 + exp(y <β,x> )) * x
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return - 1.* y / (1. + np.exp(y*beta.dot(x)) ) * x

def totalLoss(data,beta,lam = 0.0):
    """  Given a sparse vector beta and a dataset  compute the regularized total logistic loss :
              
               L(β) = Σ_{(x,y) in data}  l(β;x,y)  + λ ||β ||_2^2             
        
         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

         Output is:
            - The loss L(β) 
    """
    loss = 0.0
    for pair in data:
        loss += logisticLoss(beta, pair[0], pair[1])
    loss += lam * beta.dot(beta)
    return loss

def gradTotalLoss(data,beta, lam = 0.0):
    """  Given a sparse vector beta and a dataset perform compute the gradient of regularized total logistic loss :
            
              ∇L(β) = Σ_{(x,y) in data}  ∇l(β;x,y)  + 2λ β   
        
         Inputs are:
            - data: a python list containing pairs of the form (x,y), where x is a sparse vector and y is a binary value
            - beta: a sparse vector β
            - lam: the regularization parameter λ

         Output is:
            - The gradient ∇L(β) 
    """    
    loss = SparseVector({})
    for pair in data:
        loss += gradLogisticLoss(beta, pair[0], pair[1])
    loss += 2 * lam * beta
    return loss

def lineSearch(fun,x,grad,a=0.2,b=0.6):
    """ Given function fun, a current argument x, and gradient grad=∇fun(x), 
        perform backtracking line search to find the next point to move to.
        (see Boyd and Vandenberghe, page 464).

        Both x and grad are presumed to be SparseVectors.
        
        Inputs are:
            - fun: the objective function f.
            - x: the present input (a Sparse Vector)
            - grad: the present gradient ∇f(x) (as Sparse Vector)
            - Optional parameters a,b  are the parameters of the line search.

        Given function fun, and current argument x, and gradient grad=∇fun(x), the function finds a t such that
        fun(x - t * ∇f(x)) <= f(x) - a * t * <∇f(x),∇f(x)>

        The return value is the resulting value of t.
    """
    t = 1.0

    fx = fun(x)
    gradNormSq = grad.dot(grad)

    while fun(x-t*grad) > fx- a * t * gradNormSq:
        t = b * t
    return t 
    
def basicMetrics(data,beta):
    """ Output the quantities necessary to compute the accuracy, precision, and recall of the prediction of labels in a dataset under a given β.
        
        The accuracy (ACC), precision (PRE), and recall (REC) are defined in terms of the following sets:

                 P = datapoints (x,y) in data for which <β,x> > 0
                 N = datapoints (x,y) in data for which <β,x> <= 0
                 
                 TP = datapoints in (x,y) in P for which y=+1  
                 FP = datapoints in (x,y) in P for which y=-1  
                 TN = datapoints in (x,y) in N for which y=-1
                 FN = datapoints in (x,y) in N for which y=+1

        Inputs are:
             - data: an RDD containing pairs of the form (x,y)
             - beta: vector β

        The return value is a tuple containing
             - #P,#N,#TP,#FP,#TN,#FN
    """
    pairs = ( ( int(np.sign(beta.dot(x))), int(y)) for (x,y) in data  )
    new_pairs = [ (pred_label,pred_label*true_label)  for (pred_label,true_label) in pairs ]        
    

    TP = 1.*new_pairs.count( (1,1) )
    FP = 1.*new_pairs.count( (1,-1) )
    TN = 1.*new_pairs.count( (-1,1) )
    FN = 1.*new_pairs.count( (-1,-1) )
    P = TP+FP
    N = TN+FN
    return P,N,TP,FP,TN,FN 

def metrics(P,N,TP,FP,TN,FN):
    """Regurn the accuracy (ACC), precision (PRE), and recall (REC). These are defined in terms of the following sets:

        For #XXX the number of elements in set XXX, the accuracy, precision, and recall are defined as:
         
                 ACC = ( #TP+#TN ) / (#P + #N)
                 PRE = #TP / (#TP + #FP)
                 REC = #TP/ (#TP + #FN)

        Inputs are:
             - #P,#N,#TP,#FP,#TN,#FN

        The return value is a tuple containing
             - ACC, PRE, REC
    """
    acc = (TP+TN)/(P+N)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
 
    return acc,pre,rec



