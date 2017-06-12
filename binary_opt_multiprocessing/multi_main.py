from multiprocessing import Pool
"""Using Libraly"""
import numpy as np
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm
import  binary_optimization_multi  as opt

if __name__ == '__main__':
    from sklearn import svm
    from time import time

    np.random.seed(20)
    tr_d=np.array(list(np.random.uniform(-11,9,(500,30)))+list(np.random.uniform(-9,11,(500,30))))
    te_d=np.array(list(np.random.uniform(-10,8,(100,30)))+list(np.random.uniform(-8,10,(100,30))))
    tr_l=np.array([0 for i in range(500)]+[1 for i in range(500)])
    te_l=np.array([0 for i in range(100)]+[1 for i in range(100)])


    class Evaluate:#setting class
        def __init__(self):#set train_data,label,test_data,label
            self.train_l=tr_l
            self.train_d=tr_d
            self.test_l=te_l
            self.test_d=te_d
        def evaluate(self,gen):
            """
            Setting of evaluation function.
            Here, the correct answer rate is used.
              anser_label/all_label
            """
            mask=np.array(gen) > 0
            al_data=np.array([al[mask] for al in self.train_d])
            al_test_data=np.array([al[mask] for al in self.test_d])
            #↑masking with [01]sequence list
            res=svm.LinearSVC().fit(al_data,self.train_l).predict(al_test_data)
            return np.count_nonzero(self.test_l==res)/len(self.test_l)
            #↑evaluate with fittness function
        def check_dimentions(self,dim):#check number of all feature
            if dim==None:
                return len(self.train_d[0])
            else:
                return dim

    alg=["BGA","BPSO","BCS","BFFA","BBA","BGSA","BBA"]
    for al in alg:
        print(al)
        print("Algorithm:\n\t{0}  {1} {2}".format("best_gen","best_val","number_of_dim"))
        start=time()
        for t in range(3):
            exec("s,g,l=opt.{0}(Eval_Func=Evaluate, n=20, m_i=50,prog=True)".format(al))
            #s,g,l=opt.BDFA(Eval_Func=Evaluate, n=20, m_i=50,prog=True)
            print("{3}:\n\t{0}   {1}  {2}".format("".join(map(str,g)),s,l,al))
        print((time()-start)/3)
        print()
        start=time()
        for t in range(3):
            exec("s,g,l=opt.{0}(Eval_Func=Evaluate, n=20, m_i=50,prog=True)".format(al))
            #s,g,l=opt.BDFA(Eval_Func=Evaluate, n=20, m_i=50,mp=3,prog=True)
            print("{3}_M:\n\t{0}   {1}  {2}".format("".join(map(str,g)),s,l,al))
        print((time()-start)/3)
    #print(gd)
