# binary_swarm_intelligence
 Library of binary swarm intelligence mainly used for obtaining optimal solution of feature selection

This Python library is a summary of the algorithms I used for my graduation research at bachelor.


## The implemented algorithm
* Binary Genetic Algorithm
* Binary Particle Swarm optimization
* Binary Cuckoo Search
* Binary Firefly algorithm
* Binary Bat Algorithm
* Binary Gravitational Search algorithm
* Binary Dragon Fly Algorithm

#### Example to use
1. Import library.
2. Set evaluation function in class.
3. Specify the class as the argument of the algorithm.
4. If possible parameter setting.

#### Example code: Suppose svm is used
```python
import binary_optimization as opt#import library
import numpy as np
from sklearn import svm

class Evaluate:#setting class
    def __init__(self):#set train_data,train_label
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.K = 4
    def evaluate(self,gen):
        """
        Setting of evaluation function.
        Here, the correct answer rate is used.
          anser_label/all_label
        """
        mask = np.array(gen) > 0
        al_data = tr_x[:,np.nonzero(mask)[0]]
        kf = ms.KFold(n_split=self.K, shuffle=True);s=0
        for tr_ix, te_ix in kf.split(self.tr_x):          
          s += svm.LinearSVC().fit(al_data[tr_ix],self.tr_y[tr_ix]).score(al_data[te_ix], self.tr_y[te_ix])
        return s/self.K
        #↑evaluate with fittness function
    def check_dimentions(self,dim):#check number of all feature
        if dim==None:
            return len(self.train_d[0])
        else:
            return dim

print("Algorithm:\n\t{0}  {1} {2}".format("best_pos","best_val","number_of_1s"))

s,g,l=opt.BGA(Eval_Func=Evaluate, n=20, m_i=200)#score, gen_list, gen length of 1
print("BGA:\n\t{0}   {1}  {2}".format("".join(map(str,g)),s,l))

```

##### common arguments with algorithms
* Eval_Func: Evaluate function (class)
* n: number of population (int)
* m_i:  number of max iteration(int)
* dim: number of all feature(int)
* minf: minimization flag. min or max?(bool)
* prog: Do you use progress bar?(bool)<p>
<br><p>

##### Additional notes
In my research, I proposed reducing the feature dimension from half to one-third without substantially decreasing the evaluation by giving a penalty equivalent to the number of features to the evaluation formula at feature selection.
