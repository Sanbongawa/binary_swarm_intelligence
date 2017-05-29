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

print("Algorithm:\n\t{0}  {1} {2}".format("best_pos","best_val","number_of_1s"))

s,g,l=opt.BGA(Eval_Func=Evaluate, n=20, m_i=200)
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

<html>

<table width=600 height=60>
<tr align="center" >
  <th align="center" colspan=2> default </th>
  <th align="center">data Sonar</th>
  <th align="center" colspan=2 > propose </th>
</tr>
<tr>
  <td>accuracy</td>
  <td>number of feature</td>
  <td>algorithm</td>
  <td>accuracy</td>
  <td>number of feature</td>
</tr>
<tr align="center" >
  <td>0.84656</td>
  <td>32.48</td>
  <td>BGA</td>
  <td>0.84072</td>
  <td>21.22</td>
</tr>
<tr align="center" >
  <td>0.88000</td>
  <td>31.35</td>
  <td>BPSO</td>
  <td>0.88296</td>
  <td>18.40</td>
</tr>
<tr align="center" >
  <td>0.84400</td>
  <td>37.83</td>
  <td>BCS</td>
  <td>0.83456</td>
  <td>30.83</td>
</tr>
<tr align="center" >
  <td>0.83512</td>
  <td>18.73</td>
  <td>BFFA</td>
  <td>0.82480</td>
  <td>9.53</td>
</tr>
<tr align="center" >
  <td>0.88224</td>
  <td>30.24</td>
  <td>BBA</td>
  <td>0.84472</td>
  <td>18.16</td>
</tr>
<tr align="center" >
  <td>0.84136</td>
  <td>31.41</td>
  <td>BGSA</td>
  <td>0.82712</td>
  <td>22.68</td>
</tr>

<tr align="center" >
  <td>0.86624</td>
  <td>30.77</td>
  <td>BDFA</td>
  <td>0.86704</td>
  <td>20.08</td>
</tr>
</table>

</html>
