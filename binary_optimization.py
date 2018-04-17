"""Using Libraly"""
import numpy as np
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm

"""Using Algorithm
* Binary Genetic Algorithm
* Binary Particle Swarm optimization
* Binary Cuckoo Search
* Binary Firefly algorithm
* Binary Bat Algorithm
* Binary Gravitational Search algorithm
* Binary Dragon Fly Algorithm
"""

"""Evaluate Function """
class Evaluate:
    def __init__(self):
        None
    def evaluate(self,gen):
        None
    def check_dimentions(self,dim):
        None

"""Common Function"""
def random_search(n,dim):
    """
    create genes list
    input:{ n: Number of population, default=20
            dim: Number of dimension
    }
    output:{genes_list → [[0,0,0,1,1,0,1,...]...n]
    }
    """
    gens=[[0 for g in range(dim)] for _ in range(n)]
    for i,gen in enumerate(gens) :
        r=random.randint(1,dim)
        for _r in range(r):
            gen[_r]=1
        random.shuffle(gen)
    return gens

"""BGA"""
def suddn(li,n_li,num):#突然変異
    l1= [random.choice(n_li) for i in range(num)]
    l2= [random.choice([0,1]) for i in range(num)]
    al_li=dc(li)
    for i in range(len(l1)):
        al_li[l1[i]]=l2[i]
    #li=''.join(_d)
    return al_li

def BGA(Eval_Func,n=20,m_i=300,mutation=0.05,minf=0,dim=None,prog=False):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            mutation: Probability of mutation, default=0.05(5%)
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    gens=random_search(n,dim)
    fit=[0 for i in range(n)]
    num_li=range(dim)
    #flag=dr
    best_val=float("-inf") if minf == 0 else float("inf")#minf==0のときは最大化なので-infを初期ベストにし、全部0の部分集合を初期ベストにする
    best_pos=[0]*dim
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    prop=mutation

    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        for i,gen in enumerate(gens):
            if tuple(gen) in gens_dict:
                v=gens_dict[tuple(gen)]
            else:
                score=estimate(gen)
                gens_dict[tuple(gen)]=score
            fit[i]=score
            if best_val < score if minf==0 else best_val > score:
                best_val=dc(score)
                best_pos=dc(gen)
        alter_gens=sorted(gens,reverse=True)[:2]
        t1=random.randint(1,len(gens[0])-2)
        t2=random.randint(t1,len(gens[0])-1)

        fit_ind=np.argsort(fit)[::-1][:n//2]
        sample_num=random.sample(list(cb(fit_ind,2)),n-2)
        qgens=[suddn(gens[s][:t1]+gens[m][t1:t2]+gens[s][t2:],num_li,dim//3) if np.random.choice([0,1],size=1,p=[1-prop,prop])[0]==1
               else gens[s][:t1]+gens[m][t1:t2]+gens[s][t2:] for s,m in sample_num]
        gens=[]
        gens.extend(qgens)
        gens.append(alter_gens[0])
        gens.append(alter_gens[1])
    return best_val,best_pos,best_pos.count(1)

"""BPSO"""
def logsig(n): return 1 / (1 + math.exp(-n))
def sign(x): return 1 if x > 0 else (-1 if x!=0 else 0)

def BPSO(Eval_Func,n=20,m_i=200,minf=0,dim=None,prog=False,w1=0.5,c1=1,c2=1,vmax=4):
    """
    input:{ 
            Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            w1: move rate, default=0.5
            c1,c2: It's are two fixed variables, default=1,1
            vmax: Limit search range of vmax, default=4
            }

    output:{
            Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    gens=random_search(n,dim)
    pbest=float("-inf") if minf == 0 else float("inf")
    gbest=float("-inf") if minf == 0 else float("inf")
    #vec=3
    #flag=dr
    gens=random_search(n,dim)
    vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    one_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]
    zero_vel=[[random.random()-0.5 for d in range(dim)] for _n in range(n)]

    fit=[float("-inf") if minf == 0 else float("inf") for i in range(n)]
    pbest=dc(fit)
    xpbest=dc(gens)
    #w1=0.5
    if minf==0:
        gbest=max(fit)
        xgbest=gens[fit.index(max(fit))]
    else:
        gbest=min(fit)
        xgbest=gens[fit.index(min(fit))]

    #c1,c2=1,1
    #vmax=4
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        #w=0.5
        for i in range(n):
            if tuple(gens[i]) in gens_dict:
                score=gens_dict[tuple(gens[i])]
            else:
                score=estimate(gens[i])
                gens_dict[tuple(gens[i])]=score
            fit[i]=score
            if fit[i]>pbest[i] if minf==0 else fit[i]<pbest[i]:#max
                pbest[i]=dc(fit[i])
                xpbest[i]=dc(gens[i])

        if minf==0:
            gg=max(fit)
            xgg=gens[fit.index(max(fit))]
        else:
            gg=min(fit)
            xgg=gens[fit.index(min(fit))]

        if gbest<gg if minf==0 else gbest>gg:#max
            gbest=dc(gg)
            xgbest=dc(xgg)

        oneadd=[[0 for d in range(dim)] for i in range(n)]
        zeroadd=[[0 for d in range(dim)] for i in range(n)]
        c3=c1*random.random()
        dd3=c2*random.random()
        for i in range(n):
            for j in range(dim):
                if xpbest[i][j]==0:
                    oneadd[i][j]=oneadd[i][j]-c3
                    zeroadd[i][j]=zeroadd[i][j]+c3
                else:
                    oneadd[i][j]=oneadd[i][j]+c3
                    zeroadd[i][j]=zeroadd[i][j]-c3

                if xgbest[j]==0:
                    oneadd[i][j]=oneadd[i][j]-dd3
                    zeroadd[i][j]=zeroadd[i][j]+dd3
                else:
                    oneadd[i][j]=oneadd[i][j]+dd3
                    zeroadd[i][j]=zeroadd[i][j]-dd3

        one_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(one_vel,oneadd)]
        zero_vel=[[w1*_v+_a for _v,_a in zip(ov,oa)] for ov,oa in zip(zero_vel,zeroadd)]
        for i in range(n):
            for j in range(dim):
                if abs(vel[i][j]) > vmax:
                    zero_vel[i][j]=vmax*sign(zero_vel[i][j])
                    one_vel[i][j]=vmax*sign(one_vel[i][j])
        for i in range(n):
            for j in range(dim):
                if gens[i][j]==1:
                    vel[i][j]=zero_vel[i][j]
                else:
                    vel[i][j]=one_vel[i][j]
        veln=[[logsig(s[_s]) for _s in range(len(s))] for s in vel]
        temp=[[random.random() for d in range(dim)] for _n in range(n)]
        for i in range(n):
            for j in range(dim):
                if temp[i][j]<veln[i][j]:
                    gens[i][j]= 0 if gens[i][j] ==1 else 1
                else:
                    pass
    return gbest,xgbest,xgbest.count(1)

"""BCS"""
def sigmoid(x):
    try:
        return 1/(1+math.exp(-x))
    except OverflowError:
        return 0.000001
def sigma(beta):
    p=math.gamma(1+beta)* math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*(pow(2,(beta-1)/2)))
    return pow(p,1/beta)
def levy_flight(beta,best,est,alpha):
    sg=sigma(beta)
    u=np.random.normal(0,sg**2)
    v=abs(np.random.normal(0,1))
    step=u/pow(v,1/beta)
    step_size=alpha+step#+(step*(est-best))
    new=est+step_size#*np.random.normal()#random.normalvariate(0,sg)
    return new

def BCS(Eval_Func,m_i=200,n=20,minf=0,dim=None,prog=False,alpha=0.1,beta=1.5,param=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            alpha and beta: Arguments in levy flight, default=0.1,1.5
            param: Probability to destroy inferior nest, default=0.25(25%)
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    pa=param
    #flag=dr
    gens=random_search(n,dim)
    fit=[float("-inf") if minf == 0 else float("inf") for _ in range(n)]
    pos=[0 for _ in range(n)]
    g_pos=[0]*dim
    g_val=float("-inf") if minf == 0 else float("inf")
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)
    for it in miter:
        for i,g in enumerate(gens):
            if tuple(g) in gens_dict:
                score=gens_dict[tuple(g)]
            else:
                score=estimate(g)
                gens_dict[tuple(g)]=score
            if score > fit[i] if minf==0 else score < fit[i]:
                fit[i]=score
                pos[i]=g

        maxfit,maxind=max(fit),fit.index(max(fit))
        minfit,minind=min(fit),fit.index(min(fit))
        if minf==0:
            if maxfit > g_val:
                g_val=dc(maxfit)
                g_pos=dc(gens[maxind])
        else:
            if minfit < g_val:
                g_val=dc(minfit)
                g_pos=dc(gens[minind])

        if pa < random.uniform(0,1):
            if minf==0:
                gens[minind]=[0 if 0.5>random.uniform(0,1) else 1 for _ in range(dim)]#rand_gen()
                fit[minind]=float("-inf") if minf == 0 else float("inf")
            else:
                gens[maxind]=[0 if 0.5>random.uniform(0,1) else 1 for _ in range(dim)]#rand_gen()
                fit[maxind]=float("-inf") if minf == 0 else float("inf")


        for g in gens:
            for d in range(dim):
                x=levy_flight(beta,g_pos[d],g[d],alpha)
                if random.uniform(0,1) < sigmoid(x):
                    g[d]=1
                else:
                    g[d]=0
    return g_val,g_pos,g_pos.count(1)

"""BFFA"""
def exchange_binary(binary,score):#,alpha,beta,gamma,r):

    #binary in list
    al_binary=binary
    #movement=move(b,alpha,beta,gamma,r)
    movement=math.tanh(score)
    ##al_binary=[case7(b) if random.uniform(0,1) < movement else case8(b) for b in binary]
    if random.uniform(0,1) < movement:
        for i,b in enumerate(binary):
            al_binary[i]=case7(b)
    else:
        for i,b in enumerate(binary):
            al_binary[i]=case8(b)
    return al_binary

def case7(one_bin):
    return 1 if random.uniform(-0.1,0.9)<math.tanh(one_bin) else 0
def case8(one_bin):
    if random.uniform(-0.1,0.9)<math.tanh(int(one_bin)):
        if one_bin==1:
            return 0
        else:return 1
    else:return one_bin
def case9(one_bin,best):
    if random.uniform(0,1)<math.tanh(int(one_bin)):
        return best
    else:return 0

def BFFA(Eval_Func,n=20,m_i=25,minf=0,dim=None,prog=False,gamma=1.0,beta=0.20,alpha=0.25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    #flag=dr
    global_best=float("-inf") if minf == 0 else float("inf")
    pb=float("-inf") if minf == 0 else float("inf")

    global_position=tuple([0]*dim)
    gen=tuple([0]*dim)
    #gamma=1.0
    #beta=0.20
    #alpha=0.25
    gens_dict = {tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    #gens_dict[global_position]=0.001
    gens=random_search(n,dim)
    #vs = [[random.choice([0,1]) for i in range(length)] for i in range(N)]
    for gen in gens:
        if tuple(gen) in gens_dict:
            score = gens_dict[tuple(gen)]
        else:
            score=estimate(gen)
            gens_dict[tuple(gen)]=score
        if score > global_best:
            global_best=score
            global_position=dc(gen)
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)
    for it in miter:
        for i,x in enumerate(gens):
            for j,y in enumerate(gens):
                if gens_dict[tuple(y)] < gens_dict[tuple(x)]:
                    gens[j]=exchange_binary(y,gens_dict[tuple(y)])
                gen = gens[j]
                if tuple(gen) in gens_dict:
                    score = gens_dict[tuple(gen)]
                else:
                    score=estimate(gens[j])
                    gens_dict[tuple(gen)]=score
                if score > global_best if minf==0 else score < global_best:
                    global_best=score
                    global_position=dc(gen)
    return global_best,global_position,global_position.count(1)

"""BGSA"""
def Bmove(x,a,v):
    n,dim=len(x),len(x[0])#size(x)#次元がかえってくる20,13（群数,特徴次元）
    v=[[random.random()*v[j][i]+a[i] for i in range(dim)] for j in range(n)]#rand(n,nn).*v+a#要素ごとの乗算#randは次元数分のrand配列
    s=[[abs(math.tanh(_v)) for _v in vv ] for vv in v]
    temp=[[1 if rr<ss else 0 for rr,ss in zip(_r,_s)] for _r,_s in zip([[random.random() for i in range(dim)] for j in range(n)],s)]# < s:#s以上なら1,
    x_moving=[[0 if temp[ind][i]==1 else 1  for i in range(len(temp[ind])) ] for ind in range(len(temp))]#find(t==1)#1のインデックス番号求めてそれの逆~にする
    #xm(moving)=~xm(moving)
    return x_moving,v

def mc(fit,min_f):
    fmax=max(fit)
    fmin=min(fit)
    fmean=np.mean(fit)
    i,n=1,len(fit)

    if fmax==fmin:
        m=[1 for i in range(n)]#once(n,1)
    else:
        if min_f==1:
            best=fmin
            worst=fmax
        else:
            best=fmax
            worst=fmin
        m=[(f-worst)/(best-worst) for f in fit]
    mm=[_m/sum(m) for _m in m]
    return mm

def BGc(itertion,max_iter):
    g0=1
    g=g0*(1-(itertion/max_iter))
    return g

def BGf(m,x,G,Rp,EC,itertion,max_iter):
    n,dim=len(x),len(x[0])#size(x)#n=群数,dim=次元数
    final_per=2#In the last iteration, only 2 percent of agents apply force to the others
    if EC == 1:
        kbest=final_per+(1-itertion/max_iter)*(100-final_per)
        kbest=round(n*kbest/100)
    else:
        kbest=n
    mm=np.array(m)
    am=[np.argsort(mm)[::-1][i] for i in range(len(mm))]#:
    ds=sorted(am,reverse=True)#降順

    for i in range(n):
        E=[0 for i in range(dim)]#zero(1,dim)
        for ii in range(kbest):
            j=ds[ii]
            if j != i:
                R=sum([1 for xi,xj in zip(x[i],x[j]) if xi!=xj])#hammimng dist
                R=R/dim
                for k in range(dim):
                    E[k]=E[k]+random.random()* m[j] *( (x[j][k]-x[i][k]) / (R**Rp+1/dim) )
            else:
                pass
    a=[e*G for e in E]
    return a

def BGSA(Eval_Func,n=20,m_i=200,dim=None,minf=0,prog=False,EC=1,Rp=1,f_ind=25):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            EC: Elite Check, default=1
            Rp: Value between mass, default=1
            f_ind: Value of kbest, default=25
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)

    best_bin='0'*dim
    fbest=float("-inf") if minf == 0 else float("inf")
    best_val=float("-inf") if minf == 0 else float("inf")
    #EC=1
    #Rp=1
    #f_ind=25#24: max-ones, 25: royal-road(王道)
    #minf=minf#0#1:mini,0:maximization
    gens_dic={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    #flag=dr#False
    gens=random_search(n,dim)#[[random.choice([0,1]) for _ in range(dim)] for i in range(n)]
    bestc=[]
    meanc=[]
    v=[[0 for d in range(dim)] for i in range(n)]
    fit=[float("-inf") if minf == 0 else float("inf") for i in range(n)]
    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        for g_i in range(n):
            if  tuple(gens[g_i]) in gens_dic:
                fit[g_i]=gens_dic[tuple(gens[g_i])]
            else:
                fit[g_i]=estimate(gens[g_i])
                gens_dic[tuple(gens[g_i])]=fit[g_i]

        if it > 1:
            if minf==1:
                pass
                #afit=find(fitness>fitold)#minimazation#find is return index_list
                afit=[ind for ind in range(n) if fit[ind] > fitold[ind]]
            else:
                #afit=find(fittness<fitold)#max#
                afit=[ind for ind in range(n) if fit[ind] < fitold[ind]]

            if len(afit)!=0:
                for ind in afit:
                    gens[ind]=gensold[ind]
                    fit[ind]=fitold[ind]

        if minf == 1:
            best=min(fit)#min
            best_ind=fit.index(min(fit))
        else:
            best=max(fit)#max
            best_ind=fit.index(max(fit))
        if it==1:
            fbest=best
            lbest=gens[best_ind]

        if minf==1:
            if best<fbest:
                fbest=best
                lbest=gens[best_ind]
        else:
            if best>fbest:
                fbest=best
                lbest=gens[best_ind]

        bestc=fbest
        meanc=np.mean(fit)

        m=mc(fit,minf)
        g=BGc(it,m_i)
        a=BGf(m,gens,g,Rp,EC,it,m_i)

        gensold=dc(gens)
        fitold=dc(fit)

        gens,v=Bmove(gens,a,v)
    return fbest,lbest,lbest.count(1)

"""BBA"""
def BBA(Eval_Func,n=20,m_i=200,dim=None,minf=0,prog=False,qmin=0,qmax=2,loud_A=0.25,r=0.4):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            qmin: frequency minimum to step
            qmax: frequency maximum to step
            loud_A: value of Loudness, default=0.25
            r: Pulse rate, default=0.4, Probability to relocate near the best position
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    #flag=dr
    #qmin=0
    #qmax=2
    #loud_A=0.25
    #r=0.1
    #n_iter=0
    gens_dic={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}
    q=[0 for i in range(n)]
    v=[[0 for d in range(dim)] for i in range(n)]
    #cgc=[0 for i in range(max_iter)]
    fit=[float("-inf") if minf == 0 else float("inf") for i in range(n)]
    #dr=False
    gens=random_search(n,dim)#[[random.choice([0,1]) for d in range(dim)] for i in range(n)]

    for i in range(n):
        if  tuple(gens[i]) in gens_dic:
            fit[i]=gens_dic[tuple(gens[i])]
        else:
            fit[i]=estimate(gens[i])
            gens_dic[tuple(gens[i])]=fit[i]

    if minf==0:
        maxf=max(fit)
        best_v=maxf
        best_s=gens[fit.index(max(fit))]
    elif minf==1:
        minf=min(fit)
        best_v=minf
        best_s=gens[fit.index(min(fit))]


    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)

    for it in miter:
        #cgc[i]=maxf
        for i in range(n):
            for j in range(dim):
                q[i]=qmin+(qmin-qmax)*random.random()
                v[i][j]=v[i][j]+(gens[i][j]-best_s[j])*q[i]

                vstf=abs((2/math.pi)*math.atan((math.pi/2)*v[i][j]))

                if random.random()<vstf:
                    gens[i][j]= 0 if gens[i][j]==1 else 1
                else:
                    pass

                if random.random()>r:
                    gens[i][j]=best_s[j]

            if  tuple(gens[i]) in gens_dic:
                fnew=gens_dic[tuple(gens[i])]
            else:
                fnew=estimate(gens[i])
                gens_dic[tuple(gens[i])]=fnew

            if fnew >= fit[i] and random.random() < loud_A if minf==0 else fnew <= fit[i] and random.random() < loud_A:#max?
                gens[i]=gens[i]
                fit[i]=fnew

            if fnew>best_v if minf==0 else fnew<best_v:
                best_s=dc(gens[i])
                best_v=dc(fnew)

    return best_v,best_s,best_s.count(1)

"""BDFA"""
def BDFA(Eval_Func,n=20,m_i=200,dim=None,minf=0,prog=False):
    """
    input:{ Eval_Func: Evaluate_Function, type is class
            n: Number of population, default=20
            m_i: Number of max iteration, default=300
            minf: minimazation flag, default=0, 0=maximization, 1=minimazation
            dim: Number of feature, default=None
            prog: Do you want to use a progress bar?, default=False
            }

    output:{Best value: type float 0.967
            Best position: type list(int) [1,0,0,1,.....]
            Nunber of 1s in best position: type int [0,1,1,0,1] → 3
            }
    """
    estimate=Eval_Func().evaluate
    if dim==None:
        dim=Eval_Func().check_dimentions(dim)
    maxiter=m_i#500
    #flag=dr#True
    best_v=float("-inf") if minf == 0 else float("inf")
    best_p=[0]*dim
    gens_dict={tuple([0]*dim):float("-inf") if minf == 0 else float("inf")}

    enemy_fit=float("-inf") if minf == 0 else float("inf")
    enemy_pos=[0 for _ in range(dim)]
    food_fit=float("-inf") if minf == 0 else float("inf")
    food_pos=[0 for _ in range(dim)]

    fit=[0 for _ in range(n)]
    genes=random_search(n,dim)
    genesX=random_search(n,dim)

    if prog:
        miter=tqdm(range(m_i))
    else:
        miter=range(m_i)
    for it in miter:
        w=0.9 - it * ((0.9-0.4) / maxiter)
        mc=0.1- it * ((0.1-0) / (maxiter/2))
        if mc < 0:
            mc=0
        s=2 * random.random() * mc
        a=2 * random.random() * mc
        c=2 * random.random() * mc
        f=2 * random.random()
        e=mc
        if it > (3*maxiter/3):
            e=0

        for i in range(n):
            if tuple(genes[i]) in gens_dict:
                fit[i]=gens_dict[tuple(genes[i])]
            else:
                fit[i]=estimate(genes[i])
                gens_dict[tuple(genes[i])]=dc(fit[i])
            if fit[i] > food_fit if minf==0 else fit[i] < food_fit:
                food_fit=dc(fit[i])
                food_pos=dc(genes[i])

            if fit[i] > enemy_fit if minf==0 else fit[i] < enemy_fit:
                enemy_fit=dc(fit[i])
                enemy_pos=dc(genes[i])

        for i in range(n):
            ind=-1
            nn=-1
            ndx=[[0 for _d in range(dim)] for _ in range(n)]
            nx=[[0 for _d in range(dim)] for _ in range(n)]

            for j in range(n):
                if i==j:
                    pass
                else:
                    ind+=1
                    nn+=1
                    ndx[ind]=dc(genesX[j])
                    nx[ind]=dc(genes[j])

            S=[0 for _ in range(dim)]
            for k in range(nn):
                S=[_s+(_x-_y) for _s,(_x,_y) in zip(S,zip(ndx[k],genes[i]))] #s+(nx[k]-x[i])
            S=S

            A=[sum([_[_d] for _ in ndx])/nn for _d in range(dim)]#[sum(_)/nn if _ != 0 else 0 for _ in ndx]
            #[_-g for _,g in zip([sum([_[_d] for _ in nx])/nn for _d in range(dim)],genes[i])]
            C=[_-g for _,g in zip([sum([_[_d] for _ in nx])/nn for _d in range(dim)],genes[i])]#[sum(_)/nn-g if _ != 0 else 0 for _,g in zip(nx,genes[i])]

            F=[fp-g for fp,g in zip(food_pos,genes[i])]
            E=[ep+g for  ep,g in zip(enemy_pos,genes[i])]

            for j in range(dim):
                genesX[i][j]=s*S[j]+a*A[j]+c*C[j]+ f *F[j]+e*E[j]+w*genesX[i][j]

                if genesX[i][j] > 6:
                    genesX[i][j]=6
                if genesX[i][j] < -6:
                    genesX[i][j]=-6
                T = abs(genesX[i][j] / math.sqrt((1+genesX[i][j]**2)))
                if random.random()<T:
                    genes[i][j]=1 if genes[i][j] == 0 else 0
    best_p=food_pos
    best_v=food_fit

    return best_v,best_p,best_p.count(1)
