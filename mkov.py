# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 07:30:08 2019

@author: Pingouin
"""
import numpy as np
import random as rd
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter

# Utilitary
# Norming vector
def vnorm(v):
    res = v/np.linalg.norm(v,ord=1)
    return res

# Generating 010 vector
def evec(n=4):
    res = np.zeros(10)
    if n > 9 : return res
    else :
        res[n] = 1
        return res

# Get log10 range
def lrange(n = 42):
    res = np.floor(np.log10(n))
    return res

def norm(v):
    res = np.linalg.norm(v,ord=1)
    if res != 0:
        return res
    else :
        return 1

# End utilitary
class Grille:
    
    def __init__(self,lx=10,ly=20,zmin=0,zmax=42,dx=1,dy=1):
        self.lx = lx
        self.ly= ly
        self.sp = lx * ly
        self.zmin = zmin
        self.zmax = zmax
        self.dx=1
        self.dy=1
        (self.xval,self.yval) = np.divmod(np.arange(lx*ly),lx)
        self.zval = np.random.rand(lx*ly)
        self.ebord = np.array(0)
        self.ibord = np.array(0)
        
    def bord(self):
        res = np.array(0)
        res = np.append(res,np.arange(0,self.lx))
        res = np.append(res,np.arange(self.sp-self.lx,self.sp))
        res = np.append(res,np.arange(0,self.ly)*self.lx)
        res = np.append(res,(np.arange(0,self.ly)*(self.lx) + self.lx-1))
        self.ebord = np.unique(res)        
        
    def m_zval(self,zval):
        self.zval= zval
        self.zmin = np.min(self.zval)
        self.zmax = np.max(self.zval)
    
    def l_zval(self):
        return self.zval

    def c_elev(self):
        res = np.array(0)
        for (x,y) in zip(self.xval,self.yval):
            res = np.append(res,x*x +y*x * np.random.rand())
        self.m_zval(res)
        
    
        
class Mkov:
    def __init__(self,g):
        g.bord() # Compute border indexes
        self.bval = np.mean(g.zval[g.ibord])
        self.mval = np.mean(g.zval)
        self.zrng = lrange(g.zmax)
        self.lna = list()
        for i in np.arange(self.zrng +1):
            self.lna.append(self.mat_t(np.mod((np.round(g.zval/10**i)),10).astype(int)))           
        
    # Naive approach
    def mat_t(self,z):
        res = np.zeros([10,10])
        for (x,y), c in Counter(zip(z,z[1:])).items():            
            res[x,y] = c
        s = np.sum(res)
        return res/s
    
    def pred(self,v=142):
        res = 0
        for i in np.arange(self.zrng+1):
            i = int(i)            
            n = np.mod(np.round(v/100),10).astype(int)
            n = evec(n)            
            n = np.dot(self.lna[i],n)/norm(np.dot(self.lna[i],n))
            if np.sum(n) != 1:
                n = [1,0,0,0,0,0,0,0,0,0]
            n = np.random.choice(10,1,p=n).astype(int) * 10**i            
            res += int(n)        
        return(res)
        
    def gen_g(self,g):
        res = np.zeros(g.sp)
        res[g.ebord] = g.zval[g.ebord]
        for y in np.arange(1,g.ly-1):
            for x in np.arange(1,g.lx-1):
                i = y*g.lx + x
                res[i] = self.pred(res[i-1])                
        return res
    
    def plo_d(self,g,h):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_trisurf(g.xval,g.yval,g.zval, antialiased=True)
        ax.plot_trisurf(g.xval,g.yval,h, antialiased=True,alpha=0.3)
        fig.show()
            
g = Grille()
g.c_elev()
m = Mkov(g)
h = m.gen_g(g)
m.plo_d(g,h)