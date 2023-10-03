#!/usr/bin/env python3
#encoding: utf-8

import math

class MackeyGlass:
    def __init__(self,  n, d=1, Beta=0.2, Gamma=0.1, tau=17, init=[]):
        self.beta = Beta
        self.gamma = Gamma
        self.tau = tau
        self.d = d
        if len (init) <= self.tau:
            self.buff = [0.1]*(tau+1)
        else:
            self.buff = init
        self.n = n
        self.idxt = self.d
        self.idx = 0 
    def __iter__(self):
        self.idxt = self.d
        self.idx = 0
        return self
    def __next__(self):
        xt = self.buff[self.idxt]
        x  = self.buff[self.idx-1]
        x +=  self.beta * (xt / (1 + pow(xt, self.n)))-self.gamma * x
        self.buff[self.idx] = x
        self.idx = (self.idx+1) % len(self.buff)
        self.idxt = (self.idxt+1) % len(self.buff)
        return x


if __name__ == '__main__':
    mg = MackeyGlass(10, d=1, init=[0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759])
    itMG = iter(mg)
    for i in range(10000):
        x = next(itMG) 
        print (i, x)
