import numpy as np
import pandas as pd
from scipy import integrate
from math import pi,atan
import ipywidgets as ipw
import math
import warnings
warnings.filterwarnings('ignore')


class model:
    def __init__(self,**kwargs):
        """
        Parameters
        _________________       
        :param state_init:      initial options: x,y or s,i,r or r,l,c
        :param time_step:       step size
        :param tend:            stopping time
        :param N:               population size for sir
        :param intervals:      number of time intervals
        """

        self.__dict__.update((param, value) for param,value in kwargs.items()) 
        self.times = np.split(np.linspace(0.,self.tend, self.time_step),self.intervals)                 
        self.states = list()
        self.parameters = list()
        
    def _integrate(self, ode_system, x0,time,poi):
        """
        """
        return integrate.odeint(ode_system, x0, time, args = poi)
    
    def _get_parameters(self,samples, start=0,end=1):
        """
        Params:
        ---------------
        :param samples: number of parameters to return
        :param start: lower bound of support for uniform dist
        :param end: upper bound of support for uniform dist

        Returns:
        ---------------
        :param alpha: average per capita birthrate of the prey
        :param delta: conversion factor
        :param gamma: fraction of prey caught per predator per unit time
        :param beta: average per capita birthrate of the predators
        OR
        :param beta: Contact rate
        :param gamma: Mean recovery rate
        OR
        :param r: resistance (ohms)
        :param c: conductance (farads)
        :param l: inductance  (henries)
        """
        
        return tuple([np.random.uniform(start,end) for _ in range(samples)])
    
    def _save_output(self,state,parameter):
        self.states.append(state)
        self.parameters.append(parameter)

    def _get_outputs(self):
        return np.asarray(self.states), np.asarray(self.parameters), np.asarray(self.times)



class sir(model):
    """
    """
    
    def _system(self,states,time,*p):
        """
        sir state space system     s'=-beta*s*i/N
                                   i'= beta*s*i/N - gamma*i
                                   r'= gamma*i
                                   
        Parameters:
        _____________________
        :param p[0]:= beta
        :param p[1]:= gamma
        """
        dx_1dt = -p[0] * states[0] * states[1] / self.N
        dx_2dt = p[0]* states[0] * \
                  states[1] / self.N - p[1]  * states[1]
        dx_3dt = p[1] * states[1]
        diffs = np.array([dx_1dt, dx_2dt, dx_3dt])
        return diffs
    
    def _run_model(self):
        """
        """
        X0 = self.state_init
        samples = 2
        
        for i in range(self.intervals):
            pargs = self._get_parameters(samples)                 
            s,i,r = self._integrate(self._system, X0, self.times[i], pargs).T
            X0 = s[-1],i[-1],r[-1]
            self._save_output(np.column_stack([s,i,r]),pargs[0]/pargs[1])

class lotka_volterra(model):
    """
    """
    
    def _system(self,states,time,*p):
        """
        lotka_volterra state-space system:   x'= x(a-by)
                                             y'= y(-d+gx)
        
        Parameters:
        _______________
        :param p[0]:= alpha
        :param p[1]:= beta
        :param p[2]:= delta
        :param p[3]:= gamma
        """
        dx_1dt = states[0] * (p[0] - p[1] * states[1])
        dx_2dt = states[1] * (-p[2] + p[3] * states[0])
        diffs = np.array([dx_1dt, dx_2dt])
        return diffs
    
    def _run_model(self):
        """
        """
        
        X0 = list(self.state_init)   
        samples = 4

        for i in range(self.intervals):

            pargs = self._get_parameters(samples)
            res = self._integrate(self._system,X0,self.times[i],pargs)
            X0 = res[-1,:]
            self._save_output(res,pargs)


class rlc(model):
    """
    
    """
    
    def _system(self, states,times,*p):
        """
        rlc state-space system: V0 = Lx" + Rx' +x/C.
        
        Parameters:
        _______________
        :param p[0]:= r
        :param p[1]:= l
        :param p[2]:= c
        """
        v0=5
        return [states[1],(v0/p[2])-(p[0]/p[1])*states[1]-states[0]/(p[1]*p[2])]
    
    def _run_model(self):
        """

        """
#         pargs = tuple([1e3,1e-9,20e-3]) # hard coded for initial testing
        pargs = tuple([.2,1,1])
        for i in range(self.intervals):

            X0 =  self.state_init
            samples = 3
#             pargs = self._get_parameters(samples)

            res = self._integrate(self._system,X0,self.times[i],pargs)
            X0 = res[-1,:]
            self._save_output(res,pargs)

        
    
    
    