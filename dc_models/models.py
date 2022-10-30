
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
    :param model            string options: 'lv','sir'         
    :param state_init:      initial options: x,y or s,i,r
    :param time_step:       step size
    :param tend:            stopping time
    :param N:               population size for sir
    :param ninter:          number of time intervals
    """

    self.__dict__.update((param, value) for param,value in kwargs.items()) 

    self._select_model()

    self.times = np.split(np.linspace(0.,self.tend, self.time_step),self.ninter)                                    
    self.states = list()
    self.parameters = list()


  def _select_model(self):
    if self.model=='sir':
      self.samples = 2
      self.system = self._sir
    elif self.model=='lv':
      self.samples = 4
      self.system = self._lv
    else:
      ValueError('Must select model-options: sir or lv')

  def _sir(self,states,time, beta,gamma):
    dx_1dt = -beta * states[0] * states[1] / self.N
    dx_2dt = beta * states[0] * \
              states[1] / self.N - gamma  * states[1]
    dx_3dt = gamma * states[1]
    diffs = np.array([dx_1dt, dx_2dt, dx_3dt])
    return diffs


  def _lv(self,states,time, alpha, beta, delta, gamma):
      dx_1dt = states[0] * (alpha - beta * states[1])
      dx_2dt = states[1] * (-delta + gamma * states[0])
      diffs= np.array([dx_1dt, dx_2dt])
      return diffs


  def _get_parameters(self,start=0,end=1):
    """
    Params:
    ---------------
    :param samples: number of paramters to return
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
    """
    return tuple([np.random.uniform(start,end) for _ in range(self.samples)])


  def _integrate(self, ode_system, x0,time,qoi):
    """
    """
    return integrate.odeint(ode_system, x0, time, args = qoi)

  def _run_model(self):
    """

    """
    if self.model =='sir':

        X0 = self.state_init

        for i in range(self.ninter):

          pargs = self._get_parameters()                 
          s,i,r = self._integrate(self.system, X0, self.times[i], pargs).T
          X0 = s[-1],i[-1],r[-1]
          self._save_output(np.column_stack([s,i,r]),pargs[0]/pargs[1])

    elif self.model == 'lv':

        X0 = list(self.state_init)      

        for i in range(self.ninter):

            pargs = self._get_parameters()
            res = self._integrate(self.system,X0,self.times[i],pargs)
            X0 = res[-1,:]
            self._save_output(res,pargs)
    else:
        raise ValueError('Must select model-options: sir or lv')

  def _save_output(self,state,parameter):
      self.states.append(state)
      self.parameters.append(parameter)

  def _get_outputs(self):
    return np.asarray(self.states), np.asarray(self.parameters), np.asarray(self.times)