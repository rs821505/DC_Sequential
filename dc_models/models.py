import numpy as np
import pandas as pd
from scipy import integrate
from math import pi, atan
import ipywidgets as ipw
import math
import warnings

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, **kwargs):
        """Base model class for all ode system models

        Parameters
        ----------
        model : str
            string name of the desired ode model class
        number_forward_runs : int
            number of forward simulations to run to produce samples for  dc inversion
        number_observations: int
             output of our observation operator (or number of observations/sensors)
        initial_state : np.ndarray
            initial options: x,y or s,i,r or r,l,c
        true_model : bool
            true or predicted forward model choice
        total_timesteps : int
            total number of steps to propogate forward in time
        number_timesteps : int
            number of samples to take from true state trajectory
        number_samples : int
            number of samples to draw per parameter
        end_time : int
            end time of the simulation
        number_parameters : int
            number of parameters to estimate
        lambda_true : list
            true lambda parameter values to be estimated
        supports : list
            supports of the probability distributions for each lambda parameter
        drift_windows : int
            number of windows to split simulation into
        assimilation_windows : int
            number of assimilation windows for data assimilation

        """

        self.__dict__.update((param, value) for param, value in kwargs.items())

        self.times = np.split(
            np.linspace(0.0, self.end_time, self.total_timesteps), self.drift_windows
        )
        self.states = list()
        self.parameters = list()

    def integrate(self, ode_system, x0, time, poi):
        """integrate the given ode system forward in time"""
        return integrate.odeint(ode_system, x0, time, args=poi)

    def get_parameters(self):
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
        if self.true_model:
            return tuple(self.lambda_true)
        else:
            sampled_parameters = tuple(
                [
                    np.random.uniform(*self.supports[i])
                    for i in range(self.number_parameters)
                ]
            )
        return sampled_parameters

    def save_output(self, state, parameter):
        self.states.append(state)
        self.parameters.append(parameter)

    def get_outputs(self):
        return (
            np.asarray(self.states),
            np.asarray(self.parameters),
            np.asarray(self.times),
        )

    def generate_runs(self):

        for i in range(self.number_forward_runs):

            self.run_model()
            states, params, sim_times = self.get_outputs()

            if i:
                runs = np.append(
                    runs,
                    states[:, :: (states.shape[1] // self.number_observations), :],
                    axis=0,
                )
                lambdas[i, :, :] = params
            else:
                runs = states[:, :: (states.shape[1] // self.number_observations), :]
                lambdas = np.zeros(
                    [self.number_forward_runs, params.shape[0], params.shape[1]]
                )
                lambdas[0, :, :] = params

        return runs, lambdas


class SirModel(Model):
    """ """

    def system(self, states, time, *p):
        """
        sir state space system     s'=-beta*s*i/population_size
                                   i'= beta*s*i/population_size - gamma*i
                                   r'= gamma*i

        Parameters:
        _____________________
        :p[0], p[1]:= beta, gamma
        """
        dx_1dt = -p[0] * states[0] * states[1] / self.population_size
        dx_2dt = p[0] * states[0] * states[1] / self.population_size - p[1] * states[1]
        dx_3dt = p[1] * states[1]
        diffs = np.array([dx_1dt, dx_2dt, dx_3dt])
        return diffs

    def run_model(self):
        """ """
        # X0 = self.initial_state

        for i in range(self.drift_windows):
            pargs = self.get_parameters()
            s, i, r = self.integrate(
                self.system, self.initial_state, self.times[i], pargs
            ).T
            self.initial_state = s[-1], i[-1], r[-1]
            self.save_output(np.column_stack([s, i, r]), pargs[0] / pargs[1])


class LotkaVolterraModel(Model):
    def system(self, states, time, *p):
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

    def run_model(self):
        """ """

        X0 = list(self.initial_state)

        for i in range(self.drift_windows):

            pargs = self.get_parameters()
            res = self.integrate(self.system, X0, self.times[i], pargs)
            X0 = res[-1, :]
            self.save_output(res, pargs)


class RlcModel(Model):
    def system(self, states, times, *p):
        """
        rlc state-space system: V0 = Lx" + Rx' +x/C.

        Parameters:
        _______________
        :param p[0], p[1], p[2]:= r, l, c
        """
        v0 = 5
        return [
            states[1],
            (v0 / p[2]) - (p[0] / p[1]) * states[1] - states[0] / (p[1] * p[2]),
        ]

    def run_model(self):
        """ """
        #         pargs = tuple([.2,1,1])  # hard coded for initial testing
        for i in range(self.drift_windows):

            X0 = self.initial_state
            pargs = self.get_parameters()
            res = self.integrate(self.system, X0, self.times[i], pargs)
            X0 = res[-1, :]
            self.save_output(res, pargs)
