#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-03-21
File: ej_01.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import seaborn as sns
sns.set()

# Constantes
V_Na = 50       # mV
V_K = -77       # mV
V_l = -54.4     # mV
g_Na = 120      # mS/cm²
g_K = 36        # mS/cm²
g_l = 0.3       # mS/cm²
C = 1           # Capacitancia de la membrana \mu F / cm²

def a_m(V):
    aux = 0.1*V + 4.0 
    return aux / (1 - np.exp(-aux) )

def b_m(V):
    return 4 * np.exp(-(V+65.0) / 18.0)

def a_h(V):
    return 0.07 * np.exp(-(V+65.0) / 20.0)

def b_h(V):
    return 1.0 / (1.0 + np.exp(-0.1*(V+35.0)) )

def a_n(V):
    aux = 0.1 * (V + 55)
    return 0.1 * aux / (1 - np.exp(-aux) )

def b_n(V):
    return 0.125 * np.exp(-0.0125*(V+65.0))

def tau(a, b):
    return 1.0/(a+b)

def x_inf(a, b):
    return a/(a+b)

def s_inf(V):
    return 0.5 * (1 + np.tanh(0.2*V))


def Hogdkin_Huxley(z, t, g_syn, V_pre, I=0, V_sync=0, tau=3, m_inf=False, hn_cte=False):
    V, m, h, n, s = z

    # dsdt
    dsdt = (s_inf(V_pre) -s) * tau

    # dmdt
    a, b = a_m(V), b_m(V)
    if m_inf or hn_cte:
        dmdt = 0    # Aca la derivada no importa, pero si pongo None da mal no se xq
        m = x_inf(a,b)
    else:
        dmdt = (x_inf(a,b) - m) / tau(a,b)
    
    # dhdt
    a, b = a_h(V), b_h(V)
    dhdt = (x_inf(a,b) - h) / tau(a,b)
    
    # dndt
    if hn_cte:
        dndt = -dhdt
        # n = 0.8 - h
    else:
        a, b = a_n(V), b_n(V)
        dndt = (x_inf(a,b) - n) / tau(a,b)
    
    # dVdt
    I_Na = g_Na * np.power(m,3) * h * (V - V_Na)
    I_K = g_K * np.power(n,4) * (V - V_K)
    I_L = g_l * (V - V_l)
    I_syn = -g_syn * s * (V - V_sync)
    dVdt = I + I_syn - I_Na - I_K - I_L
    dVdt /= C

    return [dVdt, dmdt, dhdt, dndt]