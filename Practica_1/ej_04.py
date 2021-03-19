#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-03-21
File: ej_04.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constantes
V_Na = 50       # mV
V_K = -77       # mV
V_l = -54.4     # mV
g_Na = 120      # mS/cm²
g_K = 36        # mS/cm²
g_L = 0.3       # mS/cm²
C = 1           # Capacitancia de la membrana \mu F / cm²

def a_m(V):
    aux = 0.1*V + 4.0 
    return aux / (1 - np.exp(-aux) )

def b_m(V):
    return 4 * np.exp(-(V+65.0) / 18.0)

def a_h(V):
    return 0.07 * np.exp(-(V+65.0) / 20.0)

def b_h(V):
    return 1.0 * (1.0 + np.exp(-0.1*(V+35.0)) )

def a_n(V):
    aux = 0.1 * (V + 55)
    return 0.1 * aux / (1 - np.exp(-aux) )

def b_n(V):
    return 0.125 * np.exp(-0.125*(V+65.0))

def tau(a, b):
    return 1.0/(a+b)

def x_inf(a, b):
    return a/(a+b)

def Hogdkin_Huxley(z, t, I=0):
    V, m, h, n = z
    # dVdt
    
    # dmdt
    a, b = a_m(V), b_m(V)
    dmdt = (x_inf(a,b) - m) / tau(a,b)
    # dhdt
    a, b = a_h(V), b_h(V)
    dmdt = (x_inf(a,b) - h) / tau(a,b)
    # dndt
    a, b = a_n(V), b_n(V)
    dmdt = (x_inf(a,b) - n) / tau(a,b)








