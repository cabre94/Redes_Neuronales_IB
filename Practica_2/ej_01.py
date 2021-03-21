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

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
import seaborn as sns
sns.set()

SAVE_PATH = os.path.join("Figuras", "ej_01")
# SAVE_PATH = os.path.join("Figuras", "Ej_06")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

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


def Hogdkin_Huxley(z, t, g_syn, V_pre, I_ext=0, V_syn=0, tauS=3, m_inf=False, hn_cte=False):
    V, m, h, n, s = z

    # dsdt
    dsdt = (s_inf(V_pre) - s) / tauS

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
    I_syn = -g_syn * s * (V - V_syn)
    # print(I_syn)

    dVdt = I_ext + I_syn - I_Na - I_K - I_L
    dVdt /= C

    return [dVdt, dmdt, dhdt, dndt, dsdt]

def simulacion(n_iter=2000, V_syn=0, I_ext=15e-3, tauS=3):

    g_syn_list = np.linspace(0, 2, 51)
    # g_syn_list = np.linspace(0, 2, 101)

    for g_syn in g_syn_list:

        # z_0_1 = [-20, 0.1, 0.5, 0.3, 0.1]
        # z_0_2 = [-30, 0.1, 0.5, 0.3, 0.1]

        # z_0_1 = [-20, 0.1, 0.5, 0.3, 0.1] 50 cprriente
        # z_0_2 = [-30, 0.1, 0.5, 0.3, 0.1]

        z_0_1 = [-20,  0.1, 0.7, 0.1, 1]        # Esto con 10 da lindo, hay que ver con V!=0
        z_0_2 = [-30, 0.1, 0.3, 0.3, 1]
        
        # z_0_1 = [-30,  0.1, 0.4, 0.3, 0.1]
        # z_0_2 = [-80, 0.1, 0.4, 0.1, 0.1]

        print("g_syn: {}".format(g_syn))

        n_iter = 10000
        t = np.linspace(0, 1000, n_iter)
        # t = np.linspace(0, 200, n_iter)
        
        V_1, V_2 = np.zeros_like(t), np.zeros_like(t)
        m_1, m_2 = np.zeros_like(t), np.zeros_like(t)
        h_1, h_2 = np.zeros_like(t), np.zeros_like(t)
        n_1, n_2 = np.zeros_like(t), np.zeros_like(t)
        s_1, s_2 = np.zeros_like(t), np.zeros_like(t)
        
        V_1[0], m_1[0], h_1[0], n_1[0], s_1[0] = z_0_1
        V_2[0], m_2[0], h_2[0], n_2[0], s_2[0] = z_0_2

        # np.full(n_iter, I_ext)

        for i in range(1,n_iter):
            t_span = [t[i - 1], t[i]]

            # z_1 = odeint(Hogdkin_Huxley, z_0_1, t_span, args=(g_syn, V_2[-1], I_ext, V_syn, tauS,))
            # z_2 = odeint(Hogdkin_Huxley, z_0_2, t_span, args=(g_syn, V_1[-1], I_ext, V_syn, tauS,))
            z_1 = odeint(Hogdkin_Huxley, z_0_1, t_span, args=(g_syn, z_0_2[0], I_ext, V_syn, tauS,))
            z_2 = odeint(Hogdkin_Huxley, z_0_2, t_span, args=(g_syn, z_0_1[0], I_ext, V_syn, tauS,))
            
            z_0_1 = z_1[1]
            z_0_2 = z_2[1]

            V_1[i], m_1[i], h_1[i], n_1[i], s_1[i] = z_0_1
            V_2[i], m_2[i], h_2[i], n_2[i], s_2[i] = z_0_2
        
        plotear(t, V_1, V_2, g_syn)

def plotear(t, V_1, V_2, g_syn):
    plt.plot(t, V_1, label="Neurona 1")
    plt.plot(t, V_2, label="Neurona 2")
    plt.ylabel("Voltaje (mV)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    file_name = os.path.join(SAVE_PATH, "{:.2f}.pdf".format(g_syn))
    plt.savefig(file_name, format='pdf')
    plt.tight_layout()
    # plt.show()
    plt.close()

if __name__ == "__main__":
    # simulacion()
    # simulacion(I_ext=15e-2)
    # simulacion(I_ext=10)
    simulacion(I_ext=10, V_syn=-80)
    # simulacion(V_syn=-80)