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

def Hogdkin_Huxley(z, t, I=0):
    V, m, h, n = z
    # dVdt
    I_Na = g_Na * np.power(m,3) * h * (V - V_Na)
    I_K = g_K * np.power(n,4) * (V - V_K)
    # I_Na = g_Na * m**3 * h * (V - V_Na)
    # I_K = g_K * n**4 * (V - V_K)
    I_L = g_l * (V - V_l)
    dVdt = I - I_Na - I_K - I_L
    dVdt /= C
    # dmdt
    a, b = a_m(V), b_m(V)
    dmdt = (x_inf(a,b) - m) / tau(a,b)
    # dhdt
    a, b = a_h(V), b_h(V)
    dhdt = (x_inf(a,b) - h) / tau(a,b)
    # dndt
    a, b = a_n(V), b_n(V)
    dndt = (x_inf(a,b) - n) / tau(a,b)

    return [dVdt, dmdt, dhdt, dndt]


def primerIntento(n_iter = 2000):
    z_0 = [-70, 0.1, 0.6, 0.3]  # De donde saco esto Alvaro?
    
    t = np.linspace(0, 200, n_iter)
    V = np.zeros_like(t)
    m = np.zeros_like(t)
    h = np.zeros_like(t)
    n = np.zeros_like(t)
    V[0], m[0], h[0], n[0] = z_0
    
    I_ext = np.zeros_like(t)
    I_ext[n_iter // 4: 3 * n_iter // 4] = 10    # Cuando encajamos corriente

    for i in range(1, n_iter):
        t_span = [t[i - 1], t[i]]
        z = odeint(Hogdkin_Huxley, z_0, t_span, args=(I_ext[i], ))
        # hodgkin_huxley_m_inf,
        # hodgkin_huxley_m_inf_hn_cte,
        z_0 = z[1]
        V[i], m[i], h[i], n[i] = z_0

    I_Na = g_Na * m**3 * h * (V - V_Na)
    I_K = g_K * n**4 * (V - V_K)
    I_L = g_l * (V - V_l)

    plt.plot(t, V, label="Membrana")
    plt.ylabel("Tensión de membrana (mV)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(t, I_ext, label="Externa")
    plt.ylabel(r"Corriente externa (μA cm$^{-2}$)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(t, I_Na, label="Na")
    plt.plot(t, I_K, label="K")
    plt.plot(t, I_L, label="L")
    plt.ylabel(r"Corriente (μA cm$^{-2}$)")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(V, I_Na, label="Na")
    plt.plot(V, I_K, label="K")
    plt.plot(V, I_L, label="L")
    plt.ylabel(r"Corriente (μA cm$^{-2}$)")
    plt.xlabel("Tensión (mV)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(t, m, label="m")
    plt.plot(t, h, label="h")
    plt.plot(t, n, label="n")
    plt.ylabel("Variables de compuerta")
    plt.xlabel("Tiempo (ms)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()

    plt.plot(V, m, label="m")
    plt.plot(V, h, label="h")
    plt.plot(V, n, label="n")
    plt.ylabel("Variables de compuerta")
    plt.xlabel("Tensión (mV)")
    # sns.despine(trim=True)
    plt.legend(loc='best')
    plt.show()
    # plt.close()




if __name__ == "__main__":
    primerIntento()