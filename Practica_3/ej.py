#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 23-03-21
File: ej.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import hist
from numpy.core.shape_base import block
import seaborn as sns
sns.set()

# Creamos carpeta para guardar los archivos
SAVE_PATH = os.path.join("Informe", "Figuras")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Abrimos los archivos
spikes = np.loadtxt("spikes.dat")
spikes = spikes[:,1:]   # Tiro la primer columna de todo, ¿porque? No se

# stimulus = {}
# stimulus['t'], stimulus['I'] = np.loadtxt("stimulus.dat", unpack=True)

# suma = spikes.sum(axis=0)

# plt.hist(spikes, bins=1000, range=(0,50))
# plt.show()

isis = np.array([])

isis_aux = []
t_spikes = []

t = np.arange(0.1, 1000, 0.1)

for i in range(len(spikes)):
    isis = np.concatenate((isis, np.diff(t[spikes[i] > 0])))
    isis_aux.append(np.diff(t[spikes[i] > 0]))
    t_spikes.append(t[spikes[i] > 0])


""" Para elegir los bines de un histograma
fig, ax = plt.subplots(2, 2)
ax = ax.flatten() # ax es 2d, asi lo pasamos a 1d

for i, bins in enumerate(['scott', 'freedman', 'knuth', 'blocks']):
    histograma(isis, bins=bins, ax=ax[i])
    ax[i].set_title(f'hist(t, bins="{bins}")',fontdict=dict(family='monospace'))
plt.tight_layout()
plt.show()
"""


############################
# 1
############################
cv2 = []
isis_mean = []

for isi in isis_aux:
    cv2.append( 2 * abs(isi[:-1] - isi[1:]) / (isi[:-1] + isi[1:]))
    isis_mean.append(isi.mean())

cv2 = np.concatenate(cv2)
isis_aux = np.array(isis_aux, dtype=object)
t_spikes = np.array(t_spikes, dtype=object)

# Histograma de tiempos entre spikes
fig = plt.figure()
hist(isis, bins='freedman')
plt.tight_layout()
plt.show()

# Promedio de isi en cada prueba
plt.figure()
plt.plot(isis_mean)
plt.ylabel("Intervalo inter-spikes promedio [ms]")
plt.xlabel("# de prueba")
plt.tight_layout()
plt.show()

plt.figure()
plt.eventplot(t_spikes)
plt.title("Rasterplot")
plt.xlabel("Tiempo [ms]")
plt.xlabel("Prueba")
plt.show()

CV = isis.std() / isis.mean()

print("Coeficiente de variacion = {}".format(CV))
print("Intervalo inter-spikes promedio = {} ms".format(isis.mean()))
print("Tasa de disparo promedio = {} Hz".format(1.0 / isis.mean() * 1e3))
print("Coeficiente de variacion2 = {}".format(cv2.mean()))


############################
# 2
############################
count = spikes.sum(axis=1)
print("Count ", count.shape)

plt.figure()
plt.plot(count)
plt.xlabel("# de prueba")
plt.ylabel("Numero de spikes")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.sort(count))
plt.xlabel("# de prueba ordenadas")
plt.ylabel("Numero de spikes")
plt.tight_layout()
plt.show()

count = spikes.sum(axis=1)

plt.figure()
plt.eventplot(t_spikes[np.argsort(count)])
plt.title("Rasterplot")
plt.xlabel("Tiempo [ms]")
plt.ylabel("# de prueba ordenadas")
plt.tight_layout()
plt.show()

plt.figure()
hist(count, bins='freedman', density=True)
plt.xlabel("Numero de spikes")
plt.ylabel("Numero de pruebas")
plt.tight_layout()
plt.show()

Fano = count.var() / count.mean()
print("Factor de Fano: {}".format(Fano))



############################
# 3
############################

# Nose que es esto
plt.figure()
hist(np.concatenate(t_spikes), bins='freedman')
plt.title("Histograma tiempos de disparo")
plt.ylabel("Numero de spikes")
plt.xlabel("Tiempo de disparo [ms]")
plt.show()

from scipy import stats

rate = np.mean(spikes, axis=0) * 1e4
print("Tasa de disparo proemdio {} Hz".format(rate.mean()))

# fig, ax = plt.subplots()
plt.figure()
plt.plot(t, rate, lw=1)
# ax.plot(t, rate, lw=1)
ax = plt.gca()
ax_in = ax.inset_axes([0.24, 0.77, 0.3, 0.2])
ax_in.plot(t[3000:3100], rate[3000:3100], lw=1)
ax.indicate_inset_zoom(ax_in, edgecolor='k', alpha=0.8)
ax_in.set_xticks([])
plt.text(0.39, 0.76, "10 ms", horizontalalignment="center", verticalalignment="top", transform=ax.transAxes)
plt.title("Tasa de disparo, bin = 0.1, ms = 1 idx")
plt.ylabel("Tasa de disparo [Hz]")
plt.xlabel("Tiempo de disparo [ms]")
plt.show()




############################
# 4
############################

kk = spikes.sum(axis=0)
total = kk.sum()

kk_t, stimulus = np.loadtxt("stimulus.dat", unpack=True)

stimulus = stimulus[1:-1]

filtro = []
tau_log = []

for tau in range(1001):

    suma = 0

    # Recorremos todos los spikes
    for i in range(len(kk)):
        t_previo = i - tau
        if(t_previo >= 0):
            suma += kk[i] * stimulus[t_previo]  # Cant spikes * estimulo hace un tiempo tau
    
    filtro.append(suma / total)
    tau_log.append(tau / 10)

filtro = np.array(filtro)
tau_log = np.array(tau_log)

plt.plot(tau_log, filtro)
plt.xlabel(r"$\tau$ [ms]")
plt.ylabel(r"Algo")
plt.show()
        










