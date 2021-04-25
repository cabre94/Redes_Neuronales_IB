#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-04-21
File: ej2_2daArquitectura.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: 
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

SAVE_PATH = os.path.join(os.getcwd(), "Figuras", "ej1_2daArquitectura")
os.makedirs(SAVE_PATH, exist_ok=True)

def tanh(x_):
    return np.tanh(x_)

def gradTanh(x_):
    return 1 - np.tanh(x_)**2

def MSE(score, y_true):
    return (score-y_true)**2

def gradMSE(score, y_true):
    return 2*(score-y_true)

def predict(score):
    """El score es continuo, pero la prediccion es 1 o -1.
    Ponemos +-0.9 como threshold ya que usamos tanh como activacion"""
    if score >= 0.9:
        return 1
    elif score <= -0.9:
        return -1
    else:
        return score
        
def accuracy(score, y_true):
    "Devuelve 100% o 0% segun si le pego o no a la salida correcta"
    y_predict = predict(score)
    return (y_predict == y_true)*100.0

