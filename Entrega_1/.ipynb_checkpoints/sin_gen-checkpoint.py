# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:30:13 2025

@author: nachi
"""

import numpy as np
import matplotlib.pyplot as plt

#%%  Valores de parámetros
Fo = 1.0
Fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
ts = 1/Fs # tiempo de muestreo
df = Fs/N # resolución espectral
Ac =1 #Amplitud 
DC = 0 #Valor Contínua
tita= 0 #Defasaje
#%% Funcion
def my_sin_gen( vmax, dc, fo, ph, nn, fs): 
    tt = np.arange(0,(nn-1)*1/fs,1/fs) # grilla de sampleo temporal
    xx = vmax*np.sin(2*np.pi*fo*tt + ph) + dc #Senoidal
    return [tt, xx]
#%%

tt, s = my_sin_gen(vmax = Ac, dc = DC, fo = Fo, ph=tita, nn = N, fs = Fs )
#s = Ac*np.sin(2*np.pi*fo*tt + tita) + DC #Senoidal


plt.figure(1)
plt.title('Senoidal')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud')
plt.plot(tt, s)
plt.show()