# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 18:38:56 2025

@author: nachi
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
#%%  Valores de parámetros

Fo = 100#1.0
Fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
ts = 1/Fs # tiempo de muestreo
df = Fs/N # resolución espectral
Ac =1 #Amplitud 
DC = 0 #Valor Contínua
tita= 0 #Defasaje

#%% Funcion DFT 

def my_DFT(xx):
    ##Necesito el tamaño de la señal de entrada
    N = len(xx)
    n = np.arange(N)#Vector donde tengo de 0 hasta N
    k = n.reshape((N, 1))#Vector de k de N filas y 1 Columna
    e = np.exp(-2j * np.pi * k * n / N)#Matriz con los valores de las exponenciales
    X = np.dot(e, xx)#Multiplico La matriz de exponenciales por el vector de entrada y será la sumatoria de las exponenciales escaladas con la muestra
    return X

#%% Funcion generadora de senoidal
def my_sin_gen( vmax, dc, fo, ph, nn, fs): 
    tt = np.arange(0,nn*1/fs,1/fs) # grilla de sampleo temporal
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

S = my_DFT(s)

Sphase = np.angle(S)
Smod = np.abs(S)

#Todo esto es para Plotear el resultado de la DFT
TT = np.arange(len(S))
#plt.figure(2)
plt.figure(figsize = (8, 6))
plt.stem(TT*df, np.abs(S), 'b', \
markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()

TFF=np.fft.fft(s)

#plt.figure(3)
plt.figure(figsize = (8, 6))
plt.title('DFT')
plt.stem(tt*df, np.abs(TFF), 'b', \
markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.show()

# plt.figure(2)
# plt.title('Senoidal')
# plt.xlabel('tiempo [segundos]')
# plt.ylabel('Amplitud')
# plt.plot(TT, abs(S))
# plt.show()




