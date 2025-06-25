# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 09:26:57 2025

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
Ac =2**0.5 #Amplitud 
DC = 0 #Valor Contínua
tita = 0 #Defasaje
SNR = 97 # SNR en dB
sigma = (10**(-SNR/10)) #Varianza
desvio = sigma**0.5 #Desvío Estándar
B = 16 #Número de Bits
fc = 0.8 #Factor de Carga para no tener saturación del ADC por ruido en el pico de la señal
Vref = Ac/fc #Tensión de referencia del ADC
#q = Vref/(2**(B-1)) 
#%% Funciones
def my_sin_gen( vmax, dc, fo, ph, nn, fs): 
    tt = np.arange(0,nn*1/fs,1/fs) # grilla de sampleo temporal
    xx = vmax*np.sin(2*np.pi*fo*tt + ph) + dc #Senoidal
    return [tt, xx]

def cuantificar(signal, bits, vref):
    q = vref/(2**(bits-1)) 
    #s1 = (fc*Vref/Ac)*signal
    sq = signal*(1/q)
    sq1 = np.round(sq)
    ss_cuantizada = sq1*q#*(1/(fc*Vref/Ac))
    return ss_cuantizada
    
#%%Obtengo señal con ruido y su versión cuantizada 
plt.close('all')

tt, s = my_sin_gen(vmax = Ac, dc = DC, fo = Fo, ph=tita, nn = N, fs = Fs )
#Genero señal aleatoria para el ruido
na=np.random.normal(0, desvio, len(tt))

ss = s + na

ssq = cuantificar(signal = ss,bits = B,vref = Vref)#4Bits
ssq2 = cuantificar(signal = ss,bits = B*2,vref = Vref)#8Bits

#Ploteo Señal Original y dos versiones Cuantizadas para 2 valores de Bits

plt.figure("Señal original vs Cuantizada",figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(tt, ss, label="Señal analógica")
plt.title("Señal original vs Cuantizada")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(tt, ssq, color = 'r', label=f"Señal cuantizada con B = {B}", marker=".")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(tt, ssq2, color = 'g',label=f"Señal cuantizada con B = {2*B}", marker=".")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()

q2 = Vref/(2**(B-1))/2 
Xq=ssq-ss#Estimación ruido de cuantización

plt.figure("Histograma")
plt.title(f"Histograma de la Diferencia entre señal Cuantizada y sin Cuantizar para {B} Bits")
plt.hist(ssq-ss)
plt.show()
#%%Espectro de señal cuantizada y de la estimación de ruido

TFF=np.fft.fft(ssq)/N#Espectro de la cuantizada con 8 bits
mod=10*np.log10(2*np.abs(TFF)**2)
TFF4=np.fft.fft(ss)/N#Espectro de la señal
mod4=10*np.log10(2*np.abs(TFF4)**2)


plt.figure("DFT Senoidal Cuantizada vs Sin Cuantizar")
plt.subplot(2, 1, 1)
plt.title("DFT Senoidal Cuantizada vs Sin Cuantizar")
plt.plot(tt*len(tt)*df, mod,color = 'r',label=f"Espectro Señal cuantizada con B = {B}")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)|")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(tt*len(tt)*df, mod4,color = 'g',label="Espectro Señal sin cuantizar")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)|")
plt.legend()
plt.tight_layout()
plt.show()


TFF2=np.fft.fft(Xq)/N#Espectro de la estimacion de ruido de cuantizacion
mod2=10*np.log10(2*np.abs(TFF2)**2)
TFF3=np.fft.fft(na)/N#Espectro del ruido
mod3=10*np.log10(2*np.abs(TFF3)**2)#10*np.log10(escala)

plt.figure("DFT Estimacion de ruido Xq-Xa vs Espectro de Ruido")
plt.subplot(2, 1, 1)
plt.title('DFT Estimacion de ruido Xq-Xa vs Espectro de Ruido')
plt.plot(tt*len(tt)*df, mod2,color = 'r',label="Estimacion de ruido Xq-Xa")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(tt*len(tt)*df, mod3,color = 'g',label="DFT de na")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')
plt.legend()
plt.tight_layout()
plt.show()

TFF5=np.fft.fft(ss+Xq)/N#Espectro de la estimacion de ruido de cuantizacion
mod5=10*np.log10(2*np.abs(TFF5)**2)

plt.figure("DFT Senoidal Cuantizada vs Sin Cuantizar + Ruido Cuant")
plt.subplot(2, 1, 1)
plt.title("DFT Senoidal Cuantizada vs Sin Cuantizar + Ruido Cuant")
plt.plot(tt*len(tt)*df, mod,color = 'r',label=f"Espectro Señal cuantizada con B ={B}")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)|")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(tt*len(tt)*df, mod5,color = 'g',label="Espectro Señal sin cuantizar + Ruido Cuant")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)|")
plt.legend()
plt.tight_layout()
plt.show()

TFF6=np.fft.fft(ssq)/N#Espectro de la cuantizada con 8 bits
mod6=10*np.log10(2*np.abs(TFF6)**2)

tt2=tt[0:N//2]
plt.figure("DFT Senoidal Cuantizada + Xq + Ruido")
plt.title("DFT Senoidal Cuantizada + Xq + Ruido")
plt.plot(tt2*len(tt2)*df*2, mod6[0:N//2],color = 'b',label=f"Espectro Señal cuantizada con B = {B}")
plt.plot(tt2*len(tt2)*df*2, mod2[0:N//2],color = 'r',label="Estimacion de ruido Xq - Xa")
plt.plot(tt2*len(tt2)*df*2, mod3[0:N//2],color = 'g',label=f'Ruido con $\\sigma^2$ = {sigma:3.3}')
plt.legend()
plt.show()




