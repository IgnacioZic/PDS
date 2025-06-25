#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:55:30 2023

@author: mariano
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

#%%
Fo = 1.0
Fs = 1000.0 # frecuencia de muestreo (Hz)
N = 1000   # cantidad de muestras
ts = 1/Fs # tiempo de muestreo
df = Fs/N # resolución espectral
Ac =2**0.5 #Amplitud 
DC = 0 #Valor Contínua
tita = 0 #Defasaje
SNR = 3 # SNR en dB
sigma = (10**(-SNR/10)) #Varianza
desvio = sigma**0.5 #Desvío Estándar
Omega_0 = Fs/4
R = 200
tope_ecg = 0.985
tope_ppg = 0.95
tope_wav = 0.9999
tope_my_ecg = 0.985
#%%
plt.close('all')
##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz
##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
# mat_struct = sio.loadmat('./ECG_TP4.mat')

# ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
# N = len(ecg_one_lead)

# hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
# hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000])

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')
N_ecg = len(ecg_one_lead)
L_ecg = N_ecg//10
df_ecg = fs_ecg/N_ecg
plt.figure("ECG Sin Ruido")
plt.plot(ecg_one_lead)
#mat_struct = sio.loadmat('ECG_TP4.mat')
#ecg_real = vertical_flaten(mat_struct['ecg_lead'])
#plt.figure("ECG Con Ruido")
#plt.plot(ecg_real)

ff_ecg = np.arange(0,fs_ecg,df_ecg) # grilla de frecuencia
bfrec_ecg = ff_ecg <= fs_ecg/2


#Periodograma
f_ecg_1, PSD_ECG_1 = sig.periodogram(ecg_one_lead, fs=fs_ecg, window='flattop', nfft=N_ecg, detrend='constant', return_onesided=True, scaling='density', axis=-1)
PSD_ECG_1_NORM = PSD_ECG_1/np.max(PSD_ECG_1)#Normalizo
PSD_ECG_LOG_1=10*np.log10(2*np.abs(PSD_ECG_1)**2)
#Welch
f_ecg_2, PSD_ECG_2 = sig.welch(ecg_one_lead, fs=fs_ecg, window='flattop', nperseg=L_ecg, noverlap=None, nfft=N_ecg, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
PSD_ECG_2_NORM = PSD_ECG_2/np.max(PSD_ECG_2)#Normalizo
PSD_ECG_LOG_2=10*np.log10(2*np.abs(PSD_ECG_2)**2)
#Ploteo
plt.figure("PSD de ECG Sin Ruido")
plt.title('PSD del ECG')
plt.plot(ff_ecg[bfrec_ecg],PSD_ECG_LOG_1,color = 'r',label="PSD Periodograma",alpha=0.8)
plt.plot(ff_ecg[bfrec_ecg],PSD_ECG_LOG_2,color = 'g',label="PSD Welch")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)| dB")
plt.legend()
plt.show()

#Obtener la Estimación del BW
aprox_BW = np.cumsum(PSD_ECG_1_NORM)
aprox_BW_ecg = aprox_BW/np.max(aprox_BW)
limite_BW = np.max(aprox_BW_ecg[aprox_BW_ecg < tope_ecg])
estimacion_BW_ecg_1 = ff_ecg[np.where(aprox_BW_ecg == limite_BW)[0]]#En Hz

aprox_BW = np.cumsum(PSD_ECG_2_NORM)
aprox_BW_ecg = aprox_BW/np.max(aprox_BW)
limite_BW = np.max(aprox_BW_ecg[aprox_BW_ecg < tope_ecg])
estimacion_BW_ecg_2 = ff_ecg[np.where(aprox_BW_ecg == limite_BW)[0]]#En Hz
                                                    
# plt.figure("Estimacion de Ancho de Banda")
# plt.title('Estimo Ancho de banda')
# plt.plot(ff_ecg[bfrec_ecg],aprox_BW_ecg,color = 'orange',label="PSD Periodograma")
# plt.xlabel("Freq (Hz)")
# plt.ylabel("DFT Amplitude |X(freq)| dB")
# plt.legend()
# plt.show()

#%%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

##################
## PPG con ruido
##################

# # Cargar el archivo CSV como un array de NumPy
# ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


##################
## PPG sin ruido
##################

ppg = np.load('ppg_sin_ruido.npy')
N_ppg = len(ppg)
L_ppg = N_ppg//10
df_ppg = fs_ppg/N_ppg


ff_ppg = np.arange(0,fs_ppg,df_ppg) # grilla de frecuencia
bfrec_ppg = ff_ppg <= fs_ppg/2
#Periodograma
f_ppg_1, PSD_PPG_1 = sig.periodogram(ppg, fs=fs_ppg, window='flattop', nfft=N_ppg, detrend='constant', return_onesided=True, scaling='density', axis=-1)
PSD_PPG_1_NORM = PSD_PPG_1/np.max(PSD_PPG_1)#Normalizo
PSD_PPG_LOG_1=10*np.log10(2*np.abs(PSD_PPG_1)**2)
#Welch
f_ppg_2, PSD_PPG_2 = sig.welch(ppg, fs=fs_ppg, window='flattop', nperseg=L_ppg, noverlap=None, nfft=N_ppg, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
PSD_PPG_2_NORM = PSD_PPG_2/np.max(PSD_PPG_2)#Normalizo
PSD_PPG_LOG_2=10*np.log10(2*np.abs(PSD_PPG_2)**2)

plt.figure("PPG")
plt.plot(ppg)

plt.figure("PSD de PPG Sin Ruido")
plt.title('PSD del PPG')
plt.plot(ff_ppg[bfrec_ppg],PSD_PPG_LOG_1,color = 'r',label="PSD Periodograma")
plt.plot(ff_ppg[bfrec_ppg],PSD_PPG_LOG_2,color = 'g',label="PSD Welch")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)| dB")
plt.legend()
plt.show()

#Obtener la Estimación del BW
aprox_BW_1 = np.cumsum(PSD_PPG_1_NORM)
aprox_BW_ppg = aprox_BW_1/np.max(aprox_BW_1)
limite_BW_1 = np.max(aprox_BW_ppg[aprox_BW_ppg < tope_ppg])
estimacion_BW_ppg_1 = ff_ppg[np.where(aprox_BW_ppg == limite_BW_1)[0]]#En Hz

aprox_BW_1 = np.cumsum(PSD_PPG_2_NORM)
aprox_BW_ppg = aprox_BW_1/np.max(aprox_BW_1)
limite_BW_1 = np.max(aprox_BW_ppg[aprox_BW_ppg < tope_ppg])
estimacion_BW_ppg_2 = ff_ppg[np.where(aprox_BW_ppg == limite_BW_1)[0]]#En Hz

# plt.figure("Estimacion de Ancho de Banda")
# plt.title('Estimo Ancho de banda')
# plt.plot(ff_ppg[bfrec_ppg],aprox_BW_ppg,color = 'orange',label="PSD Periodograma")
# plt.xlabel("Freq (Hz)")
# plt.ylabel("DFT Amplitude |X(freq)| dB")
# plt.legend()
# plt.show()

#%%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

N_wav = len(wav_data)
L_wav = N_wav//10
df_wav = fs_audio/N_wav

ff_wav = np.arange(0,fs_audio,df_wav) # grilla de frecuencia
bfrec_wav = ff_wav <= fs_audio/2

#Periodograma
f_wav_1, PSD_WAV_1 = sig.periodogram(wav_data, fs=fs_audio, window='flattop', nfft=N_wav, detrend='constant', return_onesided=True, scaling='density', axis=-1)
PSD_WAV_1_NORM = PSD_WAV_1/np.max(PSD_WAV_1)#Normalizo
PSD_WAV_LOG_1=10*np.log10(2*np.abs(PSD_WAV_1)**2)
#Welch
f_wav_2, PSD_WAV_2 = sig.welch(wav_data, fs=fs_audio, window='flattop', nperseg=L_wav, noverlap=None, nfft=N_wav, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
PSD_WAV_2_NORM = PSD_WAV_2/np.max(PSD_WAV_2)#Normalizo
PSD_WAV_LOG_2=10*np.log10(2*np.abs(PSD_WAV_2)**2)

plt.figure("La Cucaracha")
plt.plot(wav_data)

plt.figure("PSD de WAV Sin Ruido")
plt.title('PSD de WAV')
plt.plot(ff_wav[bfrec_wav],PSD_WAV_LOG_1,color = 'r',label="PSD Periodograma")
plt.plot(ff_wav[bfrec_wav],PSD_WAV_LOG_2,color = 'g',label="PSD Welch")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)| dB")
plt.legend()
plt.show()

#Obtener la Estimación del BW
aprox_BW_2 = np.cumsum(PSD_WAV_1_NORM)
aprox_BW_wav = aprox_BW_2/np.max(aprox_BW_2)
limite_BW_2 = np.max(aprox_BW_wav[aprox_BW_wav < tope_wav])
estimacion_BW_wav_1 = ff_wav[np.where(aprox_BW_wav == limite_BW_2)[0]]#En Hz

aprox_BW_2 = np.cumsum(PSD_WAV_2_NORM)
aprox_BW_wav = aprox_BW_2/np.max(aprox_BW_2)
limite_BW_2 = np.max(aprox_BW_wav[aprox_BW_wav < tope_wav])
estimacion_BW_wav_2 = ff_wav[np.where(aprox_BW_wav == limite_BW_2)[0]]#En Hz


# plt.figure("Estimacion de Ancho de Banda")
# plt.title('Estimo Ancho de banda')
# plt.plot(ff_wav[bfrec_wav],aprox_BW_wav,color = 'orange',label="PSD Periodograma")
# plt.xlabel("Freq (Hz)")
# plt.ylabel("DFT Amplitude |X(freq)| dB")
# plt.legend()
# plt.show()


# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

#%%

fs = 360 #Hz

# ecg_sano = np.load('101m (2).mat')

# plt.figure("ECG SANO MIO")
# plt.title('ECG persona sana MIT')
# plt.plot(ecg_sano)
mat_struct = sio.loadmat('101m (2).mat')
ecg_sano = mat_struct['val']
ecg_sano = ecg_sano.flatten()
N = len(ecg_sano)
L = N//10
df = fs/N
plt.figure("ECG Propio")
plt.plot(ecg_sano)

ff = np.arange(0,fs,df) # grilla de frecuencia
bfrec = ff <= fs/2

f_1, PSD_MY_ECG_1 = sig.periodogram(ecg_sano, fs=fs, window='flattop', nfft=N, detrend='constant', return_onesided=True, scaling='density', axis=-1)
PSD_MY_ECG_NORM_1 = PSD_MY_ECG_1/np.max(PSD_MY_ECG_1)#Normalizo
PSD_MY_ECG_LOG_1=10*np.log10(2*np.abs(PSD_MY_ECG_1)**2)

f_2, PSD_MY_ECG_2 = sig.welch(ecg_sano, fs=fs, window='flattop', nperseg=L, noverlap=None, nfft=N, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
PSD_MY_ECG_NORM_2 = PSD_MY_ECG_2/np.max(PSD_MY_ECG_2)#Normalizo
PSD_MY_ECG_LOG_2=10*np.log10(2*np.abs(PSD_MY_ECG_2)**2)

plt.figure("PSD Mi ECGo")
plt.title('PSD de Mi ECG')
plt.plot(ff[bfrec],PSD_MY_ECG_LOG_1,color = 'r',label="PSD Periodograma")
plt.plot(ff[bfrec],PSD_MY_ECG_LOG_2,color = 'g',label="PSD Welch")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)| dB")
plt.legend()
# plt.show()

#Obtener la Estimación del BW
aprox_BW_3 = np.cumsum(PSD_MY_ECG_NORM_1)
aprox_BW_my_ecg = aprox_BW_3/np.max(aprox_BW_3)
limite_BW_3 = np.max(aprox_BW_my_ecg[aprox_BW_my_ecg < tope_my_ecg])
estimacion_BW_my_ecg_1 = f_1[np.where(aprox_BW_my_ecg == limite_BW_3)[0]]#En Hz


plt.figure("Estimacion de Ancho de Banda")
plt.title('Estimo Ancho de banda')
plt.plot(ff[bfrec],aprox_BW_my_ecg,color = 'orange',label="PSD Periodograma")
plt.xlabel("Freq (Hz)")
plt.ylabel("DFT Amplitude |X(freq)| dB")


aprox_BW_3 = np.cumsum(PSD_MY_ECG_NORM_2)
aprox_BW_my_ecg = aprox_BW_3/np.max(aprox_BW_3)
limite_BW_3 = np.max(aprox_BW_my_ecg[aprox_BW_my_ecg < tope_my_ecg])
estimacion_BW_my_ecg_2 = f_1[np.where(aprox_BW_my_ecg == limite_BW_3)[0]]#En Hz

plt.plot(ff[bfrec],aprox_BW_my_ecg,color = 'red',label="PSD Periodograma")                                               
plt.legend()
plt.show()