import numpy as np
import matplotlib.pyplot as plt
from Paras import *
from fft import fft
from wn_range import wn_range

## Pre-defined parameters
wn = 2 * np.pi * 1
dt = 0.02 ## sampling interval
df = 1/((N-1)*dt)
## time vector
t = np.zeros((len(Data),1))
for i in range(len(t)):
    t[i,0] = i * dt
## Nyquist frequenct
half_t = N * dt / 2
haft_f = N * df / 2
f_Nyquist = np.zeros((int(len(t)/2+1), 1))
for j in range(len(f_Nyquist)):
    f_Nyquist[j,0] = j * df
w = 2 * np.pi * f_Nyquist
kesi = 0.02
# Transfer functions
TransDisp = np.divide(-1, (1j*w)**2 + 2*kesi*wn*(1j*w) + wn**2)
TransVelo = (1j*w) * TransDisp
TransAcce = ((1j*w)**2) * TransDisp
# Fourier transform
Z = fft(np.array(Data.T).flatten())
Z = Z[0:int(N/2)+1]
components = np.zeros((len(Z),3),dtype=complex)
for i in range(len(Z)):
    components[i,0] = TransDisp[i]
    components[i,1] = TransVelo[i]
    components[i,2] = TransAcce[i]
Disp = components[:,0] * Z
Velo = components[:,1] * Z
Acce = components[:,2] * Z
# Put full vector of transformed ones by concatenating conjugate of up-down flip
xDispFull = np.concatenate((Disp, np.flipud(Disp[1:-1]).conjugate()), axis=0)
xVeloFull = np.concatenate((Velo, np.flipud(Velo[1:-1]).conjugate()), axis=0)
xAcceFull = np.concatenate((Acce, np.flipud(Acce[1:-1]).conjugate()), axis=0)
disp = np.fft.ifft(xDispFull).real
v = np.fft.ifft(xVeloFull).real
a = np.fft.ifft(xAcceFull).real
if len(disp) == len(v) == len(a):
    outputs = np.zeros((int(len(disp)), 3), dtype=float)
    for i in range(int(len(disp))):
        outputs[:,0] = disp[i]
        outputs[:,1] = v[i]
        outputs[:,2] = a[i]   
# Try the transfer function with series of period Tn
Tn = np.linspace(0.1,10,100)
fn =1/Tn
dmax, vmax, amax = wn_range(Z, kesi, w, Tn, fn)


# Plotting displacement, velocity and acceleration
fig1, axes = plt.subplots(2,1)
axes[0].plot(t, Data)
axes[0].set_xlabel('time [second, s]')
axes[0].set_ylabel('Amplitude')
axes[1].plot(f_Nyquist, np.abs(Z))
axes[1].set_xlabel('Frequency [Hz]')
axes[1].set_ylabel('Power density [dB]')
fig1.suptitle('Time domain and frequency domain signal')

fig2, axes = plt.subplots(3,1)
axes[0].plot(t, disp)
axes[0].set_ylabel('Displacement')
axes[1].plot(t, v)
axes[1].set_ylabel('Velocity')
axes[2].plot(t,a)
axes[2].set_ylabel('Acceleration')    
axes[2].set_xlabel('Time [second, s]')    

fig3, axes = plt.subplots(3,1)
axes[0].plot(Tn, dmax)
axes[0].set_ylabel('Max. Displacement')
axes[1].plot(Tn, vmax)
axes[1].set_ylabel('Max. Velocity')
axes[2].plot(Tn, amax)    
axes[2].set_ylabel('Max. Acceleration')    
axes[2].set_xlabel('Period [second, s]')    
plt.show()

