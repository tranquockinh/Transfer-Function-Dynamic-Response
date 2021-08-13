import numpy as np
import matplotlib.pyplot as plt

# Import data
dataInput = open('earthquake.txt','r', encoding='utf-8-sig')
data = dataInput.read().split('\n')

# Initialize earthquake data
listData = []

# Read data
for dataPoints in data:
    if dataPoints == '': 
        continue
    
    elif np.array([dataPoints]).dtype != 'float' or 'int':
        listData.append(float(dataPoints))
eqData = np.zeros((len(listData),1))

for i in range(len(listData)):
    eqData[i] = listData[i]
eqData = (eqData-np.mean(eqData)) * 9.81

# Define legnth of signal series
for i in range(len(eqData)):

    if i**3 < len(eqData): continue
    else:
        N = (i**3) * 2
        break
print('The number of data points are: {}'.format(N))

Pad_Begin = np.zeros((100,1))
Pad_End = np.zeros((N-len(eqData)-100, 1))

if print(len(Pad_Begin)+len(Pad_End)+len(eqData) - N) == 0:
    print ('Data length is fine!')
else:
    print ('Plase check length of signal!')

Data = np.concatenate((Pad_Begin, eqData,Pad_End), axis=0)
# Data properties
## Pre-defined natural frequency of the system
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

TransDisp = np.divide(-1, (1j*w)**2 + 2*kesi*wn*(1j*w) + wn**2)
TransVelo = (1j*w) * TransDisp
TransAcce = ((1j*w)**2) * TransDisp

def fft(x):
    N = len(x)
    if N == 1:
        return x
    else:
        x_even = fft(x[::2])
        x_odd = fft(x[1::2])
        factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        x_transform = np.concatenate(\
            [x_even+factor[:int(N/2)]*x_odd,
             x_even+factor[int(N/2):]*x_odd])
        return x_transform

Z = fft(np.array(Data.T).flatten())
Z = Z[0:int(N/2)+1]

# Plotting displacement, velocity and acceleration
fig1, axes = plt.subplots(2,1)
axes[0].plot(t, Data)
axes[0].set_xlabel('time [second, s]')
axes[0].set_ylabel('Amplitude')
axes[1].plot(f_Nyquist, np.abs(Z))
axes[1].set_xlabel('Frequency [Hz]')
axes[1].set_ylabel('Power density [dB]')
fig1.suptitle('Time domain and frequency domain signal')

components = np.zeros((len(Z),3),dtype=complex)

for i in range(len(Z)):
    components[i,0] = TransDisp[i]
    components[i,1] = TransVelo[i]
    components[i,2] = TransAcce[i]

Disp = components[:,0] * Z
Velo = components[:,1] * Z
Acce = components[:,2] * Z


# Put full vector of transformed ones by 
#concatenating conjugate of up-down flip
xDispFull = np.concatenate(\
(Disp, np.flipud(Disp[1:-1]).conjugate()), axis=0)
xVeloFull = np.concatenate(\
(Velo, np.flipud(Velo[1:-1]).conjugate()), axis=0)
xAcceFull = np.concatenate(\
(Acce, np.flipud(Acce[1:-1]).conjugate()), axis=0)

disp = np.fft.ifft(xDispFull).real
v = np.fft.ifft(xVeloFull).real
a = np.fft.ifft(xAcceFull).real

if len(disp) == len(v) == len(a):
    outputs = np.zeros((int(len(disp)), 3), dtype=float)
    for i in range(int(len(disp))):
        outputs[:,0] = disp[i]
        outputs[:,1] = v[i]
        outputs[:,2] = a[i]

print(outputs)     

fig2, axes = plt.subplots(3,1)
axes[0].plot(t, disp)
axes[0].set_ylabel('Displacement')
axes[1].plot(t, v)
axes[1].set_ylabel('Velocity')
axes[2].plot(t,a)
axes[2].set_ylabel('Acceleration')    
axes[2].set_xlabel('Time [second, s]')    

# Consider to range of natural frequency Tn
def wn_range(Z, kesi, w):
    Tn = np.linspace(0.1,10,100)
    fn =1/Tn
    wn = np.zeros((len(fn),1))
    dmax = np.zeros((len(fn),1))
    vmax = np.zeros((len(fn),1))
    amax = np.zeros((len(fn),1))
    
    for i in fn:  
        j = np.array(np.where(fn==i)).flatten()[0]
        wn[j] = 2 * np.pi * fn[j]        
        ## Distribution of natural frequency in 
        ##Transfer function of displacement
        TransDisp = np.divide(-1, (1j*w)**2 + 2*kesi*wn[j]*(1j*w) + wn[j]**2)       
        TransVelo = (1j*w) * TransDisp
        TransAcce = ((1j*w)**2) * TransDisp
        
        components = np.zeros((len(Z),3),dtype=complex)

        for k in range(len(Z)):
            components[k,0] = TransDisp[k]
            components[k,1] = TransVelo[k]
            components[k,2] = TransAcce[k]

        Disp = components[:,0] * Z
        Velo = components[:,1] * Z
        Acce = components[:,2] * Z
        
        xDispFull = np.concatenate((Disp, np.flipud(Disp[1:-1]).\
        conjugate()), axis=0)
        xVeloFull = np.concatenate((Velo, np.flipud(Velo[1:-1]).\
        conjugate()), axis=0)
        xAcceFull = np.concatenate((Acce, np.flipud(Acce[1:-1]).\
        conjugate()), axis=0)

        disp = np.fft.ifft(xDispFull).real
        v = np.fft.ifft(xVeloFull).real
        a = np.fft.ifft(xAcceFull).real
        
        dmax[j] = np.max(disp)
        vmax[j] = np.max(v)
        amax[j] = np.max(a)
    return Tn, dmax, vmax, amax
    
Tn, dmax, vmax, amax = wn_range(Z, kesi, w)
fig3, axes = plt.subplots(3,1)
axes[0].plot(Tn, dmax)
axes[0].set_ylabel('Max. Displacement')
axes[1].plot(Tn, vmax)
axes[1].set_ylabel('Max. Velocity')
axes[2].plot(Tn, amax)    
axes[2].set_ylabel('Max. Acceleration')    
axes[2].set_xlabel('Period [second, s]')    
plt.show()