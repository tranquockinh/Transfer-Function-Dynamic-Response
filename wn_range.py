from Paras import *
    
def wn_range(Z, kesi, w, Tn, fn):

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
    return dmax, vmax, amax
