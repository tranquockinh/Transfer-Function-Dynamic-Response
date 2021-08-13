from Paras import *
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