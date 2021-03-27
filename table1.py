import numpy as np
import matplotlib.pyplot as plt
from FaradayCage import FaradayCage

n = [4,8,16,32,64,128,256]
r = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
zs = 2

for n_ in n:
    print('$',n_,'$ ', end='')
    for r_ in r:
        if r_ >= np.sin(np.pi/n_):
            print('& $0$', end='')
            continue

        c = np.exp(2j*np.pi*np.arange(n_)/n_)
        f = FaradayCage(c,r_,zs)
        E = np.abs(f.field(0))
        print('& $','%.5f'%E,'$',end='')
    print('\\\\')
