import numpy as np
import matplotlib.pyplot as plt
from FaradayCage import FaradayCage

r = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
n1 = 5
n2 = [16,92,256,256,256,256]
N = [12,16,20,20,20,20]
zs = 2

for i,r_ in enumerate(r):
    E = []
    n = np.round(np.geomspace(n1,n2[i],N[i]))
    for n_ in n:
        c = np.exp(2j*np.pi*np.arange(n_)/n_)
        f = FaradayCage(c,r_,zs)
        E.append(np.abs(f.field(0)))
    plt.loglog(n,E,label='$r=10^{%d}$'%int(np.log10(r_)))

plt.axis([5,256,3e-3,5e-1])
plt.legend()
plt.xlabel('$n=$ number of wires', fontsize=14)
plt.ylabel('$|E(0)|=$ field strength at $z=0$', fontsize=14)
plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
