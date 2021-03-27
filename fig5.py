import numpy as np
import matplotlib.pyplot as plt
from FaradayCage import FaradayCage

r = np.geomspace(1e-1, 1e-6, 64)
n = [4,8,16,32]
zs = 2

for n_ in n:
    E = []
    for r_ in r:
        c = np.exp(2j*np.pi*np.arange(n_)/n_)
        f = FaradayCage(c,r_,zs)
        E.append(np.abs(f.field(0)))
    plt.loglog(r,E,label='$n=%d$'%n_)

plt.axis([min(r),max(r),3e-3,5e-1])
plt.legend()
plt.xlabel('$r=$ radius of wires', fontsize=14)
plt.ylabel('$|E(0)|=$ field strength at $z=0$', fontsize=14)
plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
