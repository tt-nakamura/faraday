import numpy as np
import matplotlib.pyplot as plt
from scipy.special import yvp
from HelmholtzCage import HelmholtzCage

r = 0.1 # radius of wires
n = [8,16] # number of wires
omega = np.linspace(2,6,256) # angular frequency
zs = 2 # external point charge
z = [-1e-8, 1e-8]
dz = np.diff(z)

# n==0 (no cage)
E = np.abs(yvp(0,omega*np.abs(zs))*omega)
plt.semilogy(omega,E,label='$n=0$')

for n in n:
    c = np.exp(2j*np.pi*np.arange(n)/n)
    E = []
    for om in omega:
        f = HelmholtzCage(c,r,zs,om)
        e = np.abs(np.diff(f.potential(z))/dz)[0]
        E.append(e)
        if e>4: print('resonance at', om, e)
    plt.semilogy(omega,E,label='$n=%d$'%n)

plt.axis([np.min(omega), np.max(omega), 2e-4, 1e1])
plt.legend()
plt.xlabel(r'$\omega=$ angular frequency', fontsize=14)
plt.ylabel('$|E(0)|=$ field strength at $z=0$', fontsize=14)
plt.tight_layout()
plt.savefig('fig7.eps')
plt.show()
