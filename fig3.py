import numpy as np
import matplotlib.pyplot as plt
from FaradayCage import FaradayCage

n = [10,20,40] # number of wires
r = 0.01 # radius of wires
zs = 2 # external point charge
N = 64 # number of plot points on a disk
d = np.exp(2j*np.pi*np.arange(N)/N)

x,y = np.meshgrid(np.linspace(-1.4, 2.2, 120),
                  np.linspace(-1.8, 1.8, 120))
z = x + 1j*y
levels = np.linspace(-2, 1.2, 33)

plt.figure(figsize=(6,2))

for i,n in enumerate(n):
    c = np.exp(2j*np.pi*np.arange(n)/n)
    f = FaradayCage(c,r,zs)
    u = f.potential(z)
    disks = c[:,np.newaxis] + r*d

    plt.subplot(1,3,i+1)
    plt.contour(x, y, u, levels=levels, linewidths=1)
    plt.fill(np.real(disks.T), np.imag(disks.T), color=(0.7, 0.7, 1))
    plt.plot(np.real(disks.T), np.imag(disks.T), 'b')
    plt.plot(np.real(zs), np.imag(zs), '.r')
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.xlim([np.min(x), np.max(x)])
    plt.ylim([np.min(y), np.max(y)])

plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
