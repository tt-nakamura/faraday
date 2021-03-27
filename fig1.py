import numpy as np
import matplotlib.pyplot as plt

n = 12
M = 64

c = np.exp(2j*np.pi*np.arange(n)/n)
d = np.exp(2j*np.pi*np.arange(M+1)/M)

plt.plot(np.real(d), np.imag(d), 'k--', lw=1)
plt.plot(np.real(c), np.imag(c), 'ko', ms=20, mfc='w')
plt.text(0,0.15,'small',ha='center',fontsize=20)
plt.text(0,0,'electric',ha='center',fontsize=20)
plt.text(0,-0.15,'field',ha='center',fontsize=20)
plt.text(1.1, 0.85, 'external',ha='left',fontsize=20)
plt.text(1.1, 0.7, 'electric',ha='left',fontsize=20)
plt.text(1.1, 0.55, 'field',ha='left',fontsize=20)

plt.axis('equal')
plt.axis('off')
plt.axis([-1.1,1.5,-1.1,1.1])
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
