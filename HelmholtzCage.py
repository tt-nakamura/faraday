# referece: S.J.Chapman, D.P.Hewett and L.N.Trefethen,
#  "Mathematics of Faraday Cage" SIAM Review 57 (2015) 398

import numpy as np
from scipy.special import hankel1,h1vp

class HelmholtzCage:
    """ shielding of oscillating electric fields """
    def __init__(self, c, r, zs, omega):
        """
        c = center of wires (array of complex numbers)
        r = radius of wires (scalar real number)
        zs = external point charge (scalar complex number)
        omega = angular frequency of oscillation
        """
        c = np.asarray(c)
        n = len(c)
        N = max(0, int(np.round(4+0.5*np.log10(r))))
        M = 3*N+2
        K = n*N

        d = np.exp(2j*np.pi*np.arange(M)/M)

        z = c[:,np.newaxis] + r*d
        z = z.reshape(-1)
        k = np.arange(1,N+1)
        zc = z[:,np.newaxis] - c
        rc = np.abs(zc)
        ek = ((zc/rc)[...,np.newaxis]**k).reshape(len(z),-1)
        hk = hankel1(k, omega*rc[...,np.newaxis]).reshape(len(z),-1)

        b = np.hstack([0, 1j*hankel1(0, omega*np.abs(z-zs))])

        A = np.column_stack([
                np.hstack([0, -np.ones(len(z))]),
                np.vstack([np.ones(n), hankel1(0, omega*rc)]),
                np.vstack([np.zeros(K), hk*np.real(ek)]),
                np.vstack([np.zeros(K), hk*np.imag(ek)])])

        x = np.linalg.lstsq(A,b)[0]
        e,x = x[0],x[1:]
        d,x = x[:n],x[n:]
        a = x[:K].reshape(n,-1)
        b = x[K:].reshape(n,-1)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k
        self.r = r
        self.zs = zs
        self.om = omega

    def potential(self, z):
        """
        z = points at which potentials are evaluated
            as array of complex numbers
        """
        z = np.asarray(z)
        zc = z[...,np.newaxis] - self.c
        rc = np.abs(zc)
        ek = (zc/rc)[...,np.newaxis]**self.k
        hk = hankel1(self.k, self.om*rc[...,np.newaxis])

        u = -1j*hankel1(0, self.om*np.abs(z - self.zs))
        u += np.dot(hankel1(0, self.om*rc), self.d)
        u += np.einsum('...ij,ij', hk*np.real(ek), self.a)
        u += np.einsum('...ij,ij', hk*np.imag(ek), self.b)
        u[np.any(rc < self.r, -1)] = np.nan
        return np.real(u)
