# referece: S.J.Chapman, D.P.Hewett and L.N.Trefethen,
#  "Mathematics of Faraday Cage" SIAM Review 57 (2015) 398

import numpy as np

class FaradayCage:
    """ shielding of electric fields """
    def __init__(self, c, r, zs):
        """
        c = center of wires (array of complex numbers)
        r = radius of wires (scalar real number)
        zs = external point charge (scalar complex number)
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
        zck = zc[...,np.newaxis]**(-k)
        zck = zck.reshape(len(z),-1)

        b = np.hstack([0, -np.log(np.abs(z-zs))])

        A = np.column_stack([
                np.hstack([0, -np.ones(len(z))]),
                np.vstack([np.ones(n), np.log(np.abs(zc))]),
                np.vstack([np.zeros(K), np.real(zck)]),
                np.vstack([np.zeros(K), np.imag(zck)])])

        x = np.linalg.lstsq(A,b)[0]
        e,x = x[0],x[1:]  # e = potential of wires
        d,x = x[:n],x[n:] # d = charge induced on wires
        a = x[:K].reshape(n,-1)
        b = x[K:].reshape(n,-1)

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.k = k
        self.r = r
        self.zs = zs

    def potential(self,z):
        """
        z = points at which potentials are evaluated
            as array of complex numbers
        """
        z = np.asarray(z)
        zc = z[...,np.newaxis] - self.c
        zck = zc[...,np.newaxis]**(-self.k)
        rc = np.abs(zc)

        u = np.log(np.abs(z - self.zs))
        u += np.dot(np.log(rc), self.d)
        u += np.einsum('...ij,ij', np.real(zck), self.a)
        u += np.einsum('...ij,ij', np.imag(zck), self.b)
        u[np.any(rc < self.r, -1)] = np.nan
        return u

    def field(self,z):
        """
        z = points at which electric fields are evaluated
            as array of complex numbers
        """
        z = np.asarray(z)
        zc = z[...,np.newaxis] - self.c
        zck = -self.k * zc[...,np.newaxis]**(-self.k-1)

        u = 1/(z - self.zs)
        u += np.dot(1/zc, self.d)
        u += np.einsum('...ij,ij', zck, self.a)
        u -= np.einsum('...ij,ij', zck, self.b)*1j
        return np.conj(u)
