from numpy import outer, dot, zeros, empty
from numpy.random import multivariate_normal as mvn

def wishart(sig, n):
    """
    Sample from a Wishart distribution (naive implementation).
    
    Parameters
    ----------
    sig : (d,d) ndarray
        The scale matrix for the Wishart distribution, must be pos. definite.
    n : int
        Must be >= d.
        
    Returns
    -------
    w : (d,d) ndarray
        The sample from the distribution.
    """
    d = sig.shape[0]
    s = mvn(zeros(d), sig, n)
    w = empty((d,d))
    for i in xrange(n):
        w += outer(s[i], s[i])
    return w
        
def inv3(A):
    """Return the inverse of a 3x3 matrix."""
    I = empty((3,3))
    I[0,0] = A[1,1]*A[2,2]-A[1,2]*A[2,1]
    I[1,1] = A[0,0]*A[2,2]-A[0,2]*A[2,0]
    I[2,2] = A[0,0]*A[1,1]-A[0,1]*A[1,0]
    I[1,0] = -A[1,0]*A[2,2]+A[1,2]*A[2,0]
    I[0,1] = -A[0,1]*A[2,2]+A[0,2]*A[2,1]
    I[2,0] = A[1,0]*A[2,1]-A[1,1]*A[2,0]
    I[0,2] = A[0,1]*A[1,2]-A[0,2]*A[1,1]
    I[2,1] = -A[0,0]*A[2,1]+A[0,1]*A[2,0]
    I[1,2] = -A[0,0]*A[1,2]+A[0,2]*A[1,0]
    d = A[0,0]*A[1,1]*A[2,2]+A[0,1]*A[1,2]*A[2,0]+A[0,2]*A[1,0]*A[2,1]-(A[0,2]*A[1,1]*A[2,0]+A[0,0]*A[1,2]*A[2,1]+A[0,1]*A[1,0]*A[2,2])
    return I/d    
