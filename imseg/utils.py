from numpy import outer, dot, zeros, empty, inner, trace
from numpy.random import multivariate_normal as mvn
from math import sqrt, log

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
 
def det3(A):
    """Return the determinant of a 3x3 matrix."""
    return (A[0,0]*A[1,1]*A[2,2]+A[0,1]*A[1,2]*A[2,0]+
            A[0,2]*A[1,0]*A[2,1]-(A[0,2]*A[1,1]*A[2,0]+
            A[0,0]*A[1,2]*A[2,1]+A[0,1]*A[1,0]*A[2,2]))
       
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
    d = det3(A)
    return I/d    

def logNormPDF(x,mu,V):
    """
    Return the log of the PDF of a MVN with mean mu and precision V 
    evaluated at x (up to additive constant of proportionality).
    
    Parameters
    ----------
    x : (d,) ndarray
    mu : (d,) ndarray
    V : (d,d) ndarray
        Must be symmetric, positive definite
        
    Returns
    -------
    d : float
    """
    return sqrt(det3(V))-0.5*inner(x-mu,V.dot(x-mu))
    
def logWisPDF(X,nu,Vinv):
    """
    Return the log of the PDF of a wishart with parameters nu, V 
    evaluated at X (up to additive constant of proportionality not
    dependent on X).
    
    Parameters
    ----------
    X : (3,3) ndarray
        Must be positive definite.
    nu : int
        Must be >= 3
    Vinv : (3,3) ndarray
        The inverse of the scale matrix V of the distribution
    
    Returns
    -------
    logpdf : float
    """
    return (nu-4)/2.*log(det3(X)) - .5*log(trace(dot(Vinv,X)))
