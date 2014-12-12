# Authors: Matthew Webb, Abe Frandsen
import sys

from utils import wishart, logNormPDF, logWisPDF, inv3
from factors import phi
from phi import phi_all, phi_blanket

from numpy import zeros, ones, log, exp, sum, mean, dot, array, eye, outer
from numpy import allclose, load, abs, max
from numpy.random import randint, random, multivariate_normal as mvn
from scipy.stats.distributions import norm, gamma
from scipy.misc import logsumexp

FS = load('factors.npy')*1.0 #allow optimized factor computation

def normalize(image):
    image = image.copy().astype(float)
    image -= array([image[:,:,i].mean() for i in xrange(3)])
    return image
 
def segment(image, n_segments=2, burn_in=1000, samples=1000, lag=5):
    """
    Return image segment samples.

    Parameters
    ----------
    image : (N,M,3) ndarray
        Pixel array with 3-dimensional values (e.g. RGB)

    Returns
    -------
    labels : (samples,N,M) ndarray
        The image segment label array
    emission_mus: (samples,n_segments,3) ndarray
        The Gaussian emission distribution (mean).
    emission_precs : (samples,n_segments,3,3) ndarray
        The Gaussian emission distribution (precision).
    log_prob_p : (samples,) ndarray
        Log probability portion contributed by prior
    log_prob_e : (samples,) ndarray
        Log probability portion contributed by emissions
    """
    RETURN_MINUS_ONE = False
    N,M = image.shape[:2]
    # subtract out the mean
    image = normalize(image)   

    # allocate arrays
    res_labels = zeros((samples+1,N,M))
    res_emission_mu = zeros((samples,n_segments,3))
    res_emission_prec = zeros((samples,n_segments,3,3))
    res_log_prob_p = zeros((samples,))
    res_log_prob_e = zeros((samples,))

    # initialize hyperparmeters
    nu = 4 # df hyperparameter for wishart (uninformative?)
    V = eye(3) # scale matrix hyperparameter for wishart (uninformative?)
    Vinv = inv3(V) 
    mu = zeros(3) # mean hyperparameter for MVN
    S = eye(3)*0.0001 # precision hyperparameter for MVN (uninformative?)
    
    # initialize labels/params
    padded_labels = ones((N+2,M+2), dtype=int)*-1
    padded_labels[1:-1,1:-1] = randint(n_segments,size=(N,M))
    labels = padded_labels[1:-1, 1:-1]
    for n in xrange(N):
        for m in xrange(M):
            labels[n,m] = (n*M + m)%n_segments
    res_labels[-1] = labels
    emission_mu = mvn(mu,S,size=n_segments) # draw from prior
    #emission_prec = array([wishart(V,nu) for i in xrange(n_segments)]) # draw from prior
    emission_prec = array([eye(3)*0.0001 for i in xrange(n_segments)])
    log_prob_p = None
    log_prob_e = None

    conditional = zeros((n_segments,))
    print "Starting the sampler..."
    try:
        # gibbs
        for i in xrange(burn_in + samples*lag - (lag - 1)):
            for n in xrange(N):
                for m in xrange(M):
                    # resample label
                    for k in xrange(n_segments):
                        labels[n,m] = k
                        conditional[k] = 0.
                        conditional[k] += phi_blanket(
                                memoryview(padded_labels), n, m,
                                memoryview(FS))
                        conditional[k] += logNormPDF(image[n,m,:],
                                emission_mu[k], emission_prec[k])
                    labels[n,m] = sample_categorical(conditional)

            for k in xrange(n_segments):
                mask = (labels == k)
                n_k = sum(mask)
                
                # resample label mean
                if n_k:
                    P = inv3(S+n_k*emission_prec[k])
                    xbar = mean(image[mask],axis=0)
                    emission_mu[k,:] = mvn(dot(P,n_k*dot(emission_prec[k],xbar)),P)
                else:
                    emission_mu = mvn(mu,S,size=n_segments) # draw from prior


                # resample label precision
                if n_k:
                    D = outer(image[mask][0,:]-emission_mu[k,:],image[mask][0,:]-emission_mu[k,:])
                    for ii in xrange(1,n_k):
                        D += outer(image[mask][ii,:]-emission_mu[k,:],image[mask][ii,:]-emission_mu[k,:])
                    emission_prec[k,:,:] = wishart(inv3(Vinv+D),nu+n_k)
                else:
                    emission_prec[k,:,:] = wishart(V,nu)

            log_prob_e = 0.
            for c in xrange(n_segments):
                log_prob_e += logNormPDF(emission_mu[c],mu,S)
                log_prob_e += logWisPDF(emission_prec[c],nu,Vinv)
            for n in xrange(N):
                for m in xrange(M):
                    label = labels[n,m]
                    log_prob_e += logNormPDF(image[n,m,:],emission_mu[label],emission_prec[label])
            log_prob_p = phi_all(memoryview(padded_labels), memoryview(FS))

            sys.stdout.write('\riter {} log_prob_prior {} '
                    'log_prob_emission {} k0 {} k1 {}           '.format(i,
                        log_prob_p, log_prob_e, sum(labels==0), sum(labels==1)))
            sys.stdout.flush()

            if not sum(labels==0) or not sum(labels==1):
                RETURN_MINUS_ONE = True
                raise ValueError("converged to 0...")

            if i < burn_in:
                pass
            elif not (i - burn_in)%lag:
                res_i = i/lag
                res_emission_mu[res_i] = emission_mu[:,:]
                res_emission_prec[res_i] = emission_prec[:,:,:]
                res_labels[res_i] = labels
                res_log_prob_p[i] = log_prob_p
                res_log_prob_e[i] = log_prob_e

        sys.stdout.write('\n')
    except Exception as e:
        print e
    finally:
        if RETURN_MINUS_ONE:
            return -1
        else:
            return (res_labels, res_emission_mu, res_emission_prec,
                    res_log_prob_p, res_log_prob_e)

def sample_categorical(p):
    """Sample a categorical parameterized by (unnormalized) exp(p)."""
    q = exp(p - logsumexp(p))
    if not allclose(sum(q), 1.):
        print p
        print q
        raise ValueError("ahh!")
    r = random()
    k = 0
    while k < len(q) - 1 and q[k] <= r:
        r -= q[k]
        k += 1
    return k
