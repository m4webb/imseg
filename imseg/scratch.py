# Authors: Matthew Webb, Abe Frandsen
import sys

from numpy import zeros, ones, log, exp, sum, mean, dot, array, eye, outer
from numpy.random import randint, random, multivariate_normal as mvn
from scipy.stats.distributions import norm, gamma
from scipy.misc import logsumexp

from utils import wishart, logNormPDF, logWisPDF, inv3
from factors import phi

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
    log_probs : (samples,) ndarray
    """
    N,M = image.shape[:2]
    # subtract out the mean
    image -= array([image[:,:,i].mean() for i in xrange(3)])
    
    # allocate arrays
    res_labels = zeros((samples,N,M))
    res_emission_mu = zeros((samples,n_segments,3))
    res_emission_prec = zeros((samples,n_segments,3,3))
    res_log_prob = zeros((samples,))

    # initialize hyperparmeters
    nu = 4 # df hyperparameter for wishart (uninformative?)
    V = eye(3) # scale matrix hyperparameter for wishart (uninformative?)
    Vinv = inv3(V) 
    mu = zeros(3) # mean hyperparameter for MVN
    S = eye(3) # precision hyperparameter for MVN (uninformative?)
    
    # initialize labels/params
    padded_labels = ones((N+2,M+2), dtype=int)*-1
    padded_labels[1:-1,1:-1] = randint(n_segments,size=(N,M))
    labels = padded_labels[1:-1, 1:-1]
    emission_mu = mvn(mu,S,size=n_segments) # draw from prior
    emission_prec = array([wishart(V,nu) for i in xrange(n_segments)]) # draw from prior
    log_prob = None

    conditional = zeros((n_segments,))
    try:
        # gibbs
        for i in xrange(burn_in + samples*lag - (lag - 1)):
            for n in xrange(N):
                for m in xrange(M):
                    # resample label
                    for k in xrange(n_segments):
                        labels[n,m] = k
                        conditional[k] = 0.
                        # x,y are relative to image, not padded_labels
                        for x in xrange(max(0,n-2), min(N,n+3)):
                            for y in xrange(max(0,m-2), min(M,m+3)):
                                clique = padded_labels[x:x+3,y:y+3]
                                conditional[k] += phi(clique)
                        conditional[k] += logNormPDF(image[n,m,:], 
                                                    emission_mu[k],
                                                    emission_prec[k])
                    labels[n,m] = sample_categorical(conditional)

            for k in xrange(n_segments):
                mask = (labels == k)
                n_k = sum(mask)
                
                # resample label mean
                P = inv3(S+n_k*emission_prec[k])
                xbar = mean(image[mask],axis=0)
                emission_mu[k,:] = mvn(dot(P,n_k*dot(emission_prec[k],xbar)),P)
                
                
                # resample label precision
                if n_k:
                    D = outer(image[mask][0,:]-emission_mu[k,:],image[mask][0,:]-emission_mu[k,:])
                    for ii in xrange(1,n_k):
                        D += outer(image[mask][ii,:]-emission_mu[k,:],image[mask][ii,:]-emission_mu[k,:])
                    emission_prec[k,:,:] = wishart(inv3(Vinv+D),nu+n_k)
                else:
                    emission_prec[k,:,:] = wishart(V,nu) # resample using the prior

            log_prob = 0.
            for c in xrange(n_segments):
                log_prob += logNormPDF(emission_mu[c],mu,S)
                log_prob += logWisPDF(emission_prec[c],nu,Vinv)
            for n in xrange(N):
                for m in xrange(M):
                    clique = padded_labels[n:n+3,m:m+3]
                    label = labels[n,m]
                    log_prob += phi(clique)
                    log_prob += logNormPDF(image[n,m,:],emission_mu[label],emission_prec[label])

            sys.stdout.write('\riter {} log_prob {}'.format(i, log_prob))
            sys.stdout.flush()

            if i < burn_in:
                pass
            elif not (i - burn_in)%lag:
                res_i = i/lag
                res_emission_mu[res_i] = emission_mu[:,:]
                res_emission_prec[res_i] = emission_prec[:,:,:]
                res_labels[res_i] = labels
                res_log_prob[i] = log_prob
            
        sys.stdout.write('\n')
        return res_labels, res_emission_mu, res_emission_prec, res_log_prob
    except KeyboardInterrupt:
        return res_labels, res_emission_mu, res_emission_prec, res_log_prob


def sample_labels_prior(M,N, n_segments=2, samples=1000):
    """
    Sample from the prior over cluster assignments using Gibbs sampling.
    """
    res_labels = zeros((samples+1, M, N), dtype=int)
    padded_labels = ones((M+2,N+2), dtype=int)*-1
    #padded_labels[1:-1,1:-1] = randint(n_segments,size=(M,N))
    padded_labels[1:M/2,1:-1] = 1
    padded_labels[M/2:-1,1:-1] = 0
    labels = padded_labels[1:-1, 1:-1]
    res_labels[0,:,:] = labels
    lprobs = zeros(samples)
    conditional = zeros((n_segments,))
    for i in xrange(samples):
        for n in xrange(M):
            for m in xrange(N):
                # resample label
                for k in xrange(n_segments):
                    labels[n,m] = k
                    conditional[k] = 0.
                    # x,y are relative to image, not padded_labels
                    for x in xrange(max(0,n-1), min(N,n+2)):
                        for y in xrange(max(0,m-1), min(M,m+2)):
                            clique = padded_labels[x:x+3,y:y+3]
                            conditional[k] += phi(clique)
                if m==0 and n==0:
                    print conditional
                labels[n,m] = sample_categorical(conditional)
        res_labels[i+1] = labels
        log_prob=0
        for n in xrange(M):
            for m in xrange(N):
                clique = padded_labels[n:n+3,m:m+3]
                label = labels[n,m]
                log_prob += phi(clique)
        lprobs[i] = log_prob
        sys.stdout.write('\riter {} log_prob {}'.format(i, log_prob))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return res_labels, lprobs
    
def sample_categorical(p):
    """Sample a categorical parameterized by (unnormalized) exp(p)."""
    q = exp(p - logsumexp(p))
    r = random()
    k = 0
    while k < len(q) - 1 and q[k] <= r:
        r -= q[k]
        k += 1
    return k
