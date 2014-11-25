import sys

from numpy import zeros, ones, log, exp, sum
from numpy.random import randint, random
from scipy.stats.distributions import norm, gamma 
from scipy.misc import logsumexp

from factors import phi

def segment(image, n_segments=2, burn_in=1000, samples=1000, lag=5):
    """
    Return image segment samples.

    Parameters
    ----------
    image : (N,M) ndarray
        Pixel array with single-dimension values (e.g. hue)

    Returns
    -------
    labels : (samples,N,M) ndarray
        The image segment label array
    emission_params: (samples,K,2) ndarray
        The Gaussian emission distribution parameters (mean, precision)
    log_probs : (samples,) ndarray
    """

    # allocate arrays
    res_labels = zeros((samples, image.shape[0], image.shape[1]), dtype=int)
    res_emission_params = zeros((samples, n_segments, 2))
    res_log_prob = zeros((samples,))

    padded_labels = ones((image.shape[0] + 2, image.shape[1] + 2), dtype=int)*-1
    labels = padded_labels[1:-1, 1:-1]
    emission_params = zeros((n_segments, 2))
    log_prob = None

    conditional = zeros((n_segments,))


    # init emission_params
    sample_mean = image.mean()
    sample_var = image.var()
    sample_prec = 1./sample_var
    for k in xrange(n_segments):
        emission_params[k,0] = norm.rvs(sample_mean, sample_var/n_segments)
        emission_params[k,1] = sample_prec

    # init labels
    for n in xrange(image.shape[0]):
        for m in xrange(image.shape[1]):
            labels[n,m] = randint(0, n_segments)

    try:
        # gibbs
        for i in xrange(burn_in + samples*lag - (lag - 1)):

            for n in xrange(image.shape[0]):
                for m in xrange(image.shape[1]):
                    # resample label
                    for k in xrange(n_segments):
                        labels[n,m] = k
                        blanket = padded_labels[n:n+3,m:m+3]
                        mean = emission_params[k, 0]
                        var = 1./emission_params[k, 1]
                        conditional[k] = phi(blanket)
                        conditional[k] += log(norm.pdf(image[n,m], mean, var))
                    labels[n,m] = sample_categorical(conditional)

            for k in xrange(n_segments):
                mask = (labels == k)

                # resample label mean
                mean = emission_params[k, 0]
                prec = 1./emission_params[k, 1]
                numer = sample_prec*sample_mean + prec*sum(image[mask])
                denom = sample_prec + prec*sum(mask)
                post_mean = numer/denom
                post_var = 1./(denom)
                emission_params[k, 0] = norm.rvs(post_mean, post_var)

                # resample label var
                post_alpha = 1. + sum(mask)/2.
                post_beta = sample_prec + sum((image[mask] - emission_params[k,0])**2)/2.
                post = gamma(post_alpha, scale=1./post_beta)
                emission_params[k, 1] = post.rvs()
                
            log_prob = 0.
            for n in xrange(image.shape[0]):
                for m in xrange(image.shape[1]):
                    blanket = padded_labels[n:n+3,m:m+3]
                    label = labels[n,m]
                    mean = emission_params[label, 0]
                    var = 1./emission_params[label, 1]
                    log_prob += phi(blanket)
                    log_prob += log(norm.pdf(image[n,m], mean, var))

            sys.stdout.write('\riter {} log_prob {}'.format(i, log_prob))
            sys.stdout.flush()

            if i < burn_in:
                pass
            elif not (i - burn_in)%lag:
                res_i = i/lag
                res_emission_params[res_i] = emission_params[:]
                res_labels[res_i] = labels
                res_log_prob[i] = log_prob

        sys.stdout.write('\n')
        return res_labels, res_emission_params, res_log_prob
    except KeyboardInterrupt:
        return res_labels, res_emission_params, res_log_prob

def sample_categorical(p):
    """Sample a categorical parameterized by (unnormalized) exp(p)."""
    q = exp(p - logsumexp(p))
    r = random()
    k = 0
    while k < len(q) - 1 and q[k] <= r:
        r -= q[k]
        k += 1
    return k
