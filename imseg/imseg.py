# Authors: Matthew Webb, Abe Frandsen
import sys

from numpy import zeros, ones, log, exp, sum, sqrt, load
from numpy.random import randint, random
from scipy.stats.distributions import norm, gamma 
from scipy.misc import logsumexp

from phi import phi_blanket, phi_all

TAU_0 = 0.1
MU_0 = 0.5
BETA_0 = 1./2.
ALPHA_0 = 3.

FS = load('factors.npy')

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
    res_emission_params = zeros((samples, n_segments, 6))
    res_log_prob = zeros((samples,))

    padded_labels = ones((image.shape[0] + 2, image.shape[1] + 2), dtype=int)*-1
    labels = padded_labels[1:-1, 1:-1]
    emission_params = zeros((n_segments, 6))
    log_prob = None

    conditional = zeros((n_segments,))


    # init emission_params
    sample_mean_r = image[:,:,0].mean()
    sample_mean_g = image[:,:,1].mean()
    sample_mean_b = image[:,:,2].mean()
    sample_var_r = image[:,:,0].var()
    sample_var_g = image[:,:,1].var()
    sample_var_b = image[:,:,2].var()
    sample_prec_r = 1./sample_var_r
    sample_prec_g = 1./sample_var_g
    sample_prec_b = 1./sample_var_b
    for k in xrange(n_segments):
        """
        emission_params[k,0] = norm.rvs(sample_mean_r,
                sqrt(sample_var_r/n_segments))
        emission_params[k,1] = sample_prec_r
        emission_params[k,2] = norm.rvs(sample_mean_g,
                sqrt(sample_var_g/n_segments))
        emission_params[k,3] = sample_prec_g
        emission_params[k,4] = norm.rvs(sample_mean_b,
                sqrt(sample_var_b/n_segments))
        emission_params[k,5] = sample_prec_b
        """
        emission_params[k,0] = norm.rvs(0.5, 0.1)
        emission_params[k,1] = 1/(0.25**2)
        emission_params[k,2] = norm.rvs(0.5, 0.1)
        emission_params[k,3] = 1/(0.25**2)
        emission_params[k,4] = norm.rvs(0.5, 0.1)
        emission_params[k,5] = 1/(0.25**2)

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
                        conditional[k] = 0.
                        conditional[k] += phi_blanket(
                                memoryview(padded_labels), n, m,
                                memoryview(FS))
                        """
                        for x in xrange(max(n-2,0), min(n+3,image.shape[0])):
                            for y in xrange(max(m-2,0), min(m+3,
                                    image.shape[1])):
                                clique = padded_labels[x:x+3,y:y+3]
                                conditional[k] += phi(clique)
                        """
                        mean_r = emission_params[k, 0]
                        var_r = 1./emission_params[k, 1]
                        mean_g = emission_params[k, 2]
                        var_g = 1./emission_params[k, 3]
                        mean_b = emission_params[k, 4]
                        var_b = 1./emission_params[k, 5]
                        conditional[k] += log(norm.pdf(image[n,m,0], mean_r,
                            sqrt(var_r)))
                        conditional[k] += log(norm.pdf(image[n,m,1], mean_g,
                            sqrt(var_g)))
                        conditional[k] += log(norm.pdf(image[n,m,2], mean_b,
                            sqrt(var_b)))

                    labels[n,m] = sample_categorical(conditional)

            for k in xrange(n_segments):
                mask = (labels == k)

                # resample label mean red
                mean_r = emission_params[k, 0]
                prec_r = emission_params[k, 1]
                numer_r = TAU_0*MU_0 + prec_r*sum(image[mask][:, 0])
                denom_r = TAU_0 + prec_r*sum(mask)
                post_mean_r = numer_r/denom_r
                post_var_r = 1./(denom_r)
                emission_params[k, 0] = norm.rvs(post_mean_r, sqrt(post_var_r))

                # resample label var red
                post_alpha_r = ALPHA_0 + sum(mask)/2.
                post_beta_r = BETA_0 + sum((image[mask][:, 0] - emission_params[k,0])**2)/2.
                post_r = gamma(post_alpha_r, scale=1./post_beta_r)
                emission_params[k, 1] = post_r.rvs()

                # resample label mean green
                mean_g = emission_params[k, 2]
                prec_g = emission_params[k, 3]
                numer_g = TAU_0*MU_0 + prec_g*sum(image[mask][:, 1])
                denom_g = TAU_0 + prec_g*sum(mask)
                post_mean_g = numer_g/denom_g
                post_var_g = 1./(denom_g)
                emission_params[k, 2] = norm.rvs(post_mean_g, sqrt(post_var_g))

                # resample label var green
                post_alpha_g = ALPHA_0 + sum(mask)/2.
                post_beta_g = BETA_0 + sum((image[mask][:, 1] - emission_params[k,2])**2)/2.
                post_g = gamma(post_alpha_g, scale=1./post_beta_g)
                emission_params[k, 3] = post_g.rvs()

                # resample label mean blue
                mean_b = emission_params[k, 4]
                prec_b = emission_params[k, 5]
                numer_b = TAU_0*MU_0 + prec_b*sum(image[mask][:, 2])
                denom_b = TAU_0 + prec_b*sum(mask)
                post_mean_b = numer_b/denom_b
                post_var_b = 1./(denom_b)
                emission_params[k, 4] = norm.rvs(post_mean_b, sqrt(post_var_b))

                # resample label var blue
                post_alpha_b = ALPHA_0 + sum(mask)/2.
                post_beta_b = BETA_0 + sum((image[mask][:, 2] - emission_params[k,4])**2)/2.
                post_b = gamma(post_alpha_b, scale=1./post_beta_b)
                emission_params[k, 5] = post_b.rvs()
                
            log_prob = 0.
            for n in xrange(image.shape[0]):
                for m in xrange(image.shape[1]):
                    #clique = padded_labels[n:n+3,m:m+3]
                    label = labels[n,m]
                    mean_r = emission_params[label, 0]
                    var_r = 1./emission_params[label, 1]
                    mean_g = emission_params[label, 2]
                    var_g = 1./emission_params[label, 3]
                    mean_b = emission_params[label, 4]
                    var_b = 1./emission_params[label, 5]
                    #log_prob += phi(clique)
                    log_prob += log(norm.pdf(image[n,m,0], mean_r, sqrt(var_r)))
                    log_prob += log(norm.pdf(image[n,m,1], mean_g, sqrt(var_g)))
                    log_prob += log(norm.pdf(image[n,m,2], mean_b, sqrt(var_b)))
                    # prior on theta?
            log_prob += phi_all(memoryview(padded_labels), memoryview(FS))

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
