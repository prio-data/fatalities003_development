import numpy as np
import matplotlib.pyplot as plt


def tweedie_loss(p, q, pow=1.5, eps=np.exp(-100)):
    """
    The Tweedie loss function is defined as: $L(p, q) = -p q^{1-pow} / (1-pow) + q^{2-pow} / (2-pow)$. 
    I is used to evaluate the performance of a model that predicts the mean of a Tweedie distribution. 
    The parameter $pow$ controls the variance of the distribution. For $pow < 1$, the variance is infinite. 
    For $pow = 1$, the variance is 0. For $pow > 1$, the variance is finite. 
    The parameter $eps$ is used to avoid numerical issues when $pow < 1$.
    """

    p = p + eps
    q = q + eps

    loss = -p * np.power(q, 1 - pow) / (1 - pow) + \
        np.power(q, 2 - pow) / (2 - pow)
    return np.mean(loss)


def kl_divergence(p, q, eps=np.exp(-100)):
    """
    The KL divergence between two discrete distributions p and q is defined as: $\sum_i p_i \log(p_i / q_i)$. 
    It describes the difference between two distributions in terms of information lost when q is used to approximate p.
    The parameter $eps$ is used to avoid numerical issues when $pow < 1$.
    """

    p = p + eps
    q = q + eps

    return np.sum(p * np.log(p / q))


def jeffreys_divergence(p, q, eps=np.exp(-100)):
    """
    Jeffreys divergence is a symmetrized version of KL divergence. See https://en.wikipedia.org/wiki/Hellinger_distance
    The parameter $eps$ is used to avoid numerical issues when $pow < 1$.
    """

    p = p + eps
    q = q + eps

    return 0.5 * np.sum((p - q) * np.log(p / q))


def jenson_shannon_divergence(p, q, eps=np.exp(-100)):
    """
    Jenson-Shannon divergence is also a symmetrized version of KL divergence. See https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    JSD = 0.5 * KL(p, m) + 0.5 * KL(q, m)
    The parameter $eps$ is used to avoid numerical issues when $pow < 1$.
    """

    p = p + eps
    q = q + eps

    m = 0.5 * (p + q)

    return 0.5 * np.sum(p * np.log(p / m) + q * np.log(q / m))
