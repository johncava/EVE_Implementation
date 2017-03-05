from __future__ import division
import numpy as np
import math

# Objective function
def function(x):
    return x**2

# Gradient of the objective function
def gradient(x):
    return 2*x

# alpha is the learning rate
# b1, b2, b3 in [0,1) as exponential decay
# k < acceptable relative change < K 
# epsilon, used to avoid division by Zero 
# x is the parameter vector to minize respect to objective function
def eve(alpha, b1, b2, b3, k, K, epsilon, x):
    m, v = 0, 0
    d = 1
    # Used a dictionary to 'remember' f_hat values of t-2, and t-1 in respect to current t timestep
    f_hat = {}
    f_hat[-1] =np.linalg.norm(np.array([0,0,0]))
    t = 0
    for iterations in xrange(100000):
        t = t + 1
        g = gradient(x)
        m = b1*m + (1 - b1)*g
        m_hat = m/(1 - b1 ** t)
        v = b2*v + (1 - b2)*(g**2)
        v_hat = v/(1 - b2 ** t)
        if t > 1:
            if np.linalg.norm(function(x)) >= f_hat[t - 2]:
                sigma_t = k + 1 
                delta_t = K + 1
            else:
                sigma_t = 1/(K + 1)
                delta_t = 1/(k + 1)
            c = min(max(sigma_t, np.linalg.norm(function(x))/f_hat[t - 2]), delta_t)
            f_hat[t - 1] = c*f_hat[t - 2]
            r = math.fabs(f_hat[t-1] - f_hat[t-2])
            d = b3*d + (1 - b3)*r
        else:
            f_hat[t - 1] = np.linalg.norm(function(x))
            d = 1
        x = x - alpha*(m_hat)/(d*(v_hat**(1/2)) + epsilon)
        if t % 1000 == 0:
            print x

a = np.array([100,75,130])
eve(0.1, 0.9, 0.999, 0.999, 0.1, 10,10**-8, a)