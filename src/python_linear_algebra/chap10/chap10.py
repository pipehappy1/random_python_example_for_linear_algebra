import numpy as np
from functools import partial


def phi(x, M=1):
    #print(x)
    res = 0
    res += x[0]**2 + x[1]**2
    #print(res, "res")
    term1 = (np.max((20 - x[0], 0)))**2
    #print(term1)
    term2 = (np.max((2*x[0] - x[1] - 30, 0)))**2
    #print(term2)
    res += M*(term1 + term2)
    #print(res, "res")
    return res

def phi_g(x, M=1):
    res = np.zeros(2)
    
    res[0] = 2*x[0]
    if 20 - x[0] > 0:
        res[0] += 2 * M * (20 - x[0]) * (-1)
    else:
        res[0] += 0
    #print(res, "res")
    if 2*x[0] - x[1] - 30 > 0:
        res[0] += 2 * M * (2*x[0] - x[1] - 30) * 2
    else:
        res[0] += 0
    #print(res, "res")

    res[1] = 2*x[1]
    if 2*x[0] - x[1] - 30 > 0:
        res[1] += 2 * M * (2*x[0] - x[1] - 30) * (-1)
    else:
        res[1] += 0

    return res

def phi_gg(x, M=1):
    res = np.zeros((2,2))

    res[0,0] = 2
    res[1,1] = 2

    if 20 - x[0] > 0:
        res[0,0] += 2*M

    if 2*x[0] - x[1] - 30 > 0:
        res[0,0] += 2*M*2*2
        res[0,1] += -2*M*2
        res[1,1] += 2*M
        res[1,0] += -2*M*2
    

    return res


def newton(f, f_g, f_gg, x, epsilon=0.001):
    for i in range(20):
        g = f_g(x)
        gg = f_gg(x)
        if np.dot(g, g) < epsilon:
            return x
        else:
            p = np.linalg.solve(gg, -g)
            x = x + p

    print("Warning: iteration max reached!")
    return x

def optimize_structure():

    M = 1
    x = np.array([20, 20])
    epsilon = 10e-6
    c = 8
    
    for i in range(20):
        f_g = partial(phi_g, M=M)
        f_gg = partial(phi_gg, M=M)
    
        x_n = newton(None, f_g, f_gg, x)
    
        if np.dot(x_n - x, x_n - x) <= epsilon \
           and np.abs((np.dot(x_n, x_n) - np.dot(x, x))/np.dot(x, x)) <= epsilon:
            return x_n
        else:
            x = x_n
            M = c*M

if __name__ == "__main__":
    print("The optimal position of A, B are {}.".format(optimize_structure()))
