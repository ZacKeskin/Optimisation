import numpy as np
from linesearch import linesearch, linesearch_wrapper, backtracking

# Wrapper class to store and return iteration info in a consistent manner
class opt_result():
    def __init__(self, x, k, f, errors):
        self.xs = x
        self.fs = f
        self.k = k
        self.fun = self.fs[k]
        self.x = self.xs[k]
        self.es = errors

# Steepest Gradient Descent
def SGD(x0, func, jac, hess=None, alpha0=1, tol=1e-6, maxiter=1e4):
    x = np.array(x0).reshape(1,len(x0))
    f = func(x0)
    k = 0
    es = np.linalg.norm(jac(x0))

    stop = False
    while stop==False and k < maxiter:
        p_k = -jac(x[k]) # Descent direction
        
        alpha = backtracking(func,jac,x[k],p_k)

        x = np.vstack((x, x[k] + alpha * p_k))
        k=k+1
        f = np.vstack((f, func(x[k])))
        
        stop = (np.linalg.norm(jac(x[k])) < tol*(1 + tol * abs(func(x[k]))))
        es = np.vstack((es, np.linalg.norm(jac(x[k])) ))

    res = opt_result(x,k,f,es)
    return res

# Newton
def Newton(x0, func, jac, hess, alpha0=1, tol=1e-6, maxiter=1e3):
    N = len(x0)
    x = np.array(x0).reshape(1,N)
    f = func(x0)
    es = np.linalg.norm(jac(x0)) 
    k = 0
    

    stop = False
    while stop == False and k < maxiter:
        H_k = np.linalg.inv(hess(x[k]))  # Descent direction
        p_k = - np.dot(H_k,jac(x[k]))

        if np.dot(p_k, jac(x[k])) > 0: # force to be descent direction (just in case Hess is no longer SPD due to penalty terms)
            p_k = -p_k
        
        alpha = backtracking(func, jac, x[k], p_k)
        
        x = np.vstack((x, x[k] + alpha * p_k))
        k=k+1
        f = np.vstack((f,func(x[k])))

        stop = (np.linalg.norm(jac(x[k])) < tol*(1 + tol * abs(func(x[k]))))
        es = np.vstack((es, np.linalg.norm(jac(x[k])) ))

    res = opt_result(x,k,f,es)
    return res

# BFGS
def BFGS(x0, func, jac, hess=None, alpha0=1, tol=1e-6, maxiter=1e3):
    # Algorithm 6.1 from Nocedal & Wright
    N = len(x0)
    x = np.array(x0).reshape(1,N)
    f = func(x0)
    es = np.linalg.norm(jac(x0))

    k = 0
    I = np.eye(N, dtype=int)
    H_k = I 

    stop = False
    while stop==False and k < maxiter:
        # Step direction p_k
        p_k = - np.matmul(H_k, jac(x[k]))


        # Find Step length alpha
        alpha = backtracking(func, jac, x[k], p_k) 

        # Update x_k, f(x_k)
        x = np.vstack((x, x[k] + alpha * p_k))
        k=k+1
        f = np.vstack((f, func(x[k])))

        # Careful to enforce 1D vector shape, or np.matmul and np.dot do not behave as expected
        s_k = (x[k] - x[k-1]).reshape(N,1)
        y_k = (jac(x[k]) - jac(x[k-1]) ).reshape(N,1)   
        rho_k = 1/np.dot(y_k.transpose(), s_k)

        if k == 1:
            # Update initial guess H_0.
            H_k = H_k * np.matmul(s_k.transpose(),y_k) / np.matmul(y_k.transpose(),y_k)
        
        # Update H_k using 6.17 from Nocedal & Wright
        H_k_1 = H_k
        
        A1 = I - rho_k * np.matmul(s_k,y_k.transpose())
        A2 = I - rho_k * np.matmul(y_k,s_k.transpose())
        H_k = np.matmul( A1, np.matmul(H_k_1, A2)) + rho_k * np.matmul(s_k,s_k.transpose())

        stop = (np.linalg.norm(jac(x[k])) < tol*(1 + tol * abs(func(x[k]))))
        es = np.vstack((es, np.linalg.norm(jac(x[k])) ))
        
    res = opt_result(x,k,f,es)
    return res 
