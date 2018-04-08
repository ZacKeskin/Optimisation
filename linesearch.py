import numpy as np
from scipy.optimize.optimize import _line_search_wolfe12


# Handlers for Phi() and Phi'()
def phi(func, x_k, p_k, alpha):
    x_k.resize(len(x_k),1)
    p_k.resize(len(p_k),1)
    myphi = func(x_k + alpha * p_k)
    return myphi
    
def der_phi(jac, x_k, p_k, alpha):
    N = len(x_k)
    x_k.resize(N,1)
    p_k.resize(N,1)
    arg = (x_k + alpha * p_k).flatten()
    dphi = np.matmul( jac(arg), p_k)
    return dphi


# Algorithm 3.5 from Nocedal & Wright (Not Working)
def linesearch(func, jac, x_k, p_k, alpha_max=1, c1 = 0.0001, c2=0.1, maxIter=25):
    
    alphas = [0, 0.9*alpha_max]
    i = 1

    while i < maxIter:
        
        # Calculate required terms
        phi_i = phi(func,x_k,p_k,alphas[i])
        phi_i_1 = phi(func,x_k,p_k,alphas[i-1])
        phi_prime_0 = der_phi(jac,x_k,p_k, 0)
        phi_0 = phi(func,x_k,p_k, 0) + c1 * alphas[i] * phi_prime_0
        
        if phi_i > phi_0 or (phi_i >= phi_i_1 and i > 1):
            a_star = zoom(alphas[i-1],alphas[i],func, jac, x_k, p_k, c1, c2)
            #print('Linesearch found alpha*: ', a_star)
            return a_star
        phi_prime_i = der_phi(jac,x_k,p_k, alphas[i])
        
        if abs(phi_prime_i) <= -c2*phi_prime_0:
            a_star = alphas[i]
            #print('Linesearch found alpha*: ', a_star)
            return a_star

        if phi_prime_i >= 0:
            a_star = zoom(alphas[i],alphas[i-1],func, jac, x_k, p_k, c1, c2)
            #print('Linesearch found alpha*: ', a_star)
            return a_star

        # Update alpha_i and go again
        alphas.append(min(2*alphas[i], alpha_max))
        i+=1

    print("linesearch failed to converge")
    return 0 # If convergence fails

# Algorithm 3.6 from Nocedal & Wright (Not Working)
def zoom(alpha_lo, alpha_hi, func, jac, x_k, p_k, c1, c2, maxIter=10, tol = 1e-8):
    if alpha_lo > alpha_hi:
        temp = alpha_lo
        alpha_lo = alpha_hi 
        alpha_hi = temp
    j=0
    while j < maxIter:
        alpha_j = (alpha_hi + alpha_lo)/2

        # Calculate required terms
        phi_j = phi(func,x_k,p_k,alpha_j)
        phi_prime_0 = der_phi(jac, x_k, p_k, 0)
        phi_0 = phi(func,x_k,p_k, 0) 
        phi_lo = phi(func,x_k,p_k,alpha_lo)

        if abs(alpha_hi - alpha_lo) < tol:
            return alpha_j
            print('Line search stopped because the interval became to small. Return centre of the interval.')

        if phi_j > (phi_0 + c1 * alpha_j * phi_prime_0) or phi_j >= phi_lo:
            alpha_hi = alpha_j
        else:
            # alpha_j satisfies sufficient decrease condition
            phi_prime_j = der_phi(jac, x_k, p_k, alpha_j)
            
            if abs(phi_prime_j) <= -c2*phi_prime_0:
                # alpha_j satisfies strong curvature condition
                return alpha_j

            if phi_prime_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            
            alpha_lo = alpha_j
        j+=1
    print("zoom failed to converge")
    return alpha_j # If convergence fails

# Wrapper for line_search_wolfe12
def linesearch_wrapper(func, jac, x_k, p_k, gfk, old_fval, old_old_fval, alpha_max, c1 = 0.0001, c2=0.1, maxIter=100):

    alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(func, jac, x_k, p_k, gfk,
                                          old_fval, old_old_fval,  amin=1e-100, amax=1e100)
    
    return alpha_k, fc, gc, old_fval, old_old_fval, gfkp1

# Backtracking algorithm to find alpha satisfying sufficient decrease condition
def backtracking(func, jac, x_k, p_k, alpha_max=1, c1 = 1e-4, maxiter=25):
    
    # Define parameters
    rho = 0.9
    alpha = alpha_max; 
    k=0

    #Compute f, grad f at x_k
    f_k = func(x_k)
    df_k = jac(x_k)

    # Backtracking linesearch for computing step length
    while func(x_k + alpha*p_k) > ( f_k + c1 * alpha * np.dot(df_k,p_k) ) and k < maxiter:
        alpha = rho*alpha
        k+=1
    
    return alpha
        