import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Import data
fname = 'returns.csv'
DF = pd.read_csv(fname)
N = 10  # Number of stocks to use
prices = DF.loc[:,DF.columns.str.endswith('returns')==False]
prices = prices.loc[:,prices.columns.str.endswith('date')==False]
prices = np.array(prices.iloc[:,:N])

DF = DF.loc[:,DF.columns.str.endswith('returns')]
DF = DF.iloc[:,:N]

# Required data/parameters
Mus = np.array(DF.mean(axis=0)) # Get np vector for average daily log-returns per stock

# Choose equal initial allocation
x0 = np.array([1/N for n in range(N)])

######################################################
# Define Objective, Gradient, Hessian with Penalty Function Method


def objective(W):
    Q = DF.cov()                                  # Covariance matrix
    var = np.matmul(W.transpose(),np.matmul(Q,W)) # Variance vector

    # Penalty Function method
    penalty1 =  (np.sum(W)-1)**2                    # Large for sum(x) <> 1
    penalty2 =  100*(R_min - np.matmul(Mus.transpose(), W))**2  # Large for returns <> minR
    return var + penalty1 + penalty2

def der_objective(W):
    Q = DF.cov()

    der = 2 * np.matmul(Q,W)

    penalty1_der = np.array([2*np.sum(W)-2 for i,Wi in enumerate(W)])
    penalty2_der = 100*np.array([ 2*Mus[i] * (np.matmul(Mus.transpose(),W) - R_min) for i,Wi in enumerate(W)])

    return der + penalty1_der + penalty2_der

def hess_objective(W):
    Q = DF.cov()
    hess = 2 * Q.values

    # Assemble Hessian terms for penalty1 function
    penalty1_hess = 2*np.ones_like(hess)

    # Assemble Hessian terms for penalty2 function
    bcast = np.broadcast(Mus.reshape(len(Mus),1),Mus.reshape(1,len(Mus)))
    penalty2_hess = np.empty(bcast.shape)
    penalty2_hess.flat = 100*np.array([2*a*b for (a,b) in bcast])
    
    return hess + penalty1_hess + penalty2_hess

# Iterative Line-Search Optimization Routines
from optimise import BFGS, SGD, Newton

# Awaiting results for plotting
my_means=[]
my_vars=[]


######################################################
# Minimise Portfolio Variance at expected risk level


for R_min in np.linspace(-0.001,0.004,25): # R_min now in global scope
    
    my_result = Newton(x0, objective, der_objective, hess_objective, tol = 1e-6, maxiter=1e4)
    bfgs_result = BFGS(x0, objective, der_objective, hess_objective, tol = 1e-6, maxiter=1e4)
    
    # Console Output
    print('\n\n Return: ',R_min)
    print('Newton_iters: ', my_result.k)
    print('BFGS_iters: ', bfgs_result.k)
    print('Portfolio Weight:', np.sum(my_result.x))
    
    # Store results for plotting (variance minus penalty terms)
    my_means.append(np.dot(Mus,my_result.x)) 
    my_vars.append(my_result.fun - (np.sum(my_result.x)-1)**2 - 100*(R_min - np.matmul(Mus.transpose(), my_result.x))**2 )


######################################################
# Compare Against Monte Carlo Simulation

def rand_weights(n):
    # Produces n random weights that sum to 1
    W = np.random.rand(n)
    W = np.asmatrix(W/ np.sum(W)).reshape(len(Mus),1)
    return W

def return_portfolio(W):
    Q = np.asmatrix(DF.cov())       # Covariance matrix

    R = Mus.transpose() * W         # Expected Return of the portfolio
    Var = W.transpose() * Q * W     # Expected Variance of the portfolio

    return R, Var

n_portfolios = 1000 #* N

mc_means = np.matrix(np.empty(1))
mc_vars = np.matrix(np.empty(1))

for portfolio in range(n_portfolios):
    W = rand_weights(len(Mus))
    m, s = return_portfolio(W)
    mc_means = np.vstack((mc_means,m))
    mc_vars = np.vstack((mc_vars, s))


######################################################
# Calculate Global Optimum Portfolio by using KKT to maximize Sharpe Ratio

r_f = -0.005 # Risk-free rate
Q = DF.cov().values


Mus.resize(N,1)

# KKT Stationarity in system Ax=b 
one = np.ones(1).reshape(1,1)
ones = np.ones_like(Mus)
zeros = np.zeros_like(Mus)
zero = np.zeros(1).reshape(1,1)

A = np.vstack((np.hstack((    2*Q,    -ones,        -(Mus-r_f), zeros  )),
               np.hstack((-ones.T,   zero,  zero, one )),
               np.hstack((-(Mus-r_f).T,   zero,  zero, zero )),
               np.hstack((zeros.T,   one,  zero, zero ))
              ))
b = np.zeros((N+3,1))
b[N+1]=1

# Solve for x 
res = np.linalg.solve(A,b) 
kappa = res[N+2]
x = res[:N] / kappa

# Expected Return, Variance
opt_R= np.dot(Mus.T,x)[0]
opt_var = np.dot(x.T,np.dot(Q,x))[0]

######################################################
# Plot Solutions on Mean-Variance Axis

import matplotlib.pyplot as plt
plt.style.use('seaborn')

# Plot Simulation
plt.plot(my_vars,my_means,'r',linewidth =0.8) # Use R_min values since my_means are incorrect
plt.xlabel('Variance')
plt.ylabel('Daily Mean Log Return')
plt.title(str(n_portfolios) + ' Portfolios, each containing ' + str(N) 
                            + ' stocks, with weights allocated randomly')

# Plot Efficient Frontier
plt.scatter([mc_vars[2:]], [mc_means[2:]], 1.5)
plt.scatter([opt_var],[opt_R],15, c='k')

plt.legend(['Monte-Carlo Simulation','Efficient Frontier (Newton Method)', 'Globally Optimal Portfolio'])

plt.xlim(xmin=0)
plt.show()