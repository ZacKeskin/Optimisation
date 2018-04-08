## Introduction
This Repo contains a small working example of portfolio optimisation through exploting numerical optimisation algorithms. This was implemented as part of UCL's COMPGV19 course, in 2018.

### Algorithms
optimise.py introduces implementations of Steepest Gradient Descent, Newton's Method and BFGS algorithms, largely following the scipy.optimize.minimize syntax (although of the three, only BFGS is available through SciPy). 

linesearch.py offers a simple way to switch different procedures for finding appropriate step lengths. Default is backtracking line search satisfying Armijo condition.

### Examples
Run main.py to perform the full analyis involving:
- Quadratic Minimisation with penalty terms to trace the efficient frontier
- Monte Carlo analysis of feasible portfolios
- KKT formulation and solution for globally-optimal portfolio (maximising Sharpe Ratio)

Run Jupyter notebook to see interactive version of above, plus trajectory analysis and function surface plot comparision of constrained and unconstrained penalty optimisation.