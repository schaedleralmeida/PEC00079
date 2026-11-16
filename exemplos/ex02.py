from pec00079.plot import plot_CO2D
from scipy.optimize import minimize
import numpy as np


def f(x):
    return - (- 2*x[0]**2 + x[1]**2 )

def g1(x):
    return -( -x[0] + x[1]**2 + 2)

def g2(x):
    return -( x[0] + x[1] - 4 )

def h1(x):
    return -( x[0] + 3*x[1] )


x0 = np.array([0,0])

constraints = [
    { 'type':'ineq', 'fun':g1},
    { 'type':'ineq', 'fun':g2}#,
    #{ 'type':'eq'  , 'fun':h1}
    ]

bounds = [(-0,10),(-10,10)]

res = minimize(f, x0, method= 'SLSQP', bounds= bounds, constraints= constraints)

print(res)

fig1, ax1 = plot_CO2D(f, bounds, constraints, x_opt = res.x)

fig1.show()
