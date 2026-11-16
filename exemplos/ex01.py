
#%%=========== definição do problema ==================
from pec00079.plot import plot_CO2D
from scipy.optimize import minimize
import numpy as np


def f(x):
    return x[0]**3 + 2*x[0]*x[1]**2 - x[1]**3 - 20*x[0]

def df(x):
    return np.array([3*x[0]**2 + 2*x[1]**2 - 20, 4*x[0]*x[1] - 3*x[1]**2])

bounds = [(-4,4),(-4,4)]

x0 = np.array([1,1])



#%%=========== conjugate gradient ==================
print("\n Conjugate Gradient \n")

res_CG_diff = minimize(f, x0, method='CG',
                  options={'disp': True})   

res_CG_grad = minimize(f, x0, method='CG', jac=df,
                  options={'disp': True})

#%%===========BFGS ==================

print("\n BFGS \n")
res_BFGS_diff = minimize(f, x0, method='BFGS',
               options={'disp': True})

res_BFGS_grad = minimize(f, x0, method='BFGS', jac=df,
               options={'disp': True})

fig1, ax1 = plot_CO2D(f, bounds,x_opt = res_BFGS_grad.x)

fig1.show()


#%%============= Nelder-Mead ================
print("\n Nelder-Mead \n")

res_Nelder_Mead = minimize(f, x0, method='Nelder-Mead',
               options={'disp': True})

print(res_Nelder_Mead)