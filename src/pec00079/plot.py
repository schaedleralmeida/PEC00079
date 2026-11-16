#Módulo para criação de gráficos de problemas de otimização estrutural 2D.

import numpy as np
from matplotlib import pyplot as plt


def plot_CO2D(f, bounds, constraints=[], args=(), x_opt=None, npts=(201,201) , ineq='geq',labels=('x1','x2')):
    '''
    Plot 2D constrained optimization problems.
    The parameters f, bounds, constraints and args are the same for function scipy.optimize.minimize
    '''
    
    x1 = np.linspace(bounds[0][0], bounds[0][1], npts[0])
    x2 = np.linspace(bounds[1][0], bounds[1][1], npts[1])
    
    mg = np.array( np.meshgrid( x1, x2 ))

    fig, ax = plt.subplots(figsize=(12,10))

    fel = list()
    if args==():
        fel.append( ax.contour(x1, x2, f(mg), levels=50, cmap='winter' ) )
    else:
        fel.append( ax.contour(x1, x2, f(mg,*args), levels=50, cmap='winter' ) )

    plt.colorbar(mappable=fel[0])#, cax=ax)
    
    
    for constr in constraints:
        mg_constr = constr['fun'](mg)

        if constr['type'] == 'ineq':
            if ineq == 'geq':
                ineq_levels = [mg_constr.min(), 0.0]
            elif ineq =='leq':
                ineq_levels = [0.0, mg_constr.max()]
            else:
                raise ValueError("The parameter ineq must be 'leq' or 'geq' ")
            
            fel.append( ax.contour(x1,x2, mg_constr , levels=[0.0] , colors='r',linewidths=3 )  ) 
            fel.append( ax.contourf(x1,x2, mg_constr, levels=ineq_levels, colors= '#ffd8d7ff'))
            #CS.append( ax.contourf(x1,x2, mg_constr, levels=ineq_levels, colors='grey', alpha=0.7))# '#ffd8d7ff'))
        elif constr['type'] == 'eq':
            fel.append( ax.contour(x1,x2, mg_constr , levels=[0.0] , colors='black',linewidths=3 )  ) 

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid()
    if x_opt is not None:
        fel.append( ax.scatter([x_opt[0]], [x_opt[1]], s=100.0 , c='k' ) )

    return fig, ax