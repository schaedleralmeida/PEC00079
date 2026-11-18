import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime as fdiff

#%%
#Testes para x escalar:

def dff_escalar(f,x,h):
    "diferença finita progressiva (frente)"
    return ( f(x+h) - f(x) ) / h

def dfc_escalar(f,x,h):
    "diferença finita central"
    return ( f(x+h) - f(x-h) ) / (2*h)

def dpc_escalar(f,x,h):
    "diferença finita com passo complexo"
    return np.imag(f(x + 1j*h))/h

#calculando as derivadas numéricas
x0 = 2.0e0

def compara_derivadas(f, df, x0, n=[1,20]):
    """ 
    Compara a derivada analítica df da função f com as aproximações numéricas por diferenças finitas progressivas, central e passo complexo.
    """
    h = np.array([10**(-a) for a in range(n[0],n[1])])
    df_anl = df(x0)
    df_dff = np.array([dff_escalar(f, x0, eps) for eps in h])
    df_dfc = np.array([dfc_escalar(f, x0, eps) for eps in h])
    df_dpc = np.array([dpc_escalar(f, x0, eps) for eps in h])

    err_dff = np.abs(df_dff - df_anl)/np.abs(df_anl)
    err_dfc = np.abs(df_dfc - df_anl)/np.abs(df_anl)
    err_dpc = np.abs(df_dpc - df_anl)/np.abs(df_anl)

    print(
        f"{'h':>12} | {'df analítica':>14} | {'DFF':>14} | {'DFC':>14} | "
        f"{'DPC':>14} | {'Erro DFF':>14} | {'Erro DFC':>14} | {'Erro DPC':>14}"
    )
    print("-" * 132)
    for i in range(len(h)):
        print(
            f"{h[i]:12.4e} | {df_anl:14.4e} | {df_dff[i]:14.4e} | "
            f"{df_dfc[i]:14.4e} | {df_dpc[i]:14.4e} | "
            f"{err_dff[i]:14.4e} | {err_dfc[i]:14.4e} | {err_dpc[i]:14.4e}"
        )


    fig,  ax = plt.subplots()
    plt.semilogx(h, np.abs(df_dff - df_anl)/np.abs(df_anl), label="Diferença progressiva")
    ax.semilogx(h, np.abs(df_dfc - df_anl)/np.abs(df_anl), label="Diferença central")
    ax.semilogx(h, np.abs(df_dpc - df_anl)/np.abs(df_anl), label="Passo complexo")
    ax.set_xlabel("x") 
    ax.set_ylabel("erro relativo")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"x0 = {x0}")
    ax.invert_xaxis()
    ax.grid()
    ax.legend()



#teste 1
#%%
def f1(x):
    return np.sin(x*3*np.pi)*(x**3 - 2*x**2 + 3*x - 4)

def df1(x):
    return 3*np.pi*np.cos(x*3*np.pi)*(x**3 - 2*x**2 + 3*x - 4) + np.sin(x*3*np.pi)*(3*x**2 - 4*x + 3)

compara_derivadas(f1, df1, x0=2.0e0, n=[1,20])


#teste 2
#%%
#Nesse exemplo a função é contínua mas a derivada não é contínua em x=1, o que pode gerar problemas na aproximação numérica da derivada nesse ponto.

#(ii)  
def f2(x):
    if np.real(x) < 1:
#--------------------------------------------
    #if x < 1: #com essa linha, o código não funciona com passo complexo, pois o a comparação não está definida para números complexos.
#-------------------------------------------
        return x**5
    else:
        return -2*x**2 + 3*x - 2
    
def df2(x):
    if x < 1:
        return 5*x**4
    else:
        return -4*x + 3

#observe que o resultado por diferenças centrais é exato (a menos da precisão numérica) para polinômios de grau 3 ou menor. O mesmo vale para o passo complexo.
for x0 in [.1, 1, 2.0]:
    print(f"\n Em x0 ={x0} a função f2(x0)={f2(x0)} e sua derivada df2(x0) = {df2(x0)}")
    compara_derivadas(f2, df2, x0=x0, n=[1,20])



# %%
