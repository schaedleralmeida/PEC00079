import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime as fdiff

#%%
#Testes para x escalar:

def f1(x):
    return np.sin(x*3*np.pi)*(x**3 - 2*x**2 + 3*x - 4)

def df1(x):
    return 3*np.pi*np.cos(x*3*np.pi)*(x**3 - 2*x**2 + 3*x - 4) + np.sin(x*3*np.pi)*(3*x**2 - 4*x + 3)

def dff_escalar(f,x,h):
    "diferença finita progressiva (frente)"
    return ( f(x+h) - f(x) ) / h

def dfc_escalar(f,x,h):
    "diferença finita central"
    return ( f(x+h) - f(x-h) ) / (2*h)

def dpc_escalar(f,x,h):
    "diferença finita com passo complexo"
    return np.imag(f(x + 1j*h))/h

#criando os gráficos
x = np.linspace(0,3,100)
y = f1(x)
dy = df1(x)
plt.figure()
plt.plot(x, y, label="f1(x)")
plt.plot(x, np.sin(x*3*np.pi), label="sin(3πx)")
plt.plot(x, x**3 - 2*x**2 + 3*x - 4, label="x³ - 2x² + 3x - 4")
plt.title("Função f1(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
plt.figure()
plt.plot(x,dy)
plt.title("Derivada de f1(x)")
plt.xlabel("x")
plt.ylabel("df1(x)")
plt.grid()
plt.show()

#calculando as derivadas numéricas
x0 = 1.5
h = np.linspace(1e-6, 1e-12, 100)
df_anl = df1(x0)
df_fdf = np.array([dff_escalar(f1, x0, eps) for eps in h])
df_fdc = np.array([dfc_escalar(f1, x0, eps) for eps in h])
df_cpx = np.array([dpc_escalar(f1, x0, eps) for eps in h])


plt.figure()
plt.plot(h, np.abs(df_fdf - df_anl)/np.abs(df_anl), label="Diferença progressiva")
plt.plot(h, np.abs(df_fdc - df_anl)/np.abs(df_anl), label="Diferença central")
plt.plot(h, np.abs(df_cpx - df_anl)/np.abs(df_anl), label="Passo complexo")
plt.xlabel("x")
plt.ylabel("erro relativo")
plt.grid()
plt.legend()
plt.show()

#%%
def f(x):
    return - (- 2*x[0]**2 + x[1]**2 )

def df(x):
    return np.array([4*x[0], -2*x[1]])

def g1(x):
    return -( -x[0] + x[1]**2 + 2)

def dg1(x):
    return np.array([-(-1), -(-2*x[1])])

def g2(x):
    return -( x[0] + x[1] - 4 )

def dg2(x):
    return np.array([-1, -1])

def h1(x):
    return -( x[0] + 3*x[1] )


x = np.array([0,0])
h = np.linspace(1e-6, 1e-12, 100)

df_anl = df(x)
direcoes = np.eye(len(x))
df_fdf = np.array([[fdf(lambda t: f(x + t*d), 0, eps) for d in direcoes] for eps in h])
df_fdc = np.array([[fdc(lambda t: f(x + t*d), 0, eps) for d in direcoes] for eps in h])
df_cpx = np.array([[fcpx(lambda t: f(x + t*d), 0, eps) for d in direcoes] for eps in h])

erro_fdf = np.linalg.norm(df_fdf - df_anl, axis=1)
erro_fdc = np.linalg.norm(df_fdc - df_anl, axis=1)
erro_cpx = np.linalg.norm(df_cpx - df_anl, axis=1)

plt.semilogx(h, erro_fdf, label="Diferença progressiva")
plt.semilogx(h, erro_fdc, label="Diferença central")
plt.semilogx(h, erro_cpx, label="Passo complexo")
plt.xlabel("Passo h")
plt.ylabel("Erro absoluto (norma Euclidiana)")
plt.title("Erro das derivadas em relação à referência analítica")
plt.grid(True)
plt.legend()
plt.show()


