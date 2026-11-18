import jax
import numpy as np
import jax.numpy as jnp


def f(x):
    return x[0]**2 + 2*x[1]**2

def df(x):
    return np.array([2*x[0], 4*x[1]])


grad =  jax.grad(f)

d = df(np.array([1.0, 1.0]))
d_AD = grad(jnp.array([1.0, 1.0]))

print(f"df = {d}, df_AD = {d_AD}, diferença = {d - d_AD}")