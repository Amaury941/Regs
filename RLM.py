import statsmodels.api as sm
import numpy as np

# média de tempo
y = [ 
384.6666667,
826.3333333,
1580.333333,
3137.666667,
6752.333333,
13970,
27719,
51468.33333
]

# Record Size
x = [ 
[
128,
256,
384,
512,
640,
768,
896,
1024]
]

# algoritmo de regressão (funciona com múltiplos x)
def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

print(reg_m(y, x).summary())