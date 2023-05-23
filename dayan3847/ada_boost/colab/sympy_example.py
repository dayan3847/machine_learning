import numpy as np
import sympy as sp

if __name__ == '__main__':

    # declare x and y
    a = np.array(['1'])

    weak_classifier_array = []
    print(a)
    for v in sp.symbols('x y'):
        for s in [-1, 1]:
            w: sp.Expr = weak_classifier_array.append(s * v / sp.Abs(v))
