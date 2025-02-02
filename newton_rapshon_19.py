import numpy as np


def newton_raphson(f, df, p0, tol, max_iter=50):
    print("{:<10} {:<15} {:<15}".format("Iteration", "p0", "p1"))
    for i in range(max_iter):
        if df(p0) == 0:
            print("Derivative is zero at p0, method cannot continue.")
            return None

        p = p0 - f(p0) / df(p0)

        if abs(p - p0) < tol:
            return p

        print("{:<10} {:<15.9f} {:<15.9f}".format(i, p0, p))
        p0 = p
    return p


if __name__ == "__main__":
    f = lambda x: (x * np.exp(-x ** 2 + 5 * x)) * (2 * x ** 2 - 3 * x - 5)
    df = lambda x: np.exp(-x ** 2 + 5 * x) * (2 * x ** 2 - 3 * x - 5) + x * np.exp(-x ** 2 + 5 * x) * (4 * x - 3)
    p0, tol, max_iter = 2.5, 1e-6, 50
    root = newton_raphson(f, df, p0, tol, max_iter)
    print(f"\033[94m\nThe equation f(x) has an approximate root at x = {root:<15.9f}\033[0m")