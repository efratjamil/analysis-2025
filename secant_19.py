import numpy as np


def secant_method(f, x0, x1, tol, max_iter=50):
    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "x0", "x1", "p"))
    for i in range(max_iter):
        if f(x1) - f(x0) == 0:
            print("Method cannot continue.")
            return

        p = x0 - f(x0) * ((x1 - x0) / (f(x1) - f(x0)))

        if abs(p - x1) < tol:
            return p

        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f}".format(i, x0, x1, p))
        x0, x1 = x1, p
    return p


if __name__ == "__main__":
    f = lambda x: (x * np.exp(-x ** 2 + 5 * x)) * (2 * x ** 2 - 3 * x - 5)
    x0, x1, tol, max_iter = 2, 2.5, 1e-6, 50
    root = secant_method(f, x0, x1, tol, max_iter)
    print(f"\033[94m\nThe equation f(x) has an approximate root at x = {root}\033[0m")