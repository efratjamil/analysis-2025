import numpy as np

def lu_decomposition(A):
    N = len(A)
    L = np.eye(N)
    U = A.astype(float)

    for i in range(N):
        for j in range(i + 1, N):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j] -= factor * U[i]

    return L, U

def forward_substitution(L, b):
    N = len(L)
    y = np.zeros(N)
    for i in range(N):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def backward_substitution(U, y):
    N = len(U)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def solve_lu(A, b):
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def integrate_function_manual():
    f = lambda x: (2*x*np.exp(-x) + np.log(2*x**2)) * (2*x**2 - 3*x - 5)
    a, b = 0.5, 1
    N = 1000
    dx = (b - a) / N
    integral = sum(f(a + i * dx) * dx for i in range(N))
    return integral

if __name__ == "__main__":
    integral_value = integrate_function_manual()
    print("Integral value over [0.5,1]:", integral_value)