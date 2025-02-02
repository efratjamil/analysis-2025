import numpy as np

def forward_elimination(A, b):
    N = len(A)
    for k in range(N):
        max_row = max(range(k, N), key=lambda i: abs(A[i][k]))
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, N):
            factor = A[i][k] / A[k][k]
            A[i] = [A[i][j] - factor * A[k][j] for j in range(N)]
            b[i] -= factor * b[k]
    return A, b

def backward_substitution(A, b):
    N = len(A)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, N))) / A[i][i]
    return x

def gaussian_elimination(A, b):
    A, b = forward_elimination(A, b)
    return backward_substitution(A, b)

def integrate_function_manual():
    f = lambda x: (2*x*np.exp(-x) + np.log(2*x**2)) * (2*x**2 - 3*x - 5)
    a, b = 0.5, 1
    N = 1000
    dx = (b - a) / N
    integral = sum(f(a + i * dx) * dx for i in range(N))
    return integral

if __name__ == "__main__":
    A = [[1, -1, 2, -1],
         [2, -2, 3, -3],
         [1, 1, 1, 0],
         [1, -1, 4, 3]]
    b = [-8, -20, -2, 4]
    solution = gaussian_elimination(A, b)
    print("Solution for the system:", solution)
    integral_value = integrate_function_manual()
    print("Integral value over [0.5,1]:", integral_value)