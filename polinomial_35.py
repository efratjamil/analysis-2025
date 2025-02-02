
import numpy as np

def gauss_jordan_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = len(A)
    augmented = np.hstack((A, b))
    for i in range(n):
        augmented[i] /= augmented[i, i]
        for j in range(n):
            if i != j:
                augmented[j] -= augmented[i] * augmented[j, i]
    return augmented[:, -1]

def solve_lu(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    P, L, U = np.linalg.lu(A)
    y = np.linalg.solve(L, np.dot(P, b))
    x = np.linalg.solve(U, y)
    return x

def solve_system(A, b):
    if np.linalg.det(A) != 0:
        return gauss_jordan_elimination(A, b)
    return solve_lu(A, b)

def polynomial_interpolation(points, x):
    A = np.vander([p[0] for p in points], increasing=True)
    b = np.array([p[1] for p in points])
    coeffs = solve_system(A, b)
    return np.polyval(coeffs[::-1], x)

if __name__ == "__main__":
    points = [(0.35, -213.5991), (0.4, -204.4416), (0.55, -194.9375), (0.65, -185.0256), (0.7, -174.6711), (0.85, -163.8656), (0.9, -152.6271)]
    x = 0.75
    result = polynomial_interpolation(points, x)
    print("Interpolated value at x=", x, " :", result)