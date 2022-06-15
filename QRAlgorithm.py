import numpy as np

######## 1) The implementation of QR algorithm
def QRDecomposition(A, preserve=10):
    """QR Decomposition using Householder Transformation

    Args:
        A (numpy.narray): n*n matrix
        preserve (int, optional): preserve number after dot. Defaults to 10.

    Returns:
        Tuple(numpy.narray,numpy.narray): (Q,R) Q and R matrix in QR-decomposition
    """
    n = np.shape(A)[0]
    Q = np.identity(n)
    R = np.copy(A)
    for i in range(n - 1):
        # each time deal with the i-th row
        a = R[i:, i] 
        e = np.zeros_like(a) # e:(n*1)
        e[0] = 1
        alpha = np.linalg.norm(a, ord=2)
        if np.any(a - alpha * e):
            u = (a - alpha * e) / np.linalg.norm(a - alpha * e)
        else:
            u = a - alpha * e
        H = np.identity(n)
        H[i:, i:] -= 2.0 * np.outer(u, u)  # H =I-2uu^H
        R = np.dot(H, R)  # R=H_(n-1)*...*H_2*H_1*A
        Q = np.dot(Q, H)  # Q=H_(n-1)*...*H_2*H_1
    return np.around(Q, preserve), np.around(R, preserve)


def isNotCovergence(A, bound_val):
    """Decide whether A is Convergent with the boundary value bound_val

    Args:
        A (numpy.narray): The matrix that are calculated in the QR algorithm
        bound_val (float): the threshold which each element in A will be compared with

    Returns:
        bool: If A is not convergent, return True, else False
    """
    m = np.shape(A)[0]
    for i in range(m):
        for j in range(i):
            if A[i][j] > bound_val:
                return True
    return False


def QRAlgorithm(A, preserve=10, bound_val=1e-10):
    """QR Algorithm

    Args:
        A (numpy.narray): The matrix that are calculated in the QR algorithm
        preserve (int, optional):The preserve number in QR deomposition. Defaults to 10.
        bound_val (float, optional): the threshold in decide whether the algorithm coverges. Defaults to 1e-10.

    Returns:
        Tuple(numpy.narray,int): The convergent A and the iter time
    """
    times = 0
    while isNotCovergence(A, bound_val):
        Q, R = QRDecomposition(A, preserve)
        A = np.dot(R, Q)
        times += 1
    return A, times


######## 2) QR Algorithm Calculation
#### A_1
A_1 = np.array([[10, 7, 8, 7], [7, 5, 6, 5], [8, 6, 10, 9], [7, 5, 9, 10]], dtype=float)
print("A_1 is:")
print(A_1)
A_1_result, times = QRAlgorithm(A_1, 10, 1e-6)
print("It takes {0} times to finish QR Algorithm.".format(times))
print("The result of QR Algorithm is:")
print(np.around(A_1_result, 3))

#### A_2
print("=========")
A_2 = np.array(
    [
        [2, 3, 4, 5, 6],
        [4, 4, 5, 6, 7],
        [0, 3, 6, 7, 8],
        [0, 0, 2, 8, 9],
        [0, 0, 0, 1, 0],
    ],
    dtype=float,
)
print("A_2 is:")
print(A_2)
A_2_result, times = QRAlgorithm(A_2, 10, 1e-6)
print("It takes {0} times to finish QR Algorithm.".format(times))
print("The result of QR Algorithm is:")
print(np.around(A_2_result, 3))

#### A_3
print("=========")
A_3 = np.ones((6, 6))
for i in range(6):
    for j in range(6):
        A_3[i][j] = A_3[i][j] / (i + j + 1)

print("A_3 is:")
print(A_3)
A_3_result, times = QRAlgorithm(A_3, 10, 1e-6)
print("It takes {0} times to finish QR Algorithm.".format(times))
print("The result of QR Algorithm is:")
print(np.around(A_3_result, 3))

######## 3) eigenvalues and matrix 2-norm condition number


def calculateConditionNumber(A):
    """calculate 2-norm Condition Number

    Args:
        A (numpy.narray): The matrix to calculate condition number 

    Returns:
        float: the Condition Number of A
    """    
    A_reverse = np.linalg.inv(A)
    return np.linalg.norm(A, ord=2) * np.linalg.norm(A_reverse, ord=2)


print("=======================================================")
print("The Eigen Values of A_1 are", np.linalg.eig(A_1)[0])
print("The Condition Number of A_1 are", calculateConditionNumber(A_1))
print("=====")
print("The Eigen Values of A_2 are", np.linalg.eig(A_2)[0])
print("The Condition Number of A_2 are", calculateConditionNumber(A_2))
print("=====")
print("The Eigen Values of A_3 are", np.linalg.eig(A_3)[0])
print("The Condition Number of A_3 are", calculateConditionNumber(A_3))
