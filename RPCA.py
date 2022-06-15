import os
import numpy as np
from scipy.linalg import svd
import imageio
import math
from SVD import getM
import matplotlib.pyplot as plt

############# 3) RPCA decomposition
def sgn(x):
    """sgn function

    Args:
        x (float): the input number

    Returns:
        int: return -1,0 or 1 that represent the sign of the input number
    """
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1


def S_function(delta, X):
    """S_function is the shrinkage operator
    According to the formula:
    S_{\delta}(x) = sgn(x)max(|x|-\delta,0)

    Args:
        delta (float): delta value
        X (numpy.narray): the input matrix

    Returns:
        numpy.narray: return the matrix after shrinkage operation
    """
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = sgn(X[i][j]) * max(abs(X[i][j]) - delta, 0)
    return X


def D_function(delta, X, d=8):
    """D_function is the singular value thresholding operator
    According to the formula:
    D_{\delta}(X) = U \times S_{\delta}(Sigma) \times V*
    X =  U \times Sigma \times V*

    Args:
        delta (float): the
        X (numpy.narray): the input matrix
        d (int, optional): d value for SVD. Defaults to 8.

    Returns:
        numpy.narray: return the matrix after singular value thresholding operation
    """
    U, Sigma, I = svd(X)  # U (K,K) Sigma(N,) I(N,N)
    K, N = U.shape[0], I.shape[1]
    Sigma = np.vstack((np.diag(Sigma), np.zeros((K - N, N))))  # Sigma (K,N)
    Sigma = S_function(delta, Sigma)
    return U[:, :d].dot(Sigma[:d, :d]).dot(I[:d, :])


def isConverge(M, L, S, M_F, threshold):
    """Decide whether to stop the while loop in main algorithm

    Args:
        M (numpy.narray): M matrix
        L (numpy.narray): L matrix
        S (numpy.narray): S matrix
        M_F (float): the 2-norm of origin matrix M
        threshold (float): the threshold for deciding convergence

    Returns:
        Bool: If convergent return True, else return False
    """
    f1 = np.linalg.norm(M - L - S, ord="fro")
    print("f1:", f1, "threshold*f2:", threshold * M_F)
    if f1 <= threshold * M_F:
        return True
    return False


def getRPCACompressedImage(
    miu, lam, folder_name="0000045", default_miu=True, default_lam=True
):
    """The main algorithm of RPCA

    Args:
        miu (float): miu parameter in algorithm
        lam (float): lambda parameter in algorithm
        folder_name (str, optional): The choosen folder.Defaults to "0000045".
        default_miu (bool, optional): The bool value to decide whether use default miu. Defaults to True.
        default_lam (bool, optional): The bool value to decide whether use default lambda. Defaults to True.
    """
    path = "./dataset/"
    sub_folders = os.listdir(path)
    for sub_folder in sub_folders:
        if sub_folder == folder_name:
            subpath = path + sub_folder
            M, H, W = getM(subpath)
            #### 3) RPCA decomposition
            K, N = M.shape
            # use default miu and lambda
            if default_miu == True:
                M_m1_norm = 0
                for i in M.flat:
                    M_m1_norm += abs(i)
                miu = K * N / (4 * M_m1_norm)
            if default_lam == True:
                lam = 1 / math.sqrt(0.1 * max(N, K))
            # RPCA algorithm
            S, Y = np.zeros((K, N)), np.zeros((K, N))
            L = np.zeros((K, N))
            threshold = 1e-7
            times = 0
            M_F = np.linalg.norm(M, ord="fro")
            while isConverge(M, L, S, M_F, threshold) == False:
                print(times)
                L = D_function(math.pow(miu, -1), (M - S + (math.pow(miu, -1) * Y)), N)
                S = S_function(
                    math.pow(miu, -1) * lam, (M - L + (math.pow(miu, -1) * Y))
                )
                Y = Y + miu * (M - L - S)
                times += 1
            save_path = subpath + "/%s_%s" % (miu, lam)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # save L
            for idx in range(N):
                l = L[:, idx].reshape((H, W))
                save_img_path = save_path + "/%s_L.jpg" % idx
                imageio.imsave(save_img_path, l, "jpg")
            # save S
            for idx in range(N):
                s = S[:, idx].reshape((H, W))
                save_img_path = save_path + "/%s_S.jpg" % idx
                imageio.imsave(save_img_path, s, "jpg")
            # save multiple plot: M recover_M L S
            recover_M = L + S
            for idx in range(N):
                m = M[:, idx].reshape((H, W))
                recover_m = recover_M[:, idx].reshape((H, W))
                l = L[:, idx].reshape((H, W))
                s = S[:, idx].reshape((H, W))
                plt.subplot(1, 4, 1)
                plt.imshow(m, cmap="gray")
                plt.subplot(1, 4, 2)
                plt.imshow(recover_m, cmap="gray")
                plt.subplot(1, 4, 3)
                plt.imshow(l, cmap="gray")
                plt.subplot(1, 4, 4)
                plt.imshow(s, cmap="gray")
                f = plt.gcf()
                save_img_path = save_path + "/%s_whole.jpg" % idx
                f.savefig(save_img_path)
                f.clear()
            print("cost times:", times)


if __name__ == "__main__":
    miu, lam = 0.01, 0.01
    getRPCACompressedImage(miu, lam)


"""
def getOptimizationTarget(L, lam, S):
    decide whether to stop the while loop in main algorithm

    Args:
        L (numpy.narray): L matrix
        lam (float): lambda value 
        S (numpy.narray): S matrix

    Returns:
        _type_: _description_
       
    nuclear_norm = np.linalg.norm(L, "nuc")
    m1_norm = 0
    for i in S.flat:
        m1_norm += abs(i)
    return nuclear_norm + lam * m1_norm
"""
