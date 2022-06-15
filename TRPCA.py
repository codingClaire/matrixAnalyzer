import os
import matplotlib.image as mpimg
import numpy as np
import imageio
import math
import re
from scipy.fft import fft, ifft
from scipy.linalg import svd
import matplotlib.pyplot as plt


def getM(path):
    """get M tensor

    Args:
        path (str): the RGB photos path

    Returns:
        Tuple(numpy.narray,int,int): Tensor M, Height, Width
    """
    photos = os.listdir(path)
    pattern = ".*\.jpg$"
    photos = [photo for photo in photos if re.match(pattern, photo)]
    photos.sort(key=lambda x: int(x.split(".")[0]))
    N = len(photos)
    H, W, _ = mpimg.imread(path + "/" + photos[0]).shape
    M = np.zeros((H * W, N, 3))
    for idx in range(len(photos)):
        I = mpimg.imread(path + "/" + photos[idx])
        M[:, idx, :] = I.reshape(H * W, 3)
    return M, H, W


def transpose(X):
    """The transpose of a tensor
    Args:
        X (np.array): Tensor

    Returns:
        np.array: X.T
    """
    n1, n2, n3 = X.shape
    Xt = np.zeros(n2, n1, n3)
    Xt[:, :, 0] = np.copy(X[:, :, 0].T)
    if n3 > 1:
        for i in range(1, n3):
            Xt[:, :, i] = X[:, :, n3 - i + 1]
    return Xt


def tprod(A, B):
    """Product of two tensors
    Args:
        A (np.array): Tensor
        B (np.array): Tensor
    Returns:
        np.array: C=A*B
    """
    n1, _, n3 = A.shape
    m2 = B.shape[2]
    A = fft(A)
    B = fft(B)
    C = np.zeros(n1, m2, n3)
    for i in range(n3):
        C[:, :, i] = A[:, :, i] * B[:, :, i]
    C = ifft(C)
    return C


def prox_tnn(rho, Y):
    """update E of TRPCA
     The proximal operator of the tensor nuclear norm of a 3 way tensor
        min_X rho*||X||_*+0.5*||X-Y||_F^2
    args:
        Y:n1*n2*n3 tensor

    Returns:
        X:n1*n2*n3 tensor
        tnn:tensor nuclear norm of X
        trank:tensor tubal rank of X
    """
    _, _, n3 = Y.shape
    X = np.zeros(Y.shape)
    Y = fft(Y)
    # first slice
    U, S, V = svd(Y[:, :, 0])
    S = np.diag(S)
    r = len(np.argwhere(np.abs(S) > rho))
    if r >= 1:
        for i in range(0, r):
            S[i][i] -= rho
    for i in range(1, math.floor(n3 / 2)):
        U, S, V = svd(Y[:, :, i])
        S = np.diag(S)
        r = len(np.argwhere(np.abs(S) > rho))
        if r >= 1:
            for i in range(0, r):
                S[i][i] -= rho
            X[:, :, 0] = U[:, 0:r].dot(S[0:r, 0:r]).dot(V[:, 0:r].transpose(1, 0))
        X[:, :, n3 + 2 - i] = X[:, :, i].conj()

    if n3 % 2 == 1:
        i = math.floor(n3 / 2)
        U, S, V = svd(Y[:, :, i])
        S = np.diag(S)
        r = len(np.argwhere(np.abs(S) > rho))
        if r >= 1:
            for i in range(0, r):
                S[i][i] -= rho
            X[:, :, 0] = U[:, 0:r].dot(S[0:r, 0:r]).dot(V[:, 0:r].transpose(1, 0))
    X = ifft(X)
    return X


def prox_l1(lam, b):
    """The proximal operator of the l1 norm

    Args:
        lam (float): parameter
        b (numpy.narray): the input for caculate l1 norm proximal operator

    Returns:
        numpy.narray: proximal operator of the l1 norm of b
    """
    x = np.maximum(0, b + lam) + np.minimum(0, b - lam)
    return x


def TRPCA(X, lam, epsi=1e-8, max_time=500, rho=1.1, miu=1e-4, max_miu=1e10):
    """The Algorithm for TRPCA

    Args:
        X (numpy.narray): _description_
        lam (float): _description_
        epsi (float, optional): parameter. Defaults to 1e-8.
        max_time (int, optional): parameter. Defaults to 500.
        rho (float, optional): parameter. Defaults to 1.1.
        miu (float, optional): parameter. Defaults to 1e-4.
        max_miu (float, optional): parameter. Defaults to 1e10.

    Returns:
       Tuple(numpy.narray,numpy.narray,int): The result of L,S and iter time
    """
    L = np.zeros(X.shape)
    S = np.zeros(X.shape)
    Y = np.zeros(X.shape)
    time = 0
    while time < max_time:
        Lk = L.copy()
        Sk = S.copy()
        L = prox_tnn(
            1 / miu, -S + X - Y / miu
        )  # Tensor Singular Value Thresholding (t-SVT)
        S = prox_l1(lam / miu, -L + X - Y / miu)
        dY = L + S - X
        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chgY = np.max(np.abs(dY))
        # finish iter
        print(
            "time:",
            time,
            "chgL:",
            chgL,
            "chgS:",
            chgL,
            "chgY:",
            chgY,
        )
        if chgL <= epsi and chgS <= epsi and chgY <= epsi:
            break
        Y = Y + miu * dY
        miu = np.minimum(rho * miu, max_miu)
        time += 1
    return L, S, time


def getTRPCACompressedImage(lam, default_lam=True):
    """main function to get compressed image with TRPCA

    Args:
        lam (float): parameter.
        default_lam (bool, optional):Decide whether to use default lambda value. Defaults to True.
    """
    path = "./dataset/"
    sub_folders = os.listdir(path)
    for sub_folder in sub_folders:
        if sub_folder == "0000045":
            subpath = path + sub_folder + "/RGB"
            M, H, W = getM(subpath)
            n1, n2, n3 = M.shape  # (H*W,N,3)
            if default_lam == True:
                lam = 1 / math.sqrt(max(n1, n2) * n3)
            L, S, time = TRPCA(M, lam, 1e-2)  # H*W,N,3
            recover_M = L + S
            save_path = subpath + "/lam=%s" % lam
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for idx in range(n2):
                # save l
                l = L[:, idx, :].reshape((H, W, 3))
                save_l_path = save_path + "/%s_L.jpg" % (idx + 1)
                imageio.imsave(save_l_path, l, "jpg")
                # save s
                s = S[:, idx, :].reshape((H, W, 3))
                save_s_path = save_path + "/%s_S.jpg" % (idx + 1)
                imageio.imsave(save_s_path, s, "jpg")
                # save s
                recover_m = recover_M[:, idx, :].reshape((H, W, 3))
                save_m_path = save_path + "/%s_re_M.jpg" % (idx + 1)
                imageio.imsave(save_m_path, recover_m, "jpg")
                # save multiple plt
                # m = M[:,idx,:].reshape((H, W,3))
                # recover_m = recover_M[:,idx,:].reshape((H,W,3))
                # plt.subplot(1, 4, 1)
                # plt.imshow(np.abs(m), cmap="gray")
                # plt.subplot(1, 4, 2)
                # plt.imshow(np.abs(recover_m), cmap="gray")
                # plt.subplot(1, 4, 3)
                # plt.imshow(np.abs(l), cmap="gray")
                # plt.subplot(1, 4, 4)
                # plt.imshow(np.abs(s), cmap="gray")
                # f = plt.gcf()
                # save_img_path = save_path + "/%s_whole.jpg" % (idx+1)
                # f.savefig(save_img_path)
                # f.clear()
            break


if __name__ == "__main__":
    lam = 0.01
    getTRPCACompressedImage(lam)
