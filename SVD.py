import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import imageio
import re

####### 1) load the files into matrices
def getM(path):
    """get M matrix in path

    Args:
        path (str): the path that contains multiple photos

    Returns:
        Tuple(numpy.narray,int,int): M matrix, Height of photo, Width of photo
    """
    photos = os.listdir(path)
    pattern = ".*\.jpg$"
    photos = [photo for photo in photos if re.match(pattern, photo)]
    photos.sort(key=lambda x: int(x.split(".")[0]))
    N = len(photos)
    H, W = mpimg.imread(path + "/" + photos[0]).astype(np.uint8).shape
    M = np.zeros((H * W, N))
    for idx in range(len(photos)):
        I = mpimg.imread(path + "/" + photos[idx]).astype(np.uint8)
        M[:, idx] = I.reshape(H * W)
    return M, H, W


def getCompressedImage(d, folder_name="0000045"):
    """the main function that access the compressed images in folder_name
    with SVD at d value equals to d

    Args:
        d (int): an empirical parameter
        folder_name (str, optional): _description_. Defaults to "0000045".
    """
    path = "./dataset/"
    sub_folders = os.listdir(path)
    for sub_folder in sub_folders:
        if sub_folder == folder_name:
            subpath = path + sub_folder
            M, H, W = getM(subpath)
            ##### 2) SVD method to factorize the matrix
            K, N = M.shape
            if d > N:
                break
            U, Sigma, I = svd(M)  # SVD
            Sigma_NK = np.vstack((np.diag(Sigma), np.zeros((K - N, N))))
            new_M = U[:, :d].dot(Sigma_NK[:d, :d]).dot(I[:d, :])  # recover M

            save_path = subpath + "/%s" % d
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # print 3 value
            print("U (%d,%d):" % U[:, :d].shape)
            print(U[:, :d])
            print("I (%d,%d):" % I[:d, :].shape)
            print(I[:d, :])
            print("single value (%d,%d):" % Sigma_NK[:d, :d].shape)
            for idx in range(d):
                print(Sigma_NK[idx][idx], end=" ")
            print("\n")
            # save 3 value
            np.save(save_path + "/u_value", U[:, :d])
            np.save(save_path + "/sigma_value", Sigma_NK[:d, :d])
            np.save(save_path + "/v_value", I[:d, :])
            # save reconstruct result
            for idx in range(new_M.shape[1]):
                new_face = new_M[:, idx].reshape((H, W))
                save_img_path = save_path + "/%s.jpg" % idx
                imageio.imsave(save_img_path, new_face, "jpg")

            # save multiple plot  M new_M U V
            for idx in range(N):
                m = M[:,idx].reshape((H, W))
                recover_m = new_M[:,idx].reshape((H, W))
                plt.subplot(1, 2, 1)
                plt.imshow(m, cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(recover_m, cmap="gray")
                f = plt.gcf()
                save_img_path = save_path + "/%s_whole.jpg" % idx
                f.savefig(save_img_path)
                f.clear()
            break

if __name__ == '__main__':
    d_list = [8, 16, 32, 64]
    for d in d_list:
        print("========%d=======" % d)
        getCompressedImage(d)
