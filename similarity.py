import numpy as np
from numpy import dot
from numpy.linalg import norm
import cv2

# def angular_similarity(p, q): #Angular similarity between 2 RBG numpy arrays
#     return np.arccos((p*q).sum(axis=-1) / (norm(p, axis=-1)*norm(q, axis=-1))) #smaller the better

def euclidean_similarity(p, q):
    return norm(p-q, axis = -1) #smaller the better

def threshold(x, t):
    return np.sum((x>t) + 0.0)

# def nhb_angular_similarity(np, nq, t):
#     # np is a np array of size (k, k, 3)
#     return threshold(angular_similarity(np, nq), t)  #smaller the better


def nhb_euclidean_similarity(np, nq, t):
    # np is a np array of size (k, k, 3)
    return threshold(euclidean_similarity(np, nq), t)  #smaller the better


def get_neighbour(i, j, img):
    return img[max(0, i-KERNEL): i+KERNEL, max(0, j-KERNEL): j+KERNEL, :]

def pixel_is_moving_object(i, j, imgs, t, euclidean_thres, kernel_thres):
    neighbourhood = get_neighbour(i, j, imgs[t])
    differences = []
    for l in range(len(imgs)):
        if l!=t:
            differences.append(nhb_euclidean_similarity(neighbourhood, get_neighbour(i, j, imgs[l]), euclidean_thres))
    return threshold(np.array(differences),kernel_thres)

if __name__ == '__main__':

    img1 = np.array(cv2.imread("1.png", cv2.IMREAD_COLOR))

    img2 = np.array(cv2.imread("2.png", cv2.IMREAD_COLOR))

    KERNEL = 2 #2 pixels to each direction

    H, W, C= img1.shape

    for i in range(H):
        for j in range(W):
            if pixel_is_moving_object(i, j, [img1, img2], 0, 1, 1):
                print("pixel is moving object", i, j)