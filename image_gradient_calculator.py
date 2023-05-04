import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_gradients_among_directions(image):
    # image preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.pad(image, (1, ), 'reflect')

    gradients = compute_gradient(image)

    # postprocessing
    gx = cv2.convertScaleAbs(gradients[0, 0, :, :])
    gy = cv2.convertScaleAbs(gradients[0, 1, :, :])

    return gx, gy


def compute_gradient(image):
    return F.conv2d(torch.Tensor(image[np.newaxis, np.newaxis, :, :]),
                    weight=get_sobel_kernel()).numpy()


def get_sobel_kernel():
    Kx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Ky = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    return torch.Tensor([[Kx], [Ky]])
