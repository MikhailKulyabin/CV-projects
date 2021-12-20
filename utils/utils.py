import numpy as np


def normalize_image(input):
    """
    Minimax normalization of an image to range [0,1]
    :param input: image in np array format
    :type input: np.ndarray
    :return: normalized output image
    :rtype: np.ndarray
    """
    output = (input - input.min()) / (input.max() - input.min())
    return output


def threshold_image(input, threshold):
    """
    Creates a binary mask by thresholding the gray image.
    :param input: image in np array format
    :type input: np.ndarray
    :param threshold: value between 0 and 1 to threshold the image
    :type threshold: float
    :return: normalized output image
    :rtype: np.ndarray
    """
    output = np.zeros_like(input)
    output[input>=threshold] = 1
    return output

