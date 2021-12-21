import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image


def energy_image(image, mask, mean=None, std=None):
    """
    This function computes the energy image using the laplacian kernel.
    :param std: std width in pixels of the path
    :param mean: Mean column of the path
    :param mask: Mask with high value regions to be preserved
    :param image: input image
    :return: updated energy map
    """
    cols_idx = None
    if mean is None and std is None:
        cropped = image[:, :, 0:3]
    else:
        first_col = 0 if int(np.round(mean) - 2*np.ceil(std) - 1) < 0 else int(np.round(mean) - 2*np.ceil(std) - 1)
        last_col = image.shape[1] if int(np.round(mean) + 2*np.ceil(std) + 1) > image.shape[1] \
            else int(np.round(mean) + 2*np.ceil(std) + 1)
        cols_idx = (first_col, last_col)
        cropped = image[:, cols_idx[0]:cols_idx[1], 0:3]

    laplacian = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
    filter = np.stack([laplacian]*3, axis=2)
    cropped = cropped.astype('float32')
    convolved = np.absolute(scipy.signal.convolve(cropped, filter, mode='same', method='auto'))
    energy_map = convolved.sum(axis=2)  # Sum channels

    out_map = image[:,:,3]
    if mean is None and std is None:
        out_map = energy_map + mask
    else: # Only add the mask in the processed region, if not the values will accumulate to be very large
        out_map[:, cols_idx[0]:cols_idx[1]] = energy_map + mask[:, cols_idx[0]:cols_idx[1]]

    return out_map


def find_row_index(image):
    """
    Get the index with the lowest row sum.
    :param image: input image
    :return: index of row
    """
    rows = np.sum(image, axis=0)
    row_index = np.argmin(rows)
    return row_index


def find_column_index(image):
    """
    Get the index with the lowest column sum.
    :param image: input image
    :return: index of column
    """
    columns = np.sum(image, axis=1)
    column_index = np.argmin(columns)
    return column_index


def cum_cost(matrix, mean=None, std=None):
    """
    Computes the cummulative sum of the energy image. If mean and std are done, the update is only done in the
    desired columns (mean - std*2 - 1, mean + std*2 + 1).
    The cummulative cost is computed over the cost matrix (we update its values iteratively).
    To speed up the computation, we substitute the areas of the cost image that are not going to be updated (based on
    the mean and std), and we substitute them by the cumulative cost image of the previous iteration.
    :param matrix: imput matrix of 6 channels + image
    :param mean: mean column
    :param std: std width of path
    :return: updated accumulated cost image
    """
    cost_matrix = matrix[:,:,0]
    cum_cost_matrix = matrix[:,:,1]

    # Get the columns that need to be updated
    cols_idx = [0, matrix.shape[1]]
    if mean is not None and std is not None:
        first_col = 0 if int(np.round(mean) - 2 * np.ceil(std) - 1) < 0 else int(np.round(mean) - 2 * np.ceil(std) - 1)
        last_col = matrix.shape[1] if int(np.round(mean) + 2 * np.ceil(std) + 1) > matrix.shape[1] \
            else int(np.round(mean) + 2 * np.ceil(std) + 1)
        cols_idx = [first_col, last_col]

    # Substitute parts of the cost_matrix that are not going to be updated by the cumulative cost image
    cost_matrix[:, :cols_idx[0]] = cum_cost_matrix[:, :cols_idx[0]]
    cost_matrix[:, cols_idx[1]:] = cum_cost_matrix[:, cols_idx[1]:]

    # Compute cumulative cost
    for i in range(1, matrix.shape[0]):
        for j in range(cols_idx[0], cols_idx[1]):
            if 0 < j < matrix.shape[1] - 1:
                cost_matrix[i][j] += np.min(cost_matrix[i - 1][j-1:j+2])
            elif j == 0:  # If we are in the leftmost column
                cost_matrix[i][j] += np.min(cost_matrix[i - 1][j:j+2])
            else:  # If we are in the rightmost column
                cost_matrix[i][j] += np.min(cost_matrix[i - 1][j-1:j+1])

    return cost_matrix


def path(matrix):
    """
    Computes the minimum cost path over the cumulative cost image.
    :param matrix: cumulative cost image.
    :return: list of path
    """
    i = matrix.shape[0] - 1
    path = []
    index = np.argmin(matrix[i])
    path.append(index)
    while i > 0:
        if index == 0:  # leftmost column
            index = index + np.argmin(matrix[i - 1][index:index+2])
            path.append(index)
            i = i - 1
        elif index == matrix.shape[1] - 1:  # rightmost column
            index = index + np.argmin(matrix[i - 1][index-1:index+1]) - 1
            path.append(index)
            i = i - 1
        else:
            index = index + np.argmin(matrix[i - 1][index-1:index+2]) - 1
            path.append(index)
            i = i - 1
    return path[::-1]


def get_seam_mask(path, image_shape):
    """
    This function creates a mask out of a seam path.
    :param path: list of column indexes for each row of the image
    :param image_shape: shape of the image used to compute the seam path
    :return: mask of the seam
    """
    path = np.array(path).astype(int)
    # Compute the mask
    seam_mask = (np.zeros(image_shape) + 1).astype(bool)
    seam_mask[np.arange(image_shape[0]), path] = False

    # Compute the expanded mask: also include pixels on the right
    seam_mask_expanded = seam_mask.copy()

    # Add 1 column to the right to account for rightmost minimum pixels
    seam_mask_expanded = np.hstack((seam_mask_expanded, (np.zeros((image_shape[0], 1)) + 1).astype(bool)))
    seam_mask_expanded[np.arange(image_shape[0]), path + 1] = False
    seam_mask_expanded = seam_mask_expanded[:,:-1]  # Remove extra column
    return seam_mask, seam_mask_expanded


def downscale_seam_image(image, seam_mask):
    """
    Creates a new image with a reduced column dimension and copies the data which is not in the seam.
    :param image: input image
    :param seam_mask: mask with the seam path
    :return: downscaled image
    """
    # Reshape the image to be a 2d array: each row is the depth vector
    res_image = image.reshape(image.shape[0] * image.shape[1], -1)
    res_mask = seam_mask.reshape(image.shape[0] * image.shape[1], -1)

    res_new_image = res_image[res_mask[:,0], :]

    # Reshape back to original size -1
    new_image = res_new_image.reshape(image.shape[0], image.shape[1]-1, image.shape[2])
    return new_image



def downscale_alg():
    """
    Downscaling the image using the seam carving method.
    """
    image = birds
    mask = birds_mask
    #mask = np.zeros((image.shape[0], image.shape[1], 1))

    cost = np.zeros((image.shape[0], image.shape[1], 1))
    ccost = np.zeros((image.shape[0], image.shape[1], 1))
    image = np.dstack((image, cost, ccost, mask))

    original_image = image.copy()

    mean = None
    std = None
    print()
    for it in range(800):  # For N iteration
        # Get some feedback
        print("\rIt: ", it, end="", flush=True)

        # Compute energy/cost
        image[:,:,3] = energy_image(image[:,:,0:4], image[:,:,5]*4, mean, std)  # Value of mask is 765

        # Compute cumulative cost
        image[:,:,4] = cum_cost(image[:,:,3:5].copy(), mean, std)

        # Get Seam:
        seam_path = path(image[:,:,4])
        seam_mask, seam_mask_expanded = get_seam_mask(seam_path, image[:,:,4].shape)

        # Compute mean and std for next iteration
        mean = np.mean(seam_path)
        std = np.std(seam_path)

        # Add Expanded Seam to mask
        image[:,:,5] += np.logical_not(seam_mask_expanded).astype(int)
        image[:,:,5] = np.where(image[:,:,5]==0, image[:,:,5], 255)

        # Show results
        if it%20 == 0:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle("Image After Removing " + str(it) + " Columns")
            axs[0, 0].set_title("Cropped Image")
            axs[0, 0].imshow(image[:, :, 0:3].astype(int))
            axs[0, 1].set_title("Energy Image")
            axs[0, 1].imshow(image[:, :, 3].astype(int))
            axs[1, 0].set_title("Cummulative cost image")
            axs[1, 0].imshow(image[:, :, 4])
            axs[1, 1].set_title("Mask")
            axs[1, 1].imshow(image[:, :, 5])
            plt.savefig("birds_it" + str(it) + ".png")
            plt.close(fig)

            im = Image.fromarray(image[:, :, 0:3].astype(np.uint8))
            im.save("DownscaledImage_" + str(it) + ".png")

        # Remove seam parts
        image = downscale_seam_image(image, seam_mask)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title("Original image")
    axs[0, 0].imshow(original_image[:, :, 0:3].astype(int))
    axs[0, 1].set_title("Cropped image")
    axs[0, 1].imshow(image[:, :, 0:3].astype(int))
    axs[1, 0].set_title("Original mask")
    axs[1, 0].imshow(original_image[:, :, 5])
    axs[1, 1].set_title("Cropped mask")
    axs[1, 1].imshow(image[:, :, 5])
    plt.show()
    print("\nFinished Exercise 4.4")


def expand_row(row, index):
    """
    This function expands a row. It shifts 1 pixel to the right all pixels right of the index. It fills the index pixel
    with the average of the pixels from its left and right.
    :param row: row of pixels 1D array
    :param index: index to fill and shift
    :return: shifted row
    """
    if index == row.shape[0] - 1:  # rightmost column
        average = row[index, :]  # question
    else:
        average = (row[index, :]+row[index + 1, :])/2
    row_expanded = np.zeros((row.shape[0] + 1, row.shape[1]))
    row_expanded[:index + 1, :] = row[0:index + 1, :]
    row_expanded[index + 1, :] = average
    row_expanded[index + 2:, :] = row[index + 1:, :]
    return row_expanded


def upscale_seam_image(image, path):
    """
    Upsacales the input image given the path.
    :param image: input image.
    :param path: path, defines where the image is split and filled
    :return: the upscaled image
    """
    new_image = np.zeros((image.shape[0], image.shape[1]+1, image.shape[2]))
    for i in range(image.shape[0]):
        new_image[i, :, :] = expand_row(image[i,:,:], path[i])
    return new_image


def upscale_alg():
    """
    This exercise consists in using the same method to upscale the image. We compare a nature photo (vincent on a cliff)
    and a city landscape. We appreciate how the results are better on the nature landscape. Probably due to the less
    amount of details the nature landscape has.
    """

    image = paris
    mask = np.zeros((image.shape[0], image.shape[1], 1))
    cost = np.zeros((image.shape[0], image.shape[1], 1))
    ccost = np.zeros((image.shape[0], image.shape[1], 1))
    image = np.dstack((image, cost, ccost, mask))

    mean = None
    std = None
    print()
    for it in range(501):
        print("\rIt: ", it, end="", flush=True)
        # Compute energy/cost
        image[:,:,3] = energy_image(image[:,:,0:4], image[:,:,5]*4, mean, std)

        # Compute cumulative cost
        image[:, :, 4] = cum_cost(image[:, :, 3:5].copy(), mean, std)

        # Get Seam:
        seam_path = path(image[:, :, 4])
        seam_mask, seam_mask_expanded = get_seam_mask(seam_path, image[:, :, 4].shape)

        # Compute mean and std for next iteration
        mean = np.mean(seam_path)
        std = np.std(seam_path)

        # Add Expanded Seam to mask
        image[:, :, 5] += np.logical_not(seam_mask_expanded).astype(int)
        image[:, :, 5] = np.where(image[:, :, 5] == 0, image[:, :, 5], 255)

        # Show results
        if it % 20 == 0:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle("Image After Adding " + str(it) + " Columns")
            axs[0, 0].set_title("Upscaled Image")
            axs[0, 0].imshow(image[:, :, 0:3].astype(int))
            axs[0, 1].set_title("Energy Image")
            axs[0, 1].imshow(image[:, :, 3].astype(int))
            axs[1, 0].set_title("Cumulative cost image")
            axs[1, 0].imshow(image[:, :, 4])
            axs[1, 1].set_title("Mask")
            axs[1, 1].imshow(image[:, :, 5])
            plt.savefig("upscale_it" + str(it) + ".png")
            plt.close(fig)

            im = Image.fromarray(image[:, :, 0:3].astype(np.uint8))
            im.save("UpscaledImage_" + str(it) + ".png")

        # Upscale image
        image = upscale_seam_image(image, seam_path)


if __name__ == '__main__':

    # Load exercise data
    kestrel_name = 'additional-data/common-kestrel.jpg'
    birds_name = 'additional-data/kingfishers.jpg'
    mask_name = 'additional-data/kingfishers-mask.png'
    paris_name = 'additional-data/paris.jpg'

    paris = np.array(Image.open(paris_name))
    birds = np.array(Image.open(birds_name))
    birds_mask = np.array(Image.open(mask_name))[:,:,0]

    #upscale_alg()
    downscale_alg()
