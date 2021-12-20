import numpy as np
from scipy import io as sio
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from utils import utils
from utils.RANSAC import RANSAC
from matplotlib.patches import Circle


def main():
    # Load the data
    image_n = 3
    if image_n == 4:
        file_name = 'files/example4kinect.mat'
        data = sio.loadmat(file_name)
        A = data['amplitudes3']
    else:
        file_name = 'files/example' + str(image_n) + 'kinect.mat'
        data = sio.loadmat(file_name)
        A = data['amplitudes' + str(image_n)]

    D = data['distances' + str(image_n)]
    C = data['cloud' + str(image_n)]

    # Visualize the data
    show_exercise_images(A, D, C, title="Original input images")

    # Prepare data to be processed
    A = utils.normalize_image(A)
    D = utils.normalize_image(D)
    C_res = C.reshape((-1, 3))  # C into an array of 3d vectors

    # 1. Use ransac algorithm to find the floor plane equations
    threshold = 0.15
    ransac = RANSAC(C_res)
    best_model_floor = ransac.fit(threshold=threshold, max_iter=100, n_samples=10000)
    floor_mask = ransac.get_mask(best_model_floor, threshold=threshold, show=False)
    floor_mask = floor_mask.reshape((424,512))
    show_exercise_images(A, floor_mask, title="Floor Mask")

    # 2. Get Box mask (inverted floor mask) and filter it for a better result
    box_mask = D.copy()
    box_mask[floor_mask] = 0  # allows to get a better box mask
    box_mask = ndimage.median_filter(box_mask, 15)
    box_mask = ndimage.morphology.binary_dilation(box_mask)
    box_mask = ndimage.morphology.binary_opening(box_mask)
    show_exercise_images(A, box_mask.astype(int), title="Filtered Box Mask")

    # 3. Reshape the box mask to remove the ground points from the point cloud
    box_mask_res = box_mask.reshape((-1, 1))[:,0].astype(bool)
    box_points_res = C_res[box_mask_res, :]

    # 4. Use Ransac in the box points to find the top plane equation
    new_threshold = 0.015
    ransac = RANSAC(box_points_res)
    best_model_top = ransac.fit(threshold=new_threshold, max_iter=100)
    top_plane_mask = ransac.get_mask(best_model_top, threshold=new_threshold, show=False, new_cloud=C_res)
    top_plane_mask = top_plane_mask.reshape((424, 512))
    show_exercise_images(A, box_mask.astype(int), title="Top Plane Mask")

    # 5. Extract the second largest set of connected components from the top plane mask
    # to remove noise and get only the top plane mask
    labeled_array, num_features = ndimage.label(top_plane_mask)
    unique_labels, counts = np.unique(labeled_array, return_counts=True)
    max_label = np.argmax(counts[1:]) + 1
    labeled_array[labeled_array != max_label] = 0
    labeled_array[labeled_array == max_label] = 1
    top_plane_mask = labeled_array

    # 6. Filter top plane mask
    filter_size = 7
    top_plane_mask = ndimage.morphology.binary_closing(top_plane_mask, structure=np.ones((filter_size,filter_size)))
    top_plane_mask = ndimage.morphology.binary_erosion(top_plane_mask, structure=np.ones((filter_size,filter_size)))
    show_exercise_images(A, top_plane_mask, title="Filtered Top Plane Mask")

    # Show the computed planes
    test_planes(best_model_floor, best_model_top, C=C)

    # 7. Find the corners of the top plane using the top plane mask
    data = cv2.findContours(top_plane_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = data[0][0] if len(data) == 2 else data[0][1]
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.05 * perimeter, True)  # corners of the top plane
    corners = corners[:, 0]
    corners_sort = corners[corners[:, 1].argsort()]
    corners_up = corners_sort[:-2]
    corners_up = corners_up[corners_up[1, :].argsort()]
    corners_down = corners_sort[2:]
    corners_down = corners_down[corners_down[1, :].argsort()]

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(A)
    for corner in corners:
        circle = Circle(corner, 2, color='k')
        ax.add_patch(circle)
    ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]], 'r')
    ax.plot([corners[1, 0], corners[2, 0]], [corners[1, 1], corners[2, 1]], 'r')
    ax.plot([corners[2, 0], corners[3, 0]], [corners[2, 1], corners[3, 1]], 'r')
    ax.plot([corners[3, 0], corners[0, 0]], [corners[3, 1], corners[0, 1]], 'r')
    text_shift = 20
    ax.text(corners_up[0][0]+text_shift, corners_up[0][1]+text_shift, "UR")
    ax.text(corners_up[1][0]-2*text_shift, corners_up[1][1]+text_shift, "UL")
    ax.text(corners_down[0][0]+text_shift, corners_down[0][1]-text_shift, "DR")
    ax.text(corners_down[1][0]-2*text_shift, corners_down[1][1]-text_shift, "DL")
    fig.suptitle("Located Corners")
    plt.show()

    # 8. Get the corner coordinates
    coordinates = []
    heights = []  # We average heights of each corner
    for corner in corners:
        coord = C[corner[1], corner[0], :]
        coordinates.append(coord)
        top_height = best_model_top[0,0] + best_model_top[1,0]*coord[0] + best_model_top[2,0]*coord[1]
        floor_height = best_model_floor[0,0] + best_model_floor[1,0]*coord[0] + best_model_floor[2,0]*coord[1]
        height = -(top_height - floor_height)
        heights.append(height)

    coordinates = np.array(coordinates)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 'r')
    ax.plot([coordinates[0, 0], coordinates[1, 0]], [coordinates[0, 1], coordinates[1, 1]], [coordinates[0, 2], coordinates[1, 2]], 'r')
    ax.plot([coordinates[1, 0], coordinates[2, 0]], [coordinates[1, 1], coordinates[2, 1]], [coordinates[1, 2], coordinates[2, 2]], 'r')
    ax.plot([coordinates[2, 0], coordinates[3, 0]], [coordinates[2, 1], coordinates[3, 1]], [coordinates[2, 2], coordinates[3, 2]], 'r')
    ax.plot([coordinates[3, 0], coordinates[0, 0]], [coordinates[3, 1], coordinates[0, 1]], [coordinates[3, 2], coordinates[0, 2]], 'r')
    ax.scatter(C_res[::250, 0], C_res[::250, 1], C_res[::250, 2], 'b')
    fig.suptitle("Located Corners in point cloud")
    plt.show()

    # 7. Compute dimensions: length and width are computer as the average of the length of the 2 opposite edges
    # width is computed as the average of the height of the 4 vertex
    final_height = np.mean(heights)

    width1 = np.sqrt((coordinates[0, 0] - coordinates[1, 0]) ** 2 + (coordinates[0, 1] - coordinates[1, 1]) ** 2 + (
                coordinates[0, 2] - coordinates[1, 2]) ** 2)
    width2 = np.sqrt((coordinates[2, 0] - coordinates[3, 0]) ** 2 + (coordinates[2, 1] - coordinates[3, 1]) ** 2 + (
                coordinates[2, 2] - coordinates[3, 2]) ** 2)
    final_width = (width1 + width2)/2  # Average of the 2

    length1 = np.sqrt((coordinates[1, 0] - coordinates[2, 0]) ** 2 + (coordinates[1, 1] - coordinates[2, 1]) ** 2 + (
            coordinates[1, 2] - coordinates[2, 2]) ** 2)
    length2 = np.sqrt((coordinates[3, 0] - coordinates[0, 0]) ** 2 + (coordinates[3, 1] - coordinates[0, 1]) ** 2 + (
            coordinates[3, 1] - coordinates[0, 1]) ** 2)
    final_length = (length1 + length2) / 2  # Average of the 2

    print()
    print("Final dimensions:")
    print("Height= ", final_height)
    print("Width= ", final_width)
    print("Length= ", final_length)

    print("Program finished")


def show_exercise_images(A, D, C=None, title=""):
    """Shows the images of this exercise"""
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(title)
    axs[0].imshow(A, cmap='Greys_r')
    axs[0].title.set_text("ToF Image")
    axs[1].imshow(D, cmap='jet')
    axs[1].title.set_text("Distance Image")

    if C is not None:
        fig2 = plt.figure()
        subsample_factor = 5
        ax = fig2.add_subplot(projection='3d')
        ax.scatter(C[::subsample_factor, ::subsample_factor, 0], C[::subsample_factor, ::subsample_factor, 1],
                   C[::subsample_factor, ::subsample_factor, 2])  # Subsampled
        ax.title.set_text("Point Cloud")
        fig2.suptitle(title)
    plt.show()


def test_planes(model_floor, model_top, n_points=15, C=None):
    """
    Test a model wrt the input data and visualizes it.
    :param model: regressed parameters
    :type model: np.ndarray
    :param n_points: number of test points (in each axis)
    :type n_points: int
    """
    # Test fit
    ls_x = np.linspace(-1, 1, n_points)
    ls_y = np.linspace(-1, 1, n_points)
    x, y = np.meshgrid(ls_x, ls_y)
    tmp = np.stack((x, y), axis=-1).reshape(-1, 2)

    test_x = np.ones((n_points**2, 3))
    test_x[:, 1:] = tmp
    test_y_1 = test_x@model_floor
    test_y_2 = test_x@model_top

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(test_x[:, 1], test_x[:, 2], test_y_1, label="Floor plane")
    ax.scatter(test_x[:, 1], test_x[:, 2], test_y_2, label="Top plane")
    if C is not None:
        ax.scatter(C[::20, ::20, 0], C[::20, ::20, 1], C[::20, ::20, 2], label="Original Point Cloud")
    ax.legend()
    fig.suptitle("Found planes")
    plt.show()


if __name__ == '__main__':
    main()
