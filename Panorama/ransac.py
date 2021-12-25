import numpy as np
from homography import computeHomography


def numInliers(points1, points2, H, threshold):
    inlierCount = 0

    # Computed the number of inliers for the given homography
    # Projected the image points from image 1 to image 2
    # Point is an inlier if the distance between the projected point and the point in image 2 is smaller than threshold.

    for i in range(len(points1)):
        p = points1[i]
        q = points2[i]
        hom_q = np.array([q[0], q[1], 1])
        hom_p = np.array([p[0], p[1], 1])
        if np.linalg.norm(hom_q - np.dot(H, hom_p)/np.dot(H, hom_p)[2], ord=2) < threshold:
            inlierCount += 1
    return inlierCount


def computeHomographyRansac(img1, img2, matches, iterations, threshold):
    points1 = []
    points2 = []
    for i in range(len(matches)):
        points1.append(img1['keypoints'][matches[i].queryIdx].pt)
        points2.append(img2['keypoints'][matches[i].trainIdx].pt)
    ## The best homography and the number of inlier for this H
    bestInlierCount = 0
    bestH = None
    for i in range(iterations):
        subset1 = []
        subset2 = []
        # Constructed the subsets by randomly choosing 4 matches.
        index_subset = []
        # Avoiding repetitiveness of points
        while len(index_subset) < 4:
            x = np.random.randint(0, len(points1))
            if x not in index_subset:
                index_subset.append(x)
                subset1.append(points1[x])
                subset2.append(points2[x])
        H = computeHomography(subset1, subset2)  # homography for this subset
        inlierCount = numInliers(points1, points2, H, threshold)  # number of inliers
        # Track of the best homography
        if inlierCount > bestInlierCount:
            bestInlierCount = inlierCount
            bestH = H
    print("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(bestInlierCount) + " RANSAC inliers.")
    return bestH