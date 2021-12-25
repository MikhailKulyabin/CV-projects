import cv2
from utils import createMatchImage


def matchknn2(descriptors1, descriptors2):
    knnmatches = []
    # Two nearest neighbors for every descriptor in image 1.
    # For a given descriptor i in image 1:
    # Stored the best match (smallest distance) in knnmatches[i][0]
    # Stored the second best match in knnmatches[i][1]
    for i in range(descriptors1.shape[0]):
        matches_temp = []
        for j in range(descriptors2.shape[0]):
            distance = cv2.norm(descriptors1[i], descriptors2[j], cv2.NORM_HAMMING)
            matches_temp.append(cv2.DMatch(i, j, distance))
        matches_temp_sorted = sorted(matches_temp, key=lambda n: n.distance)
        knnmatches.append([matches_temp_sorted[0], matches_temp_sorted[1]])
    return knnmatches


def ratioTest(knnmatches, ratio_threshold):
    matches = []
    # Computed the ratio between the nearest and second nearest neighbor.
    # Added the nearest neighbor to the output matches if the ratio is smaller than ratio_threshold.
    for i in range(len(knnmatches)):
        if knnmatches[i][0].distance/knnmatches[i][1].distance < ratio_threshold:
            matches.append(knnmatches[i][0])
    return matches


def computeMatches(img1, img2):
    knnmatches = matchknn2(img1['descriptors'], img2['descriptors'])
    matches = ratioTest(knnmatches, 0.7)
    print("(" + str(img1['id']) + "," + str(img2['id']) + ") found " + str(len(matches)) + " matches.")
    return matches
