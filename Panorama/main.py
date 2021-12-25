import numpy as np
import cv2
import os
from matching import computeMatches, createMatchImage
from ransac import computeHomographyRansac
from stitcher import createStichedImage
from utils import computeFeatures

IMAGE_DIR = './data_uni/'


def main():
    images = [
        "7.jpg",  "8.jpg",  "9.jpg",  "10.jpg", "11.jpg",
        "12.jpg", "13.jpg", "14.jpg", "15.jpg"]
    image_data_dicts = []
    for i, image_name in enumerate(images):
        image_data = {}
        image_data['file'] = image_name
        image_path = os.path.join(IMAGE_DIR, image_name)
        image_data['img'] = cv2.imread(image_path)
        image_data['img'] = cv2.resize(image_data['img'], None, fx=0.5, fy=0.5)
        image_data['id'] = i
        image_data['HtoReference'] = np.eye(3)
        image_data['HtoPrev'] = np.eye(3)
        image_data['HtoNext'] = np.eye(3)
        assert len(image_data['img']) > 0  # Image read properly check
        image_data_dicts.append(image_data)
        print("Loaded image " + str(image_data['id']) + " " +
                str(image_data['file']) + " " + str(image_data['img'].shape[0])
                + "x" + str(image_data['img'].shape[1]))

    # =========== Feature Detection ===========
    temp_image_data_dicts = []
    for image_data in image_data_dicts:
        new_image_data = computeFeatures(image_data)
        temp_image_data_dicts.append(new_image_data)
    image_data_dicts = temp_image_data_dicts

    # =========== Pairwise Feature Matching ===========
    for i in range(1, len(image_data_dicts)):
        matches = computeMatches(image_data_dicts[i-1], image_data_dicts[i])

        # Debug output
        matchImg = createMatchImage(image_data_dicts[i-1],  image_data_dicts[i], matches)
        h = 200
        w = int((float(matchImg.shape[1]) / matchImg.shape[0]) * h)
        matchImg = cv2.resize(matchImg, (w, h))
        name = "Matches (" + str(i - 1) + "," + str(i) + ") " + image_data_dicts[i - 1]['file'] + " - " + image_data_dicts[i]['file']
        # cv2.namedWindow(name)
        # cv2.moveWindow(name, int(10 + w * ((i - 1) % 2)), int(10 + (h + 30) * ((i - 1) / 2)))
        # cv2.imshow(name, matchImg)

        H = computeHomographyRansac(image_data_dicts[i-1], image_data_dicts[i], matches, 1000, 2.0)
        image_data_dicts[i]['HtoPrev'] = np.linalg.inv(H)
        image_data_dicts[i-1]['HtoNext'] = H

    # =============== Stitching ==================
    simg = createStichedImage(image_data_dicts)
    cv2.imwrite("output.png", simg)


if __name__ == "__main__":
    main()
