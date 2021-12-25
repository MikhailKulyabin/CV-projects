import numpy as np
import cv2
from math import floor, ceil


def computeHtoref(image_data_dir, center):
    for i in range(center-1, -1, -1):
        c = image_data_dir[i]
        next_ = image_data_dir[i+1]
        c['HtoReference'] = np.matmul(next_['HtoReference'], c['HtoNext'])

    for i in range(center+1, len(image_data_dir), 1):
        c = image_data_dir[i]
        prev = image_data_dir[i-1]
        c['HtoReference'] = np.matmul(prev['HtoReference'], c['HtoPrev'])
    return image_data_dir


def createStichedImage(image_data_dir):
    print("Stitching with " + str(len(image_data_dir)) + " images..")
    center = len(image_data_dir) // 2
    ref = image_data_dir[center]
    image_data_dir = computeHtoref(image_data_dir, center)
    print("Reference Image : " + str(center) + " - " + ref['file'])

    minx = 2353535
    maxx = -2353535
    miny = 2353535
    maxy = -2353535

    for i in range(len(image_data_dir)):
        img2 = image_data_dir[i]
        corners2 = [0, 0, 0, 0]
        corners2[0] = (0, 0)
        corners2[1] = (img2['img'].shape[1], 0)
        corners2[2] = (img2['img'].shape[1], img2['img'].shape[0])
        corners2[3] = (0, img2['img'].shape[0])
        corners2 = np.array(corners2, dtype='float32')
        corners2_in_1 = cv2.perspectiveTransform(corners2[None, :, :], img2['HtoReference'])

        for p in corners2_in_1[0]:
            minx = min(minx, p[0])
            maxx = max(maxx, p[0])
            miny = min(miny, p[1])
            maxy = max(maxy, p[1])

    roi = np.array([floor(minx), floor(miny), ceil(maxx)-floor(minx), ceil(maxy)-floor(miny)])
    print("ROI " + str(roi))

    # Translated everything so the top left corner is at (0,0)
    offsetX = floor(minx)
    offsetY = floor(miny)
    ref['HtoReference'][0, 2] = -offsetX
    ref['HtoReference'][1, 2] = -offsetY
    computeHtoref(image_data_dir, center)

    # cv2.namedWindow('Panorama')
    # cv2.moveWindow('Panorama', 0, 500)

    stitchedImage = np.zeros([roi[3], roi[2], 3], dtype='uint8')
    for k in range(len(image_data_dir) + 1):
        if k % 2 == 0:
            tmp = 1
        else:
            tmp = -1
        i = center + tmp * ((k + 1) // 2)

        # Out of index bounds check
        if (i < 0 or i >= len(image_data_dir)):
            continue

        # Projected onto the reference image plane
        img2 = image_data_dir[i]
        tmp = np.zeros([roi[3], roi[2], 3])
        rgba_img = cv2.cvtColor(img2['img'], cv2.COLOR_RGB2RGBA)
        rgba_img[:, :, 3] = 255
        tmp = cv2.warpPerspective(rgba_img, img2['HtoReference'], (tmp.shape[1], tmp.shape[0]), cv2.INTER_NEAREST)
        ## Added it to the output image
        for y in range(stitchedImage.shape[0]):
            for x in range(stitchedImage.shape[1]):
                if (tmp[y, x, 3] == 255 and np.array_equal(stitchedImage[y, x], np.array([0, 0, 0]))):
                    stitchedImage[y, x] = tmp[y, x, 0:3]

        print("Added image " + str(i) + " - " + str(img2['file']) + ".")
        print("Press any key to continue...")
        # cv2.imshow("Panorama", stitchedImage)
        # cv2.waitKey(0)
    return stitchedImage
