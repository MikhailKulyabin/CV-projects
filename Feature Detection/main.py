import cv2
import numpy as np


def show(name, img, x, y):
    windowStartX  = 10
    windowStartY  = 50
    windowXoffset = 5
    windowYoffset = 40

    w = img.shape[0] + windowXoffset
    h = img.shape[1] + windowYoffset
    cv2.namedWindow(name)
    cv2.moveWindow(name, windowStartX + w * x, windowStartY + h * y)
    cv2.imshow(name, img)
    cv2.waitKey(0)


def harrisResponseImage(img):
    # Compute the spatial derivatives in x and y direction.
    dIdx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dIdy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    #show("dI/dx", abs(dIdx), 1, 0)
    #show("dI/dy", abs(dIdy), 2, 0)

    # Compute Ixx, Iyy, and Ixy with
    Ixx = np.multiply(dIdx, dIdx)
    Iyy = np.multiply(dIdy, dIdy)
    Ixy = np.multiply(dIdx, dIdy)

    #show("Ixx", abs(Ixx), 0, 1)
    #show("Iyy", abs(Iyy), 1, 1)
    #show("Ixy", abs(Ixy), 2, 1)

    # Compute the images A,B, and C by blurring the images Ixx, Iyy, and Ixy with a Gaussian filter
    kernelSize = (3, 3)
    sdev = 1
    A = cv2.GaussianBlur(Ixx, kernelSize, sigmaX=sdev, sigmaY=sdev)
    B = cv2.GaussianBlur(Iyy, kernelSize, sigmaX=sdev, sigmaY=sdev)
    C = cv2.GaussianBlur(Ixy, kernelSize, sigmaX=sdev, sigmaY=sdev)

    #show("A", abs(A) * 5, 0, 1)
    #show("B", abs(B) * 5, 1, 1)
    #show("C", abs(C) * 5, 2, 1)

    # Compute the harris response with the following formula:
    Det = np.subtract(np.multiply(A, B), np.multiply(C, C))
    Trace = np.add(A, B)
    k = 0.06
    response = Det - k * np.multiply(Trace, Trace)
    # Normalize the response image
    dbg = (response - np.min(response)) / (np.max(response) - np.min(response))
    dbg = dbg.astype(np.float32)
    #show("Harris Response", dbg, 0, 2)

    cv2.imwrite("dIdx.png", (abs(dIdx) * 255.0))
    cv2.imwrite("dIdy.png", (abs(dIdy) * 255.0))
    cv2.imwrite("A.png", (abs(A) * 5 * 255.0))
    cv2.imwrite("B.png", (abs(B) * 5 * 255.0))
    cv2.imwrite("C.png", (abs(C) * 5 * 255.0))
    cv2.imwrite("response.png", np.uint8(dbg * 255.0))
    return response


def harrisKeypoints(response, threshold=0.1):
    # Generate a keypoint for a pixel, if the response is larger than the threshold and it is a local maximum.
    points = []
    for x in range(len(response)):
        for y in range(len(response[x])):
            if response[x, y] > threshold:
                if response[x, y] > response[x, y + 1] \
                        and response[x, y] > response[x, y - 1] \
                        and response[x, y] > response[x + 1, y] \
                        and response[x, y] > response[x - 1, y] \
                        and response[x, y] > response[x - 1, y - 1] \
                        and response[x, y] > response[x + 1, y + 1] \
                        and response[x, y] > response[x - 1, y + 1] \
                        and response[x, y] > response[x + 1, y - 1]:
                            points.append(cv2.KeyPoint(y, x, 1))
    return points


def harrisEdges(input, response, edge_threshold=-0.01):
    # Set edge pixels to red.
    # A pixel belongs to an edge, if the response is smaller than a threshold, and it is a minimum in x or y direction.
    result = input.copy()
    for x in range(len(result)):
        for y in range(len(result[x])):
            if response[x, y] < edge_threshold:
                if (response[x, y] < response[x + 1, y] and response[x, y] < response[x - 1, y]) \
                    or (response[x, y] < response[x, y + 1] and response[x, y] < response[x, y - 1]):
                    result[x][y] = (0, 0, 255)
    return result


def main():
    input_img = cv2.imread('blox.jpg')  # read the image
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    input_gray = (input_gray - np.min(input_gray)) / (np.max(input_gray) - np.min(input_gray))  # normalize
    input_gray = input_gray.astype(np.float32)  # convert to float32 for filtering

    # Obtain Harris Response, corners and edges
    response = harrisResponseImage(input_gray)
    points = harrisKeypoints(response)
    edges = harrisEdges(input_img, response)
    imgKeypoints1 = cv2.drawKeypoints(input_img, points,  outImage=None, color=(0, 255, 0))
    #show("Harris Keypoints", imgKeypoints1, 1, 2)
    #show("Harris Edges", edges, 2, 2)
    cv2.imwrite("edges.png", edges)
    cv2.imwrite("corners.png", imgKeypoints1)


if __name__ == '__main__':
    main()

