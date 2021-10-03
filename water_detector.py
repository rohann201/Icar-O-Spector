
import cv2 as cv
import numpy as np
import math

# Color of a lake [blue green red]
BGR = np.array([255, 218, 170])
upper = BGR + 30
lower = BGR - 30

crd_x, crd_y = 0, 0  # the coordinates of the potential hostpot in the location
pop = 100000  # population of the potential hotspot region

# Read an image from disk
# @param {path} the path of the image to read
# @returns {image} the image


def utilize(pop, area):  # function to calculate the ratio of water being utilized
    # assuming that an average person utilizes 30 pounds of water
    return area/(30*0.02*pop)
    # value betw 1-10


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return 50 - dist
 # value 1-50


def read_image(path):
    return cv.imread(path)

# applies a threshold to an image based on two boundaries


def find_mask(image):
    return cv.inRange(image, lower, upper)


def find_contours(mask):
    (cnts, hierarchy) = cv.findContours(
        mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    print("Found %d black shapes" % (len(cnts)))
    return cnts

# draw contours on an image


def show_contours(contours, image):
    cv.drawContours(image, contours, -1, (0, 0, 255), 2)

    cv.imshow("contours", image)


def get_main_contour(contours):
    copy = contours.copy()
    copy.sort(key=len, reverse=True)
    return copy[0]


if __name__ == "__main__":
    image = read_image("pond.png")
    # print(image.shape)
    mask = find_mask(image)
    # print(mask.shape)
    contours = find_contours(mask)
    for c in contours:
        # compute the center of the contour
        M = cv.moments(c)
        # The coordinates of the centroid for the given contour
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    main_contour = get_main_contour(contours)
    show_contours([main_contour], image)
    # calculating the distance between the potential hotspot and the center of the water body
    d = calculateDistance(cX, cY, crd_x, crd_y)

    for contour in contours:
        rect = cv.boundingRect(contour)
        area = rect[2] * rect[3]

    u = utilize(pop, area)
    score = u*5 + d
    key = cv.waitKey(0)
