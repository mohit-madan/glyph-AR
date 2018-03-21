import cv2
import numpy as np
from order_pts import order_pts


# image: complete screen
# dst: list of points [x,y] of corners of quadrilaterlas
# superimposes substitute_image at dst
def superimpose_image(image, substitute_image, dst):

    dst = order_pts(dst)

    (tl, tr, br, bl) = dst
    # compute max height
    h1 = np.sqrt(np.square(tl[1] - bl[1]) + np.square(tl[0] - bl[0]))
    h2 = np.sqrt(np.square(tr[1] - br[1]) + np.square(tr[0] - br[0]))

    # width
    w1 = np.sqrt(np.square(tl[1] - tr[1]) + np.square(tl[0] - tr[0]))
    w2 = np.sqrt(np.square(bl[1] - tr[1]) + np.square(tl[0] - tr[0]))

    newHeight = int(max(h1, h2))
    newWidth = int(max(w1, w2))

    # get position where to substitute in original image
    min_x = min(int(tl[0]), int(bl[0]))
    min_y = min(int(tl[1]), int(tr[1]))
    max_x = max(int(tr[0]), int(br[0]), int(tl[1]), int(bl[0]))
    max_y = max(int(bl[1]), int(br[1]), int(tl[1]), int(tr[1]))

    # size of patch to be replaced in original image
    destHeight = max_y - min_y
    destWidth = max_x - min_x

    subst_width = newWidth
    subst_height = newHeight

    # resize to distorted topdown view + 1 transparent channel
    substitute_image = cv2.resize(substitute_image, (subst_width, subst_height))
    transp_image = np.zeros((subst_height, subst_width, 4), np.uint8)
    transp_image[:, :, 3] = 255
    transp_image[:, :, 0:3] = substitute_image

    # prepare dst and src
    # set dst to zero origin
    for el in dst:
        el[0] = el[0] - min_x
        el[1] = el[1] - min_y

    # get topdown view as src
    src = np.array([[0, 0], [subst_width-1, 0], [subst_width-1, subst_height-1], [0, subst_height-1]], dtype="float32")

    # prepare output size + 1 for transparent channel
    warped = np.zeros((destHeight, destWidth, 4), np.uint8)
    warped[:, :, 0:3] = 255
    warped[:, :, 3] = 0

    # apply matrix to topdown view
    matrix2 = cv2.getPerspectiveTransform(src, dst)
    cv2.warpPerspective(transp_image, matrix2, (destWidth, destHeight), warped, borderMode=cv2.BORDER_TRANSPARENT)
    # cv2.imshow("warped warped", warped)
    warped.resize(destHeight, destWidth, 4)
    # cv2.imshow("topdown substitute",substitute_image)
    # cv2.waitKey(0)

    # insert warped substitute into image without white border
    # get crop of image with + 1 channel for transparency
    newImage = np.zeros((destHeight, destWidth, 4), np.uint8)
    newImage[:, :, 0:3] = image[min_y:min_y + destHeight, min_x:min_x + destWidth]
    newImage[:, :, 3] = 255

    # get map of pixels that are not border
    transp_map = warped[:, :, 3] > 0
    # apply map on cropped (same size) part of original image
    newImage[transp_map] = warped[transp_map]
    # transfer crop to original size image
    image[min_y:min_y + destHeight, min_x:min_x + destWidth] = newImage[:, :, 0:3]

    return None
