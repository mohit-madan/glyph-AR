from order_pts import order_pts
import numpy as np
import cv2


def extractMatrix(image, pts):
    pts = order_pts(pts)
    # print(pts)
    (tl, tr, br, bl) = pts

    # compute max height
    h1 = np.sqrt(np.square(tl[1] - bl[1]) + np.square(tl[0] - bl[0]))
    h2 = np.sqrt(np.square(tr[1] - br[1]) + np.square(tr[0] - br[0]))

    # width
    w1 = np.sqrt(np.square(tl[1] - tr[1]) + np.square(tl[0] - tr[0]))
    w2 = np.sqrt(np.square(bl[1] - tr[1]) + np.square(tl[0] - tr[0]))

    newHeight = max(h1, h2)
    newWidth = max(w1, w2)

    newPoints = np.array([[0, 0], [newWidth-1, 0], [newWidth-1, newHeight-1], [0, newHeight-1]], dtype="float32")
    # print(newPoints)
    H = cv2.getPerspectiveTransform(pts, newPoints)
    warped_img = cv2.warpPerspective(image, H, (newWidth, newHeight))

    return warped_img, H
