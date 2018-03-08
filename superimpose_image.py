import cv2
import numpy as np
from detection import order_pts

# image: complete screen
# dst: list of points [x,y] of corners of quadratile
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

    newHeight = int(max(h1,h2))
    newWidth = int(max(w1,w2))

    # resize to distorted topdown view
    substitute_image = cv2.resize(substitute_image,(newWidth,newHeight))

    # get position where to substitute in original image
    min_x = min(int(tl[0]), int(bl[0]))
    min_y = min(int(tl[1]), int(tr[1]))
    max_x = max(int(tr[0]), int(br[0]))
    max_y = max(int(bl[1]), int(br[1]))

    # todo not sure why it works with newHeight/newWidth instead of destHeight/destWidth
    destHeight = max_y - min_y
    destWidth = max_x - min_y

    # prepare dst and src
    # set dst to zero origin
    for el in dst:
        el[0] = el[0] - min_x
        el[1] = el[1] - min_y

    # get topdown view as src
    src = np.array([[0,0],[newWidth-1,0],[newWidth-1,newHeight-1],[0,newHeight-1]],dtype = "float32")


    # prepare output size
    warped = np.zeros((newHeight,newWidth, 3), np.uint8)
    warped[:,:,:] = 255


    # apply matrix to topdown view
    matrix2 = cv2.getPerspectiveTransform(src,dst)
    cv2.warpPerspective(substitute_image, matrix2, (newWidth,newHeight), warped, borderMode=cv2.BORDER_TRANSPARENT)
    # cv2.imshow("warped warped", warped)
    # cv2.imshow("topdown substitute",substitute_image)
    # cv2.waitKey(0)


    # insert warped substitute into image
    image[min_y:min_y + newHeight, min_x:min_x + newWidth] = warped

    return None
