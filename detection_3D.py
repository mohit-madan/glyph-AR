import cv2
from extractMatrix import extractMatrix
from pattern_recognition import pattern_recognition
from order_pts import check_if_rect, order_pts


def capture(frame):
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)  # applying histogram equalisation

    gray = cv2.GaussianBlur(gray, (5, 5), 1)  # gaussian blur-to smoothen out random edges
    gray_edge = cv2.Canny(gray, 100, 200)  # applying canny edge detection

    im2, contours, _ = cv2.findContours(gray_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[
               :10]  # sorting contours in reverse order to get biggest one first

    # approximating contours
    idx = None
    approx = None
    for cnt in contours:
        idx = None
        # We will find if each of the detected contour is of quad shape, then we will do the perspective
        # transform of the image to get the quad in top down view and then the glyph detection algo will proceed
        epsilon = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * epsilon, True)  # remove noise along contour, return corners
        vert_num = len(approx)
        # only consider contours that are quadrilateral
        if vert_num == 4:
            approx = approx.reshape(4, 2)
            (tl, tr, br, bl) = order_pts(approx)
            valid = check_if_rect(approx)  # reject small rectangles
            if valid:
                warped_img, _ = extractMatrix(gray, approx)
                idx = pattern_recognition(warped_img)
                break

    return idx, approx
