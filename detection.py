import cv2
from extractMatrix import extractMatrix
from superimpose_image import superimpose_image
from pattern_recognition import pattern_recognition
from order_pts import check_if_rect, order_pts

i = 0
cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)  # applying histogram equalisation
    
    gray = cv2.GaussianBlur(gray, (5, 5), 1)  # gaussian blur-to smoothen out random edges
    gray_edge = cv2.Canny(gray, 100, 200)  # applying canny edge detection
    cv2.imshow("grayimg", gray_edge)
    im2, contours, _ = cv2.findContours(gray_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # sorting contours in reverse order - why? don't know
    # approximating contours

    for cnt in contours:
        # We will find if each of the detected contour is of quad shape, then we will do the perspective
        # transform of the image to get the quad in top down view and then the glyph detection algo will proceed
        epsilon = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01*epsilon, True)  # with greater percentage a large set is coming
        cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)
        cv2.drawContours(frame, approx, -1, (0, 0, 255), 3)
        cv2.imshow('frame', frame)
        vert_num = len(approx)
        if vert_num == 4:
            approx = approx.reshape(4, 2)
            (tl, tr, br, bl) = order_pts(approx)
            valid = check_if_rect(approx)    
            # valid = qFalse

            if valid:
                i += 1
                # print(i)
                # print(approx)
                warped_img, H = extractMatrix(gray, approx)
                cv2.imshow("original", gray)
                cv2.imshow("transformed", warped_img)

                idx = pattern_recognition(warped_img)

                if idx == 0:
                    substitute_image = cv2.imread('data/tree2.jpg',1)
                    superimpose_image(frame, substitute_image, approx)
                    #break
                if idx == 1:
                    substitute_image = cv2.imread('data/mohit.jpeg',1)
                    superimpose_image(frame, substitute_image, approx)
                    #break
                # print(idx)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()


