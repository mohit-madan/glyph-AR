import numpy as np
def order_pts(pts):
    s = np.sum(pts,axis = 1)
    ord_pts = np.zeros((4,2), dtype = "float32")
    ord_pts[0] = pts[np.argmin(s)]
    ord_pts[2] = pts[np.argmax(s)]

    diff = np.diff(pts,axis = 1)
    ord_pts[1] = pts[np.argmin(diff)]
    ord_pts[3] = pts[np.argmax(diff)]
    return ord_pts