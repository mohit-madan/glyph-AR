import numpy as np


def order_pts(pts):
    s = np.sum(pts, axis=1)
    ord_pts = np.zeros((4, 2), dtype="float32")
    ord_pts[0] = pts[np.argmin(s)]
    ord_pts[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    ord_pts[1] = pts[np.argmin(diff)]
    ord_pts[3] = pts[np.argmax(diff)]
    return ord_pts


def check_if_rect(pts):
    (tl, tr, br, bl) = pts
    check = True

    if (np.abs(tl[1] - tr[1]) + np.abs(tl[0] - tr[0])) < 5:
        check = False
    if (np.abs(tl[1] - br[1]) + np.abs(tl[0] - br[0])) < 5:
        check = False
    if (np.abs(tl[1] - bl[1]) + np.abs(tl[0] - bl[0])) < 5:
        check = False
    if (np.abs(tr[1] - bl[1]) + np.abs(tr[0] - bl[0])) < 5:
        check = False

    return check
