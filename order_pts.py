def order_pts(pts):
	s = pts.sum(axis = 1)
	ord_pts[0] = pts[np.argmin(s)]
	ord_pts[2] = pts[np.argmax(s)]

	diff = pts.sum(axis = 1)
	ord_pts[1] = pts[np.argmin(s)]
	ord_pts[3] = pts[np.argmax(s)]

	return ord_pts