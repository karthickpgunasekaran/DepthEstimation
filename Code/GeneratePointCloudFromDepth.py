import numpy as np
from imageio import imread, imsave
def point_cloud(depth):
    rows, cols = depth.shape
    cx =245
    cy=234
    fx=100
    fy =56
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 256.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))





depth = imread("DepthMap/-0000000004_depth.png")
point_cloud = point_cloud(depth)

print(point_cloud)

f = open("PointCloud/pc1.txt","w+")

f.write(str(point_cloud))

f.close()