import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import random
import plotly.graph_objects as go

def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def get_transformed_pixels_coords(I, H, shift=None):
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]
    
    return cart_H.reshape((*I.shape[:2], 2))

def apply_H_fixed_image_size(I, H, corners):
    # Fixed canvas
    xmin, xmax, ymin, ymax = corners
    out_w = int(xmax - xmin + 1)
    out_h = int(ymax - ymin + 1)

    # Build a grid containing all pixel coordinates of the output canvas
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax + 1),
        np.arange(ymin, ymax + 1)
    )

    pts_out = np.vstack([
        xx.ravel(),
        yy.ravel(),
        np.ones(xx.size)
    ])

    # Get inverse warp of all pixels
    H_inv = np.linalg.inv(H)
    src = H_inv @ pts_out
    src /= src[2]

    x_src = src[0].reshape(out_h, out_w)
    y_src = src[1].reshape(out_h, out_w)

    coords = [y_src.ravel(), x_src.ravel()]

    if I.ndim == 2:
        # Grayscale
        out = map_coordinates(
            I,
            coords,
            order=1, # Bilinear
            mode='constant',
            cval=0
        ).reshape(out_h, out_w)
    else:
        # 3 channel
        channels = []
        for c in range(I.shape[2]):
            ch = map_coordinates(
                I[:, :, c],
                coords,
                order=1,
                mode='constant',
                cval=0
            ).reshape(out_h, out_w)
            channels.append(ch)
        out = np.stack(channels, axis=2)

    return out
    
def Normalise_last_coord(x):
    xn = x  / x[2,:]
    
    return xn

def Normalize_points(points):
    """
    Normalization: centroid -> (0,0) and mean distance -> sqrt(2)
    points: 3xN homogeneous
    returns: points_n (3xN), T (3x3) with points_n = T @ points
    """
    p = Normalise_last_coord(points)
    x, y = p[0, :], p[1, :]

    cx, cy = np.mean(x), np.mean(y)
    x_shift = x - cx
    y_shift = y - cy

    mean_dist = np.mean(np.sqrt(x_shift**2 + y_shift**2)) + 1e-8
    s = np.sqrt(2) / mean_dist

    T = np.array([
        [s, 0, -s*cx],
        [0, s, -s*cy],
        [0, 0,   1  ]
    ], dtype=float)

    points_n = T @ p
    return points_n, T


def DLT_homography(points1, points2):
    """
    Normalized DLT for homography estimation
    points1: 3xN (homogeneous) in image 1
    points2: 3xN (homogeneous) in image 2
    returns H such that points2 ~ H * points1
    """
    assert points1.shape[0] == 3 and points2.shape[0] == 3 
    assert points1.shape[1] >= 4 and points2.shape[1] == points1.shape[1] # Minimum 4 correspondences needed

    # Normalize points
    points1, T1 = Normalize_points(points1)
    points2, T2 = Normalize_points(points2)
    
    N = points1.shape[1]
    A = []

    # For each correspondence, compute Ai
    for i in range(N):
        x, y, w  = points1[:, i]
        xp, yp, wp = points2[:, i] 
        
        # Two equations per correspondence
        A.append([0, 0, 0, -wp*x, -wp*y, -wp*w,  yp*x,  yp*y,  yp*w])
        A.append([wp*x, wp*y, wp*w, 0, 0, 0, -xp*x, -xp*y, -xp*w])

    # Assemble the N matrices Ai to form the 2N x 9 matrix A
    A = np.asarray(A)
    assert A.shape == (2*N, 9)

    # Compute the SVD of A
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :] # Last row of Vt = last column of V, corresponds to smallest singular value 
    Hn = h.reshape(3, 3) # Reshape h to form Hn (normalized homography)

    # Denormalize the homography
    H = np.linalg.inv(T2) @ Hn @ T1

    # Normalize scale (homography is defined up to scale; we fix it by setting H[2,2] = 1 when possible)
    if abs(H[2,2]) > 1e-12:
        H = H / H[2,2]
    else:
        H = H / (np.linalg.norm(H) + 1e-12)

    return H

def Inliers(H, points1, points2, th):
    
    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx
    
    # Forward projection 
    fwd_proj = H @ points1
    fwd_proj = Normalise_last_coord(fwd_proj)

    # Backward projection
    H_inv = np.linalg.inv(H)
    bwd_proj = H_inv @ points2
    bwd_proj = Normalise_last_coord(bwd_proj)

    # Normalize points
    points1 = Normalise_last_coord(points1)
    points2 = Normalise_last_coord(points2)

    # Compute symmetric transfer error
    error1 = np.linalg.norm(points2[:2, :] - fwd_proj[:2, :], axis=0)**2
    error2 = np.linalg.norm(points1[:2, :] - bwd_proj[:2, :], axis=0)**2
    error = np.sqrt(error1 + error2)
    
    # Inliers are those with error below threshold th
    idx = np.where(error < th)[0]

    return idx


def Ransac_DLT_homography(points1, points2, th, max_it):
    
    Ncoords, Npts = points1.shape
    
    it = 0
    best_inliers = np.empty(1)
    
    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:,indices], points2[:,indices])
        inliers = Inliers(H, points1, points2, th)
        
        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
        
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        
        it += 1
    
    # compute H from all the inliers
    H = DLT_homography(points1[:,best_inliers], points2[:,best_inliers])
    inliers = best_inliers
    
    return H, inliers



def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, w, h, fig, legend):
    
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return

def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])
    
    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return
