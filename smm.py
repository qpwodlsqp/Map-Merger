import numpy as np
import cv2
from scipy.signal import argrelextrema


# Alternative implementations of Hough Spectrum Transformations
# Confirmed that they output equivalent spectrums but only differ in complexity
# But not gonna depricate them, because cost performance can be changed in robot experiments
def hough_transform_iter(M, theta_res, rho_res):

    theta_mat = np.deg2rad(np.linspace(-180, 180, theta_res + 1))
    H, W = M.shape # Assume 2d single image
    rho_bound = np.sqrt(H ** 2 + W ** 2)
    rho_mat = np.linspace(-rho_bound, rho_bound, rho_res + 1)

    hough_space = np.zeros((rho_res+1, theta_res+1))
    x_idx, y_idx = np.where(M == 0) # Find x, y coordiantes where occupied
    if x_idx.size == 0 or y_idx.size == 0:
        return np.zeros(theta_res+1)
    for x, y in zip(x_idx, y_idx):
        ros_x = map_size//2 - x; ros_y = map_size//2 - y
        for theta_idx, theta in enumerate(theta_mat):
            tmp_rho = ros_x*np.cos(theta) + ros_y*np.sin(theta)
            diff = np.abs(rho_mat - tmp_rho)
            hough_space[np.argmin(diff), theta_idx] += 1
    hough_spectrum = np.power(hough_space, 2).sum(axis=0)
    return hough_spectrum / np.max(hough_spectrum)

def hough_transform_tensor(M, theta_res, rho_res, map_size):

    theta_mat = np.deg2rad(np.linspace(-180, 180, theta_res + 1))
    H, W = M.shape # Assume 2d single image
    rho_bound = np.sqrt(H ** 2 + W ** 2)
    rho_mat = np.linspace(-rho_bound, rho_bound, rho_res + 1)

    x_idx, y_idx = np.where(M == 0) # Find x, y coordiantes where occupied
    if x_idx.size == 0 or y_idx.size == 0:
        return np.zeros(theta_res+1)
    hough_space = np.zeros((len(x_idx), theta_res+1, rho_res+1))

    # (occupied_num, theta_res+1)
    x_idx = x_idx.reshape((len(x_idx), 1)); x_idx = map_size//2 - x_idx
    y_idx = y_idx.reshape((len(y_idx), 1)); y_idx = map_size//2 - y_idx
    tmp_rho = x_idx*np.cos(theta_mat) + y_idx*np.sin(theta_mat)
    # argmin of (occupied_num, theta_res+1, rho_res+1) in axis=2 => (occupied_num, theta_res)
    diff = np.abs(np.expand_dims(tmp_rho, -1) - rho_mat)
    rho_idx = np.argmin(diff, axis=2)
    hough_space[np.hstack([np.arange(0, len(x_idx))] * (theta_res+1)),
                np.hstack([np.arange(0, theta_res+1)] * len(x_idx)),
                rho_idx.flatten()] = 1
    hough_spectrum = np.power(hough_space.sum(axis=0), 2).sum(axis=-1)
    return hough_spectrum / np.max(hough_spectrum)

def hough_transform(M, theta_res, rho_res, map_size):

    theta_mat = np.deg2rad(np.linspace(-180, 180, theta_res + 1))
    H, W = M.shape # Assume 2d single image
    rho_bound = int(np.sqrt(H ** 2 + W ** 2))//2
    rho_mat = np.linspace(-rho_bound, rho_bound, rho_res + 1)

    hough_space = np.zeros((rho_res+1, theta_res+1))
    x_idx, y_idx = np.where(M == 0) # Find x, y coordiantes where occupied
    if x_idx.size == 0 or y_idx.size == 0:
        return np.zeros(theta_res+1)
    # Transform indices to ROS space coordinates
    for x, y in zip(x_idx, y_idx):
        ros_x = map_size//2 - x; ros_y = map_size//2 - y
        tmp_rho = ros_x*np.cos(theta_mat) + ros_y*np.sin(theta_mat)
        tmp_rho = np.expand_dims(tmp_rho, 0)
        rho = np.expand_dims(rho_mat, 1)
        diff = np.abs(rho - tmp_rho)
        hough_space[np.argmin(diff, axis=0), np.arange(0, theta_res+1)] += 1
    hough_spectrum = np.power(hough_space, 2).sum(axis=0)
    return hough_spectrum / np.max(hough_spectrum)


def circ_cross_correl(hs_0, hs_1, res=360, const=30, order=10, max_only=False):

    corr_list = np.zeros(res+1)
    for i in range(res+1):
        if i != 0:
            hs_1_tmp = np.concatenate((hs_1[-i:], hs_1[i:]))
            corr = np.mean(hs_0 * hs_1_tmp)
        else:
            corr = np.mean(hs_0 * hs_1)
        corr_list[i] = corr
    if max_only:
        local_maxima = np.argmax(corr_list[res//2 - const:res//2 + const])
    else:
        local_maxima = argrelextrema(corr_list[res//2 - const:res//2 + const], np.greater, order=order)
        local_maxima = np.array(local_maxima)
    if local_maxima.size == 0:
        return np.zeros(1)
    local_maxima -= const
    return local_maxima.flatten()


def trans_correl(s_0, s_1, res=384, const=40):

    corr_list = np.zeros(res+1)
    for i in range(res+1):
        if i <= res//2:
            #s_1 = np.concatenate([s_1[res//2-i:], np.zeros(res - res//2 - i)])
            s_1_tmp = np.concatenate([s_1, np.zeros(res-res//2-i)])[res//2-i:]
        else:
            s_1_tmp = np.concatenate([np.zeros(i-res//2), s_1])[:res//2-i]
        corr = (s_0 * s_1_tmp).sum()
        corr_list[i] = corr
    optimum = np.argmax(corr_list[res//2 - const : res//2 + const])
    if optimum.size == 0:
        return np.zeros(1)
    return optimum - const


def XY_spectrum(M):

    '''
     |- - -> y
    x|
     v

     to
         x
         ^
         |
    y<---|
    '''

    M = (M == 0.0)
    sx = M.sum(axis=1)[::-1]
    sy = M.sum(axis=0)[::-1]
    #return sx[::-1], sy[::-1]
    return sx / sx.max(), sy / sy.max()


def _discretize(M):

    empty = np.where(M >= 0.805, 1., 0.)
    unknown = np.where(np.logical_and(M < 0.805, M > 0.3), 0.5, 0)
    return empty + unknown

def merge(cam, tar):

    gray = (cam == 0.5) & (tar == 0.5)
    black = (cam == 0.0) | (tar == 0.0)

    merged = np.ones(cam.shape)
    gray_mask = 0.5 * np.ones(cam.shape)
    black_mask = np.zeros(cam.shape)

    merged = np.where(gray, gray_mask, merged)
    merged = np.where(black, black_mask, merged)
    return merged


def map_rotation(M, theta):

    H, W = M.shape
    T = cv2.getRotationMatrix2D((W//2, H//2), int(theta), 1.0)
    return cv2.warpAffine(M, T, M.shape, borderValue=0.5, flags=0)


def map_translation(M, tx, ty):

    T = np.float32([[1, 0, -ty], [0, 1, -tx]])
    return cv2.warpAffine(M, T, M.shape, borderValue=0.5, flags=0)


def map_consistency(M_0, M_1):

    agree = np.logical_and(M_0==0.0, M_1==0.0).sum() + np.logical_and(M_0==1.0, M_1==1.0).sum()
    if agree == 0:
        return 0.
    else:
        disagree = np.logical_and(M_0==0.0, M_1==1.0).sum() + np.logical_and(M_0==1.0, M_1==0.0).sum()
        return agree / (agree+disagree)


def SMM(M_0, M_1, theta_res=360, rho_res=360, map_size=128):

    M_0 = _discretize(M_0.astype(np.float64)/255)
    M_1 = _discretize(M_1.astype(np.float64)/255)
    HS_0 = hough_transform(M_0, theta_res, rho_res, map_size)
    HS_1 = hough_transform(M_1, theta_res, rho_res, map_size)
    theta_local_maxima = circ_cross_correl(HS_0, HS_1, res=360, const=30)
    # Align to axes
    align_angle = np.argmax(HS_0) - theta_res//2
    M_0 = map_rotation(M_0, -align_angle)
    M_1 = map_rotation(M_1, -align_angle)
    # XY spectrum
    sx_0, sy_0 = XY_spectrum(M_0)

    best_w = np.iinfo(np.int16).min
    best_M_2 = None
    best_tx = 0.
    best_ty = 0.
    best_theta = 0.
    for theta in theta_local_maxima:
        M_2 = map_rotation(M_1, theta)
        sx_1, sy_1 = XY_spectrum(M_2)
        # These are ROS coordinates
        tx = trans_correl(sx_0, sx_1, res=map_size, const=map_size//6)
        ty = trans_correl(sy_0, sy_1, res=map_size, const=map_size//6)
        M_2 = map_translation(M_2, tx, ty) # ROS space and CV2 space axes have opposite sign
        w = map_consistency(M_0, M_2)
        if best_w < w:
            best_w = w
            best_M_2 = M_2
            best_theta = theta
            best_tx = tx
            best_ty = ty
    T3 = np.array([[np.cos(-align_angle/180*np.pi), -np.sin(-align_angle/180*np.pi), 0.], [np.sin(-align_angle/180*np.pi), np.cos(-align_angle/180*np.pi), 0.], [0., 0., 1.]])
    T0 = np.array([[np.cos(best_theta/180*np.pi), -np.sin(best_theta/180*np.pi), best_tx * 128 / 384], [np.sin(best_theta/180*np.pi), np.cos(best_theta/180*np.pi), best_ty * 128 / 384], [0., 0., 1.]])
    #T1 = np.array([[1., 0., -ty], [0., 1., -tx], [0., 0., 1.]])
    T2 = np.array([[np.cos(align_angle/180*np.pi), -np.sin(align_angle/180*np.pi), 0.], [np.sin(align_angle/180*np.pi), np.cos(align_angle/180*np.pi), 0.], [0., 0., 1.]])
    T = T2 @ T0 @ T3
    T = np.linalg.inv(T)
    return (map_rotation(merge(M_0, best_M_2), align_angle)*255).astype(np.uint8), (-T[1][2], -T[0][2], -np.arctan2(T[1][0], T[1][1]) / np.pi * 180) # Match on pose_delta in torch/PIL space


