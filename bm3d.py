#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Peng Lin 2021-05-15 14:54:55
from utils import add_gaussian_noise, symetrize
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr
from noise_estimation import noise_estimate
import os
import cv2
import numpy as np
from skimage import img_as_float


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised


if __name__ == '__main__':
    # <hyper parameter> -------------------------------------------------------------------------------
    n_H = 16
    k_H = 8
    N_H = 16
    p_H = 3
    lambda3D_H = 2.7  # ! Threshold for Hard Thresholding
    useSD_H = False
    tau_2D_H = 'BIOR'

    n_W = 16
    k_W = 8
    N_W = 32
    p_W = 3
    useSD_W = True
    tau_2D_W = 'DCT'
    # <\ hyper parameter> -----------------------------------------------------------------------------

    im_dir = './data/lena.png'
    save_dir = 'test_result'
    os.makedirs(save_dir, exist_ok=True)
    sigma_list = [20]
    for sigma in sigma_list:
        tauMatch_H = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
        tauMatch_W = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches

        im = cv2.imread(im_dir, cv2.IMREAD_GRAYSCALE)
        im = img_as_float(im)
        noisy_im = im + np.random.randn(*im.shape) * sigma / 255
        est_level = noise_estimate(noisy_im, 8)
        est_sigma = est_level * 255

        img_noise = noisy_im * 255
        img_noise = img_noise.astype(np.uint8)
        cv2.imwrite('./data/lena_sigma%d.png' % sigma, img_noise)

        im1, im2 = run_bm3d(img_noise, est_sigma,
                            n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                            n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

        psnr_1st = compute_psnr(im, im1)
        psnr_2nd = compute_psnr(im, im2)

        im1 = (np.clip(im1, 0, 255)).astype(np.uint8)
        im2 = (np.clip(im2, 0, 255)).astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, '1st.png'), im1)
        cv2.imwrite(os.path.join(save_dir, '2nd.png'), im2)
