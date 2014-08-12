# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["get_kfold_train_mask"]

import os
import logging
import numpy as np
from multiprocessing import Pool

from .data import load_data
from .solvers import linear_least_squares


def get_kfold_train_mask(length, k, rand=False):
    """
    TODO: document this function.

    """
    train_mask = np.ones(length, dtype=int)
    if rand:
        for i in range(0, length):
            group = np.random.randint(0, k-1)
            train_mask[i] = group
    else:
        step = length//k
        for i in range(0, k-1):
            train_mask[i*step:(i+1)*step] = i
        train_mask[(k-1)*step:] = k-1
    return train_mask


def get_fit_matrix(target_tpf, neighbor_tpfs, poly=0, auto=False, offset=0,
                   window=0):
    """
    ## inputs:
    - `target_tpf` - target tpf
    - `neighbor_tpfs` - neighbor tpfs in magnitude
    - `auto` - if autorgression
    - `poly` - number of orders of polynomials of time need to be added

    ## outputs:
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
    - `target_flux` - target flux
    - `covar_list` - covariance matrix for every pixel
    - `time` - one dimension array of BKJD time
    - `neighbor_kid` - KIC number of the neighbor stars in the fitting matrix
    - `neighbor_kplr_maskes` - kepler maskes of the neighbor stars in the
                fitting matrix
    - `target_kplr_mask` - kepler mask of the target star
    - `epoch_mask` - epoch mask
    """

    # Load the target TPF file.
    (time, target_flux, target_pixel_mask, target_kplr_mask, epoch_mask,
        flux_err) = load_data(target_tpf)

    # Loop over the predictor TPFs and load each one.
    neighbor_kid, neighbor_fluxes = [], []
    neighbor_pixel_maskes, neighbor_kplr_maskes = [], []
    for key, tpf in neighbor_tpfs.iteritems():
        neighbor_kid.append(key)
        tmpResult = load_data(tpf)
        neighbor_fluxes.append(tmpResult[1])
        neighbor_pixel_maskes.append(tmpResult[2])
        neighbor_kplr_maskes.append(tmpResult[3])
        epoch_mask *= tmpResult[4]

    # Remove times where the data are bad on the predictor pixels.
    time = time[epoch_mask > 0]
    target_flux = target_flux[epoch_mask > 0]
    flux_err = flux_err[epoch_mask > 0]
    target_flux_var = flux_err ** 2

    # We shouldn't need to construct the full dense matrix.
    # covar_list = np.zeros((flux_err.shape[1], flux_err.shape[0],
    #                        flux_err.shape[0]))
    # for i in range(0, flux_err.shape[1]):
    #     for j in range(0, flux_err.shape[0]):
    #         covar_list[i, j, j] = flux_err[j][i]

    # Construct the neighbor flux matrix
    neighbor_flux_matrix = np.concatenate(neighbor_fluxes, axis=1)
    neighbor_flux_matrix = neighbor_flux_matrix[:, epoch_mask > 0]

    # FIXME: Why do we need this?
    neighbor_flux_matrix = neighbor_flux_matrix.astype(float)
    target_flux = target_flux.astype(float)

    # The previous lines should do this already.
    # for i in range(0, len(neighbor_fluxes)):
    #     neighbor_fluxes[i] = neighbor_fluxes[i][epoch_mask > 0, :]

    logging.info("The baseline predictor flux matrix has the shape: {0}"
                 .format(neighbor_flux_matrix.shape))

    # Add autoregression terms.
    # FIXME: this *won't* work! What is "pixel"?
    if auto:
        raise NotImplementedError("There is a bug in the auto matrix "
                                  "construction")
        epoch_len = epoch_mask.shape[0]
        auto_flux = np.zeros(epoch_len)
        auto_flux[epoch_mask > 0] = target_flux[:, pixel]
        auto_pixel = np.zeros((epoch_len, 2*window))
        for i in range(offset+window, epoch_len-window-offset):
            auto_pixel[i, 0:window] = auto_flux[i-offset-window:i-offset]
            auto_pixel[i, window:2*window] = auto_flux[i+offset+1:
                                                       i+offset+window+1]
        for i in range(0, offset+window):
            auto_pixel[i, window:2*window] = auto_flux[i+offset+1:
                                                       i+offset+window+1]
        for i in range(epoch_len-window-offset, epoch_len):
            auto_pixel[i, 0:window] = auto_flux[i-offset-window:i-offset]
        auto_pixel = auto_pixel[epoch_mask > 0, :]
        neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix,
                                               auto_pixel), axis=1)

    # Add the polynomial (t^n) terms.
    # Note: the order of `vander` is reversed compared to `polyvander`.
    time_mean = np.mean(time)
    time_std = np.std(time)
    nor_time = (time-time_mean)/time_std
    p = np.vander(nor_time, poly + 1)
    neighbor_flux_matrix = np.concatenate((neighbor_flux_matrix, p), axis=1)

    logging.info("The final predictor flux matrix has the shape: {0}"
                 .format(neighbor_flux_matrix.shape))

    return (neighbor_flux_matrix, target_flux, target_flux_var, time,
            neighbor_kid, neighbor_kplr_maskes, target_kplr_mask, epoch_mask)


def fit_epoch(args):
    target_flux, target_flux_var, neighbor_flux_matrix, l2 = args
    w = np.empty((target_flux.shape[1], neighbor_flux_matrix.shape[1]))
    for i in range(target_flux.shape[1]):
        w[i, :] = linear_least_squares(neighbor_flux_matrix, target_flux[:, i],
                                       yvar=target_flux_var[:, i], l2=l2)
    return w


def fit_target(target_flux, target_kplr_mask, neighbor_flux_matrix, time,
               epoch_mask, target_flux_var, margin, poly, l2, thread_num,
               prefix):
    """
    ## inputs:
    - `target_flux` - target flux
    - `target_kplr_mask` - kepler mask of the target star
    - `neighbor_flux_matrix` - fitting matrix of neighbor flux
    - `time` - array of time
    - `epoch_mask` - epoch mask
    - `target_flux_var` - the observational variance
    - `margin` - half-width of the test region (in days)
    - `poly` - number of orders of polynomials of time(zero order is the
      constant level)
    - `l2` - strenght of L2 regularization strength
    - `thread_num` - thread number
    - `prefix` - output file's prefix

    ## outputs:
    - .npy file - fitting fluxes of pixels

    """
    target_kplr_mask = target_kplr_mask.flatten()
    target_kplr_mask = target_kplr_mask[target_kplr_mask > 0]

    # Only fit the pixels in the "optimal" aperture.
    optimal = target_kplr_mask == 3
    optimal_len = np.sum(optimal)
    target_flux = target_flux[:, optimal]
    logging.info("Fitting {0} 'optimal' pixels".format(optimal_len))
    target_flux_var = target_flux_var[optimal]

    # Compute the train-and-test masks.
    masks = (np.abs(t - time) < margin for t in time)
    args = ((target_flux[m], target_flux_var[m], neighbor_flux_matrix[m], l2)
            for m in masks)

    # Pre-compute the L2 regularization vector setting the polynomial
    # regularization to zero.
    l2 = l2 + np.zeros(neighbor_flux_matrix.shape[1])
    l2[-poly-1:] = 0.0

    # Compute the weights in parallel.
    pool = Pool()
    weights = pool.map(fit_epoch, args)
    pool.terminate()

    print(weights)
    print(len(weights))
    print(weights[0].shape)
    assert 0

    # Make sure that the basepath for output exists.
    filename = os.path.join(".", prefix)
    basedir = os.path.dirname(filename)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    covar = np.mean(covar_list, axis=0)**2
    fit_flux = []
    fit_coe = []
    length = target_flux.shape[0]
    total_length = epoch_mask.shape[0]

    thread_len = total_length//thread_num
    last_len = total_length - (thread_num-1)*thread_len

    class fit_epoch(threading.Thread):
        def __init__(self, thread_id, initial, len, time_initial, time_len):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            self.initial = initial
            self.len = len
            self.time_initial = time_initial
            self.time_len = time_len
        def run(self):
            print('Starting%d'%self.thread_id)
            print (self.thread_id , self.time_initial, self.time_len)
            tmp_fit_flux = np.empty((self.time_len, optimal_len))
            time_stp = 0
            for i in range(self.initial, self.initial+self.len):
                if epoch_mask[i] == 0:
                    continue
                train_mask = np.ones(total_length)
                if i<margin:
                    train_mask[0:i+margin+1] = 0
                elif i > total_length-margin-1:
                    train_mask[i-margin:] = 0
                else:
                    train_mask[i-margin:i+margin+1] = 0
                train_mask = train_mask[epoch_mask>0]

                covar_mask = np.ones((length, length))
                covar_mask[train_mask==0, :] = 0
                covar_mask[:, train_mask==0] = 0

                tmp_covar = covar[covar_mask>0]
                train_length = np.sum(train_mask, axis=0)
                tmp_covar = tmp_covar.reshape(train_length, train_length)
                result = lss.leastSquareSolve(neighor_flux_matrix[train_mask>0], target_flux[train_mask>0], tmp_covar, l2, False, poly)[0]
                tmp_fit_flux[time_stp, :] = np.dot(neighor_flux_matrix[time_stp+self.time_initial, :], result)
                np.save('./%stmp%d.npy'%(prefix, self.thread_id), tmp_fit_flux)
                time_stp += 1
                print('done%d'%i)
            print('Exiting%d'%self.thread_id)

    thread_list = []
    time_initial = 0
    for i in range(0, thread_num-1):
        initial = i*thread_len
        thread_epoch = epoch_mask[initial:initial+thread_len]
        time_len = np.sum(thread_epoch)
        thread = fit_epoch(i, initial, thread_len, time_initial, time_len)
        thread.start()
        thread_list.append(thread)
        time_initial += time_len

    initial = (thread_num-1)*thread_len
    thread_epoch = epoch_mask[initial:initial+last_len]
    time_len = np.sum(thread_epoch)
    thread = fit_epoch(thread_num-1, initial, last_len, time_initial, time_len)
    thread.start()
    thread_list.append(thread)

    for t in thread_list:
        t.join()
    print('all done')

    offset = 0
    window = 0

    for i in range(0, thread_num):
        tmp_fit_flux = np.load('./%stmp%d.npy'%(prefix, i))
        if i==0:
            fit_flux = tmp_fit_flux
        else:
            fit_flux = np.concatenate((fit_flux, tmp_fit_flux), axis=0)
    np.save('./%s.npy'%prefix, fit_flux)

    for i in range(0, thread_num):
        os.remove('./%stmp%d.npy'%(prefix, i))
