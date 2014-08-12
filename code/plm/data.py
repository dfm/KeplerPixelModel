# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["find_mag_neighbor"]

import kplr
import logging
import numpy as np


# A connection to the kplr interface.
client = kplr.API()


def find_mag_neighbor(kic, quarter, num, offset=0, ccd=True):
    """
    ## inputs:
    - `kic` - target KIC number
    - `quarter` - target quarter
    - `num` - number of tpfs needed
    - `offset` - number of tpfs that are excluded
    - `ccd` - if the tpfs need to be on the same CCD

    ## outputs:
    - `target_tpf` - tpf of the target star
    - `tpfs` - tpfs of stars that are closet to the target star in magnitude
    """

    # Find the target target pixel file.
    target_tpf = client.target_pixel_files(ktc_kepler_id=kic,
                                           sci_data_quarter=quarter,
                                           ktc_target_type="LC",
                                           max_records=1)[0]

    # Build the base query to find the predictor stars.
    base_args = dict(
        ktc_kepler_id="!={0:d}".format(target_tpf.ktc_kepler_id),
        sci_data_quarter=target_tpf.sci_data_quarter,
        ktc_target_type="LC",
        max_records=num+offset,
    )
    if ccd:
        base_args["sci_channel"] = target_tpf.sci_channel
    else:
        base_args["sci_channel"] = "!={0}".format(target_tpf.sci_channel)

    # Construct the bracketing queries.
    over_args = dict(
        kic_kepmag=">={0:f}".format(target_tpf.kic_kepmag),
        sort=("kic_kepmag", 1),
        **base_args
    )
    under_args = dict(
        kic_kepmag="<={0:f}".format(target_tpf.kic_kepmag),
        sort=("kic_kepmag", -1),
        **base_args
    )

    # Execute the queries to find the predictor TPFs.
    stars_over = client.target_pixel_files(**over_args)
    stars_under = client.target_pixel_files(**under_args)
    logging.info("Found {0} brighter / {1} fainter TPFs."
                 .format(len(stars_under), len(stars_over)))

    # Loop over the predictor stars and compute the magnitude differences.
    dtype = [('kic', int), ('bias', float), ('tpf', type(target_tpf))]
    neighbor_list = []
    tpf_list = stars_over+stars_under
    target_kepmag = target_tpf.kic_kepmag
    for tpf in tpf_list:
        neighbor_list.append((tpf.ktc_kepler_id,
                             np.fabs(tpf.kic_kepmag-target_kepmag), tpf))

    # Sort that list and extract only the targets that we want.
    neighbor_list = np.array(neighbor_list, dtype=dtype)
    neighbor_list = np.sort(neighbor_list, kind='mergesort', order='bias')
    tpfs = {}
    for i in range(offset, offset+num):
        tmp_kic, tmp_bias, tmp_tpf = neighbor_list[i]
        tpfs[tmp_kic] = tmp_tpf

    return target_tpf, tpfs


def get_pixel_mask(flux, kplr_mask):
    """
    Helper function to find the pixel mask

    """
    pixel_mask = np.zeros(flux.shape)
    pixel_mask[np.isfinite(flux)] = 1  # okay if finite
    pixel_mask[:, (kplr_mask < 1)] = 0  # unless masked by kplr
    return pixel_mask


def get_epoch_mask(pixel_mask):
    """
    Helper function to find the epoch mask

    """
    foo = np.sum(np.sum((pixel_mask > 0), axis=2), axis=1)
    epoch_mask = np.zeros_like(foo)
    epoch_mask[(foo > 0)] = 1
    return epoch_mask


def load_data(tpf):
    """
    Helper function to load data from TPF object.

    TODO: document the outputs.

    """
    kplr_mask, time, flux, flux_err = [], [], [], []
    with tpf.open() as file:
        hdu_data = file[1].data
        kplr_mask = file[2].data
        time = hdu_data["time"]
        flux = hdu_data["flux"]
        flux_err = hdu_data["flux_err"]
    pixel_mask = get_pixel_mask(flux, kplr_mask)
    epoch_mask = get_epoch_mask(pixel_mask)
    flux = flux[:, kplr_mask > 0]
    flux_err = flux_err[:, kplr_mask > 0]

    flux = flux.reshape((flux.shape[0], -1))
    flux_err = flux_err.reshape((flux.shape[0], -1))

    # Interpolate the bad points
    for i in range(flux.shape[1]):
        interMask = np.isfinite(flux[:, i])
        flux[~interMask, i] = np.interp(time[~interMask], time[interMask],
                                        flux[interMask, i])
        flux_err[~interMask, i] = np.inf

    return time, flux, pixel_mask, kplr_mask, epoch_mask, flux_err
