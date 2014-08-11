# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["find_mag_neighbor"]

import kplr


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

    tpfs = {}

    i=0
    j=0
    offset_list =[]
    while len(tpfs) <num+offset:
        while stars_over[i].ktc_kepler_id in tpfs:
            i+=1
        tmp_over = stars_over[i]
        while stars_under[j].ktc_kepler_id in tpfs:
            j+=1
        tmp_under = stars_under[j]
        if tmp_over.kic_kepmag-target_tpf.kic_kepmag > target_tpf.kic_kepmag-tmp_under.kic_kepmag:
            tpfs[tmp_under.ktc_kepler_id] = tmp_under
            j+=1
            if len(tpfs)>offset:
                pass
            else:
                offset_list.append(tmp_under.ktc_kepler_id)
        elif tmp_over.kic_kepmag-target_tpf.kic_kepmag < target_tpf.kic_kepmag-tmp_under.kic_kepmag:
            tpfs[tmp_over.ktc_kepler_id] = tmp_over
            i+=1
            if len(tpfs)>offset:
                pass
            else:
                offset_list.append(tmp_over.ktc_kepler_id)
        elif len(tpfs) < num+offset-1:
            tpfs[tmp_under.ktc_kepler_id] = tmp_under
            tpfs[tmp_over.ktc_kepler_id] = tmp_over
            i+=1
            j+=1
            if len(tpfs)>offset+1:
                pass
            elif len(tpfs) == offset+1:
                offset_list.append(tmp_under.ktc_kepler_id)
            else:
                offset_list.append(tmp_over.ktc_kepler_id)
                offset_list.append(tmp_under.ktc_kepler_id)
        else:
            tpfs[tmp_over.ktc_kepler_id] = tmp_over
            i+=1
            if len(tpfs)>offset:
                pass
            else:
                offset_list.append(tmp_over.ktc_kepler_id)

    for key in offset_list:
        tpfs.pop(key)

    return target_tpf, tpfs
