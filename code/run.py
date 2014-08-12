#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import plm
import logging

logging.basicConfig(level=logging.DEBUG)


# Locate all the required data.
kid = 5088536
quarter = 5
num = 200
offset = 0
ccd = True
target, neighbors = plm.find_mag_neighbor(kid, quarter, num, offset=offset,
                                          ccd=ccd)

# Do the fit.
margin = 6.
poly = 0
l2 = 1e5
predictions = plm.fit_target(target, neighbors, margin, poly, l2)
