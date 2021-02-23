# Copyright (C) 2021 Weixuan Zhang, Tim Clifford
#
# SPDX-License-Identifier: MIT
"""Sensors used on the robot"""
from controller import GPS, Compass, DistanceSensor


class IDPCompass(Compass):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


class IDPDistanceSensor(DistanceSensor):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)
        self.max_range = self.getLookupTable()[-3]


class IDPGPS(GPS):
    def __init__(self, name, sampling_rate):
        super().__init__(name)  # default to infinite resolution
        if self.getCoordinateSystem() != 0:
            raise RuntimeWarning('Need to set GPS coordinate system in WorldInfo to local')
        self.enable(sampling_rate)
