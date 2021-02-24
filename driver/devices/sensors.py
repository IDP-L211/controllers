# Copyright (C) 2021 Weixuan Zhang, Eleanor Clifford
#
# SPDX-License-Identifier: MIT
"""Sensors used on the robot"""
from controller import GPS, Compass, DistanceSensor


class IDPCompass(Compass):
    def __init__(self, name, sampling_rate):
        super().__init__(name)
        self.enable(sampling_rate)


class IDPDistanceSensor(DistanceSensor):
    def __init__(self, name, sampling_rate, decreasing=False, min_range=0):
        super().__init__(name)
        self.enable(sampling_rate)
        # We could approximate the min range and whether it's decreasing from
        # the data, but not worth it imo
        self.min_range = min_range
        self.max_range = self.getLookupTable()[-3]
        self.decreasing = decreasing

        lookupTable = self.getLookupTable()
        betterLookupTable = list(zip(*[
            lookupTable[i:i + 3] for i in range(0, len(lookupTable) - 2, 3)
            if lookupTable[i] >= min_range  # prevent it being many-to-one
        ]))

        # For now we will ignore the uncertainty, but we should add it later
        self.inverseLookupTable = betterLookupTable[1::-1]  # First two reversed

    def getValue(self):
        # We need some context about the response of the sensor in order to
        # decode it properly
        v = super().getValue()

        # Don't attempt to interpolate if the value is outside the bounds
        if (v > max(self.inverseLookupTable[0][0], self.inverseLookupTable[0][-1]) or
                v < min(self.inverseLookupTable[0][0], self.inverseLookupTable[0][-1])):
            return self.max_range

        # interpolate the inverse lookup table
        idx, x = next(filter(
            lambda x: x[1] < v if self.decreasing else x[1] > v,
            enumerate(self.inverseLookupTable[0])
        ))

        fractional_position = abs(
            (self.inverseLookupTable[0][idx - 1] - v)
            / (self.inverseLookupTable[0][idx - 1] - x)
        )
        return (
                self.inverseLookupTable[1][idx - 1] + fractional_position
                * (self.inverseLookupTable[1][idx] - self.inverseLookupTable[1][idx - 1])
        )


class IDPGPS(GPS):
    def __init__(self, name, sampling_rate):
        super().__init__(name)  # default to infinite resolution
        if self.getCoordinateSystem() != 0:
            raise RuntimeWarning('Need to set GPS coordinate system in WorldInfo to local')
        self.enable(sampling_rate)
