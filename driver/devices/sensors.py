# Copyright (C) 2021 Weixuan Zhang, Tim Clifford
#
# SPDX-License-Identifier: MIT
"""Sensors used on the robot"""
from controller import GPS, Compass, DistanceSensor
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar as root
from warnings import warn

DEBUG=True

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
        self.betterLookupTable = list(zip(*(
            lookupTable[i:i + 3] for i in range(0, len(lookupTable) - 2, 3)
            if lookupTable[i] >= min_range  # prevent it being many-to-one
        )))

        # Be VERY careful trying to refactor this
        # Bound function at start and end values

        # |  This helps you understand things, ok?
        # |     -
        # |     x
        # |     -   -
        # |   -     x   -
        # |   x     -   x   -
        # |             -   x     -
        # |  -              -     x        -
        # |  x                    -        x
        # |  -                             -
        # |--------------------------------

        self.f_expectation = lambda x: interp1d(
            *self.betterLookupTable[:2],
            bounds_error=False,
            fill_value=self.betterLookupTable[1][-1]
        )(x) if x > self.min_range else self.betterLookupTable[1][0]

        self.f_upper_bound = lambda x: interp1d(
            self.betterLookupTable[0],
            tuple(map(sum, zip(*self.betterLookupTable[1:]))),
            bounds_error=False,
            fill_value=tuple(
                map(sum, zip(*self.betterLookupTable[1:]))
            )[-1]
        )(x) if x > self.min_range else tuple(
            map(sum, zip(*self.betterLookupTable[1:]))
        )[0]

        self.f_lower_bound = lambda x: interp1d(
            self.betterLookupTable[0],
            tuple(map(lambda x: x[0] - x[1], zip(*self.betterLookupTable[1:]))),
            bounds_error=False,
            fill_value=tuple(
                map(lambda x: x[0] - x[1], zip(*self.betterLookupTable[1:]))
            )[-1]
        )(x) if x > self.min_range else tuple(
            map(lambda x: x[0] - x[1], zip(*self.betterLookupTable[1:]))
        )[0]


    def getValue(self):
        v = super().getValue()

        # Don't attempt to interpolate if the value is outside the bounds
        if (v > max(self.betterLookupTable[1][0], self.betterLookupTable[1][-1]) or
                v < min(self.betterLookupTable[1][0], self.betterLookupTable[1][-1])):
            return self.max_range

        try:
            sc_expectation_result = root(
                lambda x: self.f_expectation(x) - v,
                method = 'bisect',
                bracket = sorted((
                    self.betterLookupTable[0][0],
                    self.betterLookupTable[0][-1],
                )), x0 = (self.betterLookupTable[1][len(self.betterLookupTable)//2],),
            )
        except ValueError as e:
            # Shouldn't happen
            warn(f'WARNING: value out of bounds in {self.getName()}')
            if "f(a) and f(b) must have different signs" in str(e):
                return self.max_range
            else: raise e

        if not sc_expectation_result.converged:
            warn(f'WARNING: Could not calculate {self.getName()} value')
            return self.max_range

        # it shouldn't have more than one root
        return sc_expectation_result.root


    def getBounds(self):
        v = super().getValue()

        # Don't attempt to interpolate if the value is outside the bounds
        try:
            sc_bound_results = (
                root(
                    lambda x: self.f_lower_bound(x) - v,
                    method = 'bisect',
                    bracket = sorted((
                        self.betterLookupTable[0][0],
                        self.betterLookupTable[0][-1],
                    )), x0 = (self.getValue(),)
                ), root(
                    lambda x: self.f_upper_bound(x) - v,
                    method = 'bisect',
                    bracket = sorted((
                        self.betterLookupTable[0][0],
                        self.betterLookupTable[0][-1],
                    )), x0 = (self.getValue(),)
                )
            )
        except ValueError as e:
            # Doing the check ourselves here is actually not that simple
            # because the function could be increasing or decreasing
            # let's just rely on the ValueError
            if "f(a) and f(b) must have different signs" in str(e):
                return (self.max_range, 3.5) # Roughly the maximum possible
            else: raise e

        if any(map(lambda x: not x.converged, sc_bound_results)):
            warn(f'WARNING: Could not calculate {self.getName()} bounds')
            return (self.max_range, 3.5) # Roughly the maximum possible
        else:
            # it shouldn't have more than one root
            return sorted(tuple(map(lambda y: y.root, sc_bound_results)))


class IDPGPS(GPS):
    def __init__(self, name, sampling_rate):
        super().__init__(name)  # default to infinite resolution
        if self.getCoordinateSystem() != 0:
            raise RuntimeWarning('Need to set GPS coordinate system in WorldInfo to local')
        self.enable(sampling_rate)
