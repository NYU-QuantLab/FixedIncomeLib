import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ExtrapMethod(Enum):
    
    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class Interpolator1D(ABC):

    def __init__(self,
                 axis1 : np.ndarray, 
                 values : np.ndarray, 
                 interpolation_method : InterpMethod,
                 extrpolation_method : ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.length_ = len(self.axis1)

    @abstractmethod
    def interpolate(self, x : float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x : float, end_x : float):
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x : float):
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()

class Interpolator1DPCP(Interpolator1D):

    def __init__(self, axis1: np.ndarray, values: np.ndarray, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.LINEAR, extrpolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        if x <= self.axis1_[0]:
            return self.values_[0]  # flat extrapolation
        elif x >= self.axis1_[-1]:
            return self.values_[-1]  # flat extrapolation
        else:
            idx = np.searchsorted(self.axis1_, x, side='right') - 1
            return self.values_[idx]

    def integrate(self, start_x: float, end_x: float) -> float:

        if start_x > end_x:
            start_x, end_x = end_x, start_x
            sign = -1
        else:
            sign = 1

        integral = 0.0
        
        # Loop over all intervals including extrapolation zones
        for i in range(self.length_):
            interval_start = self.axis1_[i]
            interval_end = self.axis1_[i+1] if i+1 < self.length_ else np.inf
            
            # For the very first interval, allow left extrapolation
            if i == 0:
                interval_start = -np.inf
            
            # Compute overlap with [start_x, end_x]
            overlap_start = max(start_x, interval_start)
            overlap_end = min(end_x, interval_end)
            
            if overlap_end > overlap_start:
                integral += (overlap_end - overlap_start) * self.values_[i]

        return sign * integral

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        grad = np.zeros_like(self.values_)
        
        if x <= self.axis1_[0]:
            grad[0] = 1.0
        elif x >= self.axis1_[-1]:
            grad[-1] = 1.0
        else:
            idx = np.searchsorted(self.axis1_, x, side='right') - 1
            grad[idx] = 1.0

        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        n = len(self.values_)
        grad = np.zeros(n, dtype=float)

        # ---- left tail: (-inf, axis1[0]] -> values[0]
        left_overlap = min(end_x, self.axis1_[0]) - start_x
        if left_overlap > 0:
            grad[0] += left_overlap

        # ---- middle buckets: (axis1[i-1], axis1[i]] -> values[i]
        for i in range(1, n):
            lower = max(start_x, self.axis1_[i - 1])
            upper = min(end_x, self.axis1_[i])
            if upper > lower:
                grad[i] += upper - lower

        # ---- right tail: (axis1[-1], +inf) -> values[-1]
        right_overlap = end_x - max(start_x, self.axis1_[-1])
        if right_overlap > 0:
            grad[-1] += right_overlap

        return grad

class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1 : np.ndarray | List, 
                               values : np.ndarray | List, 
                               interpolation_method : InterpMethod,
                               extrpolation_method : ExtrapMethod):


        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)
    
        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrpolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
