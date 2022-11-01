"""Wrapper classes for conditional plane datasets"""

import torch
from nflows.utils import tensor2numpy

from ffflows.data.plane import PlaneDataset
import numpy as np


class ConditionalPlaneDataset(PlaneDataset):
    def __init__(self, num_points, flip_axes=False, return_cond=True):
        self.conditions = None
        self.return_cond = return_cond
        super(ConditionalPlaneDataset, self).__init__(num_points, flip_axes)

    def __getitem__(self, item):
        if self.return_cond:
            return self.data[item], self.conditions[item]
        else:
            return self.data[item]

    def get_tuple(self):
        return self.data, self.conditions


class ConditionalWrapper(ConditionalPlaneDataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        super(ConditionalWrapper, self).__init__(base_dataset.num_points)

    def _get_conditional(self):
        return 0, 0

    def _create_data(self, **kwargs):
        data, condition = self._get_conditional(**kwargs)
        if not isinstance(condition, np.ndarray):
            condition = [condition]
        cond_size = len(condition)
        if cond_size != self.num_points:
            condition = np.tile(condition, self.num_points).reshape(-1, cond_size)
        if not torch.is_tensor(condition):
            data, condition = [torch.tensor(x, dtype=torch.float32).to(self.base_dataset.data) for x in
                               [data, condition]]

        # TODO write a subclass with this default given subclassing
        # if isinstance(self.base_dataset, ConditionalPlaneDataset):
        #     condition = torch.cat([self.base_dataset.conditions, condition], axis=-1)

        self.data = data
        self.conditions = condition.reshape(self.data.shape[0], -1)

    def get_default_eval(self, n_test):
        """Set the data to some default condition and return a set of default points."""
        return torch.linspace(0, 1, n_test)


class RotatedData(ConditionalWrapper):

    def __init__(self, base_dataset, max_angle=45):
        self.max_angle = max_angle
        super(RotatedData, self).__init__(base_dataset)

    def make_rot_matrix(self, angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return R

    def rotate(self, data, angle):
        if not isinstance(angle, np.ndarray):
            R = self.make_rot_matrix(angle)
            return tensor2numpy(self.base_dataset.data) @ R
        else:
            R = np.array([self.make_rot_matrix(theta) for theta in angle])
            return np.einsum('ij,ijk->ik', data, R)

    def _get_conditional(self, angles=None):
        # write angle in degrees
        if angles is None:
            angles = np.random.randint(0, self.max_angle, self.num_points)
        cond_data = self.rotate(self.base_dataset.data, angles)
        # Assume data lies in [-4, 4] and scale so if a 45 degree rotation is applied the data is still in this square
        scale = np.sqrt(2 * 4 ** 2)
        data = cond_data / scale * 4
        return data, angles / self.max_angle

    def get_default_eval(self, n_test):
        """Set the data to some default condition and return a set of default points."""
        self._create_data(angles=0)
        return torch.linspace(0, 1, n_test + 1)[1:]


class RadialScale(ConditionalWrapper):

    def __init__(self, base_dataset, max_scale=1, minscale=0.5):
        self.max_scale = max_scale
        self.min_scale = minscale
        super(RadialScale, self).__init__(base_dataset)

    def _get_conditional(self, scale=None):
        # write angle in degrees
        if scale is None:
            # When sampling in 2D there is a volume correction for the radius which is accounted for with the square
            # root
            scale = np.random.rand(self.num_points).reshape(-1, 1) ** 0.5 * (
                        self.max_scale - self.min_scale) + self.min_scale
        cond_data = self.base_dataset.data * scale
        return cond_data, scale

    def get_default_eval(self, n_test):
        """Set the data to some default condition and return a set of default points."""
        self._create_data(scale=1)
        return torch.linspace(self.min_scale, self.max_scale, n_test + 1)[1:]


class ElipseShift(ConditionalWrapper):

    def __init__(self, base_dataset, max_shift_x=1, max_shift_y=1):
        self.max_shift_x = max_shift_x
        self.max_shift_y = max_shift_y
        super(ElipseShift, self).__init__(base_dataset)

    def get_shift(self, shift, max):
        if shift is None:
            shift = np.random.rand(self.num_points).reshape(-1, 1) ** 0.5 * max
        elif not isinstance(shift, np.ndarray):
            shift = shift * np.ones((self.num_points, 1))
        return shift

    def shift_to_condition(self, shift):
        return shift / torch.concat((self.max_shift_x, self.max_shift_y), 0).view(1, 2)

    def _get_conditional(self, shift_x=None, shift_y=None):
        # write angle in degrees
        shift_x = self.get_shift(shift_x, self.max_shift_x)
        shift_y = self.get_shift(shift_y, self.max_shift_y)
        shift = np.concatenate((shift_x, shift_y), 1)
        cond_data = self.base_dataset.data * shift
        return cond_data, shift

    def get_default_eval(self, n_test):
        """Set the data to some default condition and return a set of default points."""
        raise NotImplementedError('This has not been set up for this class.')
        # TODO how to choose a direction in the order?
        self._create_data(scale=1)
        return torch.linspace(0, self.max_scale, n_test + 1)[1:]
