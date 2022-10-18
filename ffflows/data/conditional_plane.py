import torch
from nflows.utils import tensor2numpy

from ffflows.data.plane import PlaneDataset
import numpy as np


class ConditionalWrapper(PlaneDataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        super(ConditionalWrapper, self).__init__(base_dataset.num_points)

    def _get_data(self):
        return 0, 0

    def _create_data(self, **kwargs):
        data, condition = self._get_data(**kwargs)
        if not torch.is_tensor(condition):
            data, condition = [torch.Tensor(x).to(self.base_dataset.data) for x in [data, condition]]
        self.data = [[d, r] for d, r in zip(data, condition.view(-1, 1))]


class RotatedData(ConditionalWrapper):

    def __init__(self, base_dataset, max_angle=360):
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
            return tensor2numpy(self.base_data.data) @ R
        else:
            R = np.array([self.make_rot_matrix(theta) for theta in angle])
            return np.einsum('ij,ijk->ik', data, R)

    def _get_data(self, angles=None):
        # write angle in degrees
        if angles is None:
            angles = np.random.randint(0, self.max_angle, self.num_points)
        cond_data = self.rotate(self.base_dataset.data, angles)
        # Assume data lies in [-4, 4] and scale so if a 45 degree rotation is applied the data is still in this square
        scale = np.sqrt(2 * 4 ** 2)
        data = cond_data / scale * 4
        return data, angles / self.max_angle
