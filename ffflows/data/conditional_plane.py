import torch
from nflows.utils import tensor2numpy

from ffflows.data.plane import PlaneDataset
import numpy as np

class ConditionalPlaneDataset(PlaneDataset):
    def __init__(self, num_points, flip_axes=False):
        self.conditions = None
        super(ConditionalPlaneDataset, self).__init__(num_points, flip_axes)

    def __getitem__(self, item):
        return self.data[item], self.conditions[item]

class ConditionalWrapper(ConditionalPlaneDataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        super(ConditionalWrapper, self).__init__(base_dataset.num_points)

    def _get_data(self):
        return 0, 0

    def _create_data(self, **kwargs):
        data, condition = self._get_data(**kwargs)
        if not torch.is_tensor(condition):
            data, condition = [torch.Tensor(x).to(self.base_dataset.data) for x in [data, condition]]
        self.data, self.conditions = data, condition


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

class ConditionalAnulus(ConditionalPlaneDataset):
    def __init__(self, num_points, radius=None, std=0.3, flip_axes=False):
        """
        An Anulus dataset with
        :param num_points:
        :param radius: Radius of the anulus, if None many anuli with random radii
        :param std: Width of the anulus
        :param flip_axes:
        """
        self.inner_radius = 1.0
        self.radius = radius
        self.std = std
        super().__init__(num_points, flip_axes)

    @staticmethod
    def create_circle(num_per_circle, radius=None, std=0.1, inner_radius=0.5):
        u = torch.rand(num_per_circle)
        r = torch.rand(num_per_circle) if radius is None else radius * torch.ones(num_per_circle)
        r += inner_radius
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        data = 0.5 * (r.view(-1, 1)) * data
        return data, r.view(-1,1)

    def _create_data(self):
        self.data, self.conditions = self.create_circle(self.num_points, radius=self.radius, std=self.std,
                                                        inner_radius=self.inner_radius)