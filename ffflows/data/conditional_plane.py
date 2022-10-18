from nflows.utils import tensor2numpy

from ffflows.data.plane import PlaneDataset
import numpy as np


class ConditionalWrapper(PlaneDataset):

    def _get_data(self):
        return 0, 0

    def _create_data(self):
        data, condition = self._get_data()
        return [[d, r] for d, r in zip(data, condition.view(-1, 1))]


class RotatedData(PlaneDataset):

    def __init__(self, base_dataset, max_angle=360):
        super(RotatedData, self).__init__(base_dataset.num_points)
        self.max_angle = max_angle
        self.base_dataset = base_dataset

    def make_rot_matrix(self, angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return R

    def rotate(self, data, angle):
        print(angle)
        if not isinstance(angle, np.ndarray):
            R = self.make_rot_matrix(angle)
            return tensor2numpy(self.base_data.data) @ R
        else:
            R = np.array([self.make_rot_matrix(theta) for theta in angle])
            return np.einsum('ij,ijk->ik', data, R)

    def _get_data(self):
        # write angle in degrees
        angles = np.random.randint(0, self.max_angle, self.n_points)
        cond_data = self.rotate(self.base_dataset.data, angles)
        # Assume data lies in [-4, 4] and scale so 45 degree rotation still in this square
        scale = np.sqrt(2 * 4 ** 2)
        data = cond_data / scale * 4
        return data, angles / self.max_angle
