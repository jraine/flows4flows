import torch

from ..utils import shuffle_tensor


class PairedData(torch.utils.data.Dataset):

    def __init__(self, X, y):
        super(PairedData, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def cpu(self):
        self.X = self.X.cpu()
        self.y = self.y.cpu()


class UnconditionalDataToData(object):

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def left(self):
        return self.data1

    def right(self):
        return self.data2

    def paired(self):
        return PairedData(*[shuffle_tensor(data) for data in (self.data1, self.data2)])


class ConditionalDataToData(UnconditionalDataToData):

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def left(self):
        return self.data1

    def right(self):
        return self.data2

    def paired(self):
        # assuming data is of form (data,condition)
        data1, data2 = [shuffle_tensor(data) for data in (self.data1, self.data2)]
        data1_cond_target = (*data1, data2[0])
        data2_cond_target = (*data2, data1[0])
        return PairedData(data1_cond_target, data2_cond_target)


class ConditionalDataToTarget(UnconditionalDataToData):

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def left(self):
        return self.data1

    def right(self):
        return self.data2

    def paired(self):
        # assuming data is of form (data,condition)
        data2 = torch.Tensor(self.data2)
        if data2.shape == self.data1[1].shape:
            return (*self.data1, self.data2)
        else:
            data2 = torch.broadcast_to(self.data2, self.data1.shape)
            return (*self.data1, data2)
