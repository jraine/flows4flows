import torch
import ..utils.shuffle_tensor as shuffle_tensor

# TODO torch_utils
# def shuffle_tensor(data):
#     mx = torch.randperm(len(data), device=torch.device('cpu'))
#     return data[mx]


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


class UnconditionalDataToData:

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def left(self):
        return self.data1

    def right(self):
        return self.data2

    def paired(self):
        return PairedData(*[shuffle_tensor(data) for data in (self.data1, self.data2)])

class ConditionalDataToData:

    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def left(self):
        return self.data1

    def right(self):
        return self.data2

    def paired(self):
        #assuming data is of form (data,condition)
        data1, data2 = [shuffle_tensor(data) for data in (self.data1, self.data2)]
        data1_cond_target = (*data_l,data_r[0])
        data2_cond_target = (*data_l,data_l[0])
        return PairedData(data1_cond_target,data2_cond_target)
