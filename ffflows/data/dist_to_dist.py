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

class PairedList(PairedData):

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, item):
        return [x[item] for x in self.X], [y[item] for y in self.y]

class UnconditionalDataToData(object):
    '''Data class for holding two datasets, each of same shape without conditions'''
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
    '''Data class for holding two datasets, each of the form [features,conditions].
    Associates conditions from data2 as target for data1 and vice verse.
    Builds paired data of form [features,conditions,targets]'''
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
        data1_cond_target = (*data1,data2[0])
        data2_cond_target = (*data2,data1[0])
        return PairedData(data1_cond_target,data2_cond_target)

class ConditionalDataToTarget(UnconditionalDataToData):
    '''Data class for holding two datasets.
    data1 has the form [features,conditions]
    data2 has the form [target_condition] and can either be a singular value, or a list the length of data1'''
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def left(self):
        return self.data1

    def right(self):
        return self.data2

    def paired(self):
        #assuming data1 is of form (data,condition), data2 is (condition)
        #will broadcase data2 if needed
        data2 = torch.as_tensor(self.data2)
        if data2.shape == self.data1[1].shape:
            return (*self.data1,self.data2)
        else:
            data2 = torch.broadcast_to(self.data2,self.data1[1].shape)
            return (*self.data1,data2)

class PairedConditionalDataToTarget(ConditionalDataToTarget):
    '''Data class for holding two datasets.
    Both have the form [features,conditions]
    Shuffles both and then duplicates condition as target for all data in form [features, condition, target].'''
    def paired(self):
        # assuming data is of form (data,condition)
        data1, data2 = [shuffle_tensor(data) for data in (self.data1, self.data2)]
        return PairedList((*data1, data1[1]), (*data2, data2[1]))