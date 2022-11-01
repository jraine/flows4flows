import torch


def set_trainable(model, trainable=True):
    '''Set parameters of an nn.module class to (non-)trainable'''
    for param in model.parameters():
        param.requires_grad = trainable
    model.eval()


def shuffle_tensor(data, device=torch.device('cpu')):
    '''Shuffle torch.Tensor with random permutation'''
    mx = torch.randperm(len(data), device=device)
    return data[mx]
