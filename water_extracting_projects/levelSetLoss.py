import torch
from torch import nn
import torch.nn.functional as F


def Cal_phi(tensor):
    '''
    args:
        :param tensor: output for CNN
    :return: sigmoid(output) - 0.5
    '''
    return torch.sigmoid(tensor) - 0.5


def H_epsilon(z, epsilon=0.05):
    return 1 / 2 * (1 + torch.tanh(z / epsilon))


class BinaryLevelSetLoss(nn.Module):
    def __init__(self, ):
        super(BinaryLevelSetLoss, self).__init__()

    def forward(self, pred, gt):
        phi = Cal_phi(pred)
        he = H_epsilon(phi)
        c1 = torch.sum(gt * he) / he.sum()
        c2 = torch.sum(gt * (1-he)) / torch.sum(1-he)

        loss = torch.sum(torch.pow((gt-c1), 2)*he) + torch.sum(torch.pow((gt-c2), 2)*(1 - he))
        return loss


if __name__ == '__main__':
    a = torch.randn(1, 2, 2)
    b = torch.sigmoid(a)
    print(a)
    LSLoss = BinaryLevelSetLoss()
    loss = LSLoss(a, b)
    print(loss)
