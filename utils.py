import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import sys


def R2(y, y_pred):
    # up = (y_pred-y.mean(dim=0)).pow(2).sum(dim=0)
    # down = (y-y.mean(dim=0)).pow(2).sum(dim=0)
    up = (y - y_pred).norm(2, dim=0)
    down = (y - y.mean(dim=0)).norm(2, dim=0)
    r2 = torch.ones_like(up) - up / down

    return r2


def R2_(y, y_pred):
    n = y.shape[0]
    k = y.shape[1]

    up = (y - y_pred).pow(2).sum() / n - k - 1
    down = (y - y.mean(dim=0)).pow(2).sum() / n - 1

    return torch.ones_like(up) - up / down


class MyLoss(nn.modules.loss._Loss):
    def __init__(self,
                 MSE=None,
                 L1=None,
                 L2=None,
                 r2=None,
                 size_average=None,
                 reduce=None,
                 reduction=None):
        super(MyLoss, self).__init__(size_average, reduce, reduction)
        if MSE is None and L1 is None and L2 is None and R2 is None:
            print('WARNING : Loss is always zero.')
        self.MSE = MSE
        self.L1 = L1
        self.L2 = L2
        self.r2 = r2

    def name(self):
        return 'MSE={:} L1={:} L2={:} r2={:}'.format(self.MSE, self.L1,
                                                     self.L2, self.r2)

    def forward(self, y, y_pred, params):
        loss = torch.zeros(1).to(y)

        if self.MSE is not None:
            mseloss = nn.MSELoss()
            loss += self.MSE * mseloss(y, y_pred)

        if self.L1 is not None:
            l1loss = torch.zeros(1).to(loss)
            for p in params:
                l1loss += p.norm(1)

            loss += self.L1 * l1loss

        if self.L2 is not None:
            l2loss = torch.zeros(1).to(loss)
            for p in params:
                l2loss += p.norm(2)

            loss += self.L2 * l2loss

        if self.r2 is not None:
            r2loss = R2(y, y_pred).norm(2) - 1

            if torch.isfinite(r2loss).item():
                loss += self.r2 * r2loss

        return loss


class RnnLoss(nn.modules.loss._Loss):
    def __init__(self,
                 MSE=None,
                 L1=None,
                 L2=None,
                 Output=None,
                 Curriculum=None,
                 curriculum_num=2,
                 curriculum_rnd=1,
                 size_average=None,
                 reduce=None,
                 reduction=None,
                 topk=None,
                 k=1):
        super(RnnLoss, self).__init__(size_average, reduce, reduction)
        if MSE is None and L1 is None and L2 is None and R2 is None:
            print('WARNING : Loss is always zero.')
        self.MSE = MSE
        self.L1 = L1
        self.L2 = L2
        self.Output = Output
        self.Curriculum = Curriculum
        self.first = True
        self.curriculum_num = curriculum_num
        self.topk = topk
        self.k = k

    def name(self):
        return 'MSE={:} L1={:} L2={:} r2={:}'.format(self.MSE, self.L1,
                                                     self.L2, self.r2)

    def forward(self, y, y_pred, length, params):
        loss = torch.zeros(1).to(y)
        length = torch.tensor(length)
        if self.MSE is not None:
            mseloss = nn.MSELoss()
            loss += self.MSE * mseloss(y, y_pred)

        if self.L1 is not None:
            l1loss = torch.zeros(1).to(loss)
            for p in params:
                l1loss += p.norm(1)

            loss += self.L1 * l1loss

        if self.L2 is not None:
            l2loss = torch.zeros(1).to(loss)
            cnt = 0
            for p in params:
                l2loss += p.pow(2).mean()
                cnt += 1

            loss += self.L2 * l2loss / cnt

        if self.Curriculum is not None or self.Output is not None:
            y_out = seq2out(y, length)
            pred_out = seq2out(y_pred, length)

        if self.Output is not None:
            outloss = nn.MSELoss()
            loss += self.Output * outloss(y_out, pred_out)

        if self.Curriculum is not None:
            curriculum_mse = (y_out - pred_out).pow(2).mean(dim=0)
            if self.first:
                self.uniform_sampler = torch.distributions.Uniform(
                    0, len(curriculum_mse))
            if not self.first:
                rank = (curriculum_mse -
                        self.past_curriculum_mse).pow(2).sort()[1].to(
                            torch.float)
                # mask = (rank < self.curriculum_num).to(torch.float)
                mask = torch.zeros_like(curriculum_mse)
                mask[rank.argmax()] = 1
                mask[rank.argmin()] = 1
                rnd_mask = torch.zeros_like(curriculum_mse)
                rnd_mask[int(self.uniform_sampler.sample())] = 1
                mask = mask + rnd_mask

                curriculum_loss = curriculum_mse.dot(mask)
                loss += curriculum_loss * self.Curriculum
                print(curriculum_loss)
            self.past_curriculum_mse = curriculum_mse
            self.first = False

        if self.topk is not None:
            topk_loss = (y - y_pred).pow(2).topk(self.k)[0].norm(2)
            loss += topk_loss

        return loss


class RnnDataNorm():
    def __init__(self, mode='standard', min=0, max=1, batch_first=True):
        self.mode = mode
        self.min = min
        self.max = max
        self.batch_first = batch_first

    def fit_damy(self, x):
        self.isfit = True

    def transform_damy(self, x):
        return x.clone()

    def fit_min_max(self, x):
        self.x_min = x.min(dim=0)[0].min(dim=0)[0]
        self.x_max = x.max(dim=0)[0].max(dim=0)[0]

    def transform_min_max(self, X):
        x = X.clone()
        x = (x - self.x_min) / (self.x_max -
                                self.x_min) * (self.max - self.min) + self.min
        return x

    def fit_standard(self, X):
        x = X.clone()
        self.mean = x.sum(dim=(0, 1)) / (x != 0).sum(dim=(0, 1)).to(
            torch.float)
        # get variance
        res = x - self.mean
        res[x == 0] = 0
        self.var=res.pow(2).sum(dim=(0, 1)) / \
            (x != 0).sum(dim=(0, 1)).to(torch.float)
        self.std = self.var.sqrt()

    def transform_standard(self, X):
        x = X.clone()
        ans = x - self.mean
        ans = ans / self.std
        return ans

    def fit(self, x):
        self.isfit = True
        if self.mode == 'standard':
            self.fit_standard(x)
        elif self.mode == 'min_max':
            self.fit_min_max(x)
        elif self.mode == 'damy':
            self.fit_damy(x)

    def transform(self, X, x_len=None):
        if not self.isfit:
            print('not fit')
            return
        if self.mode == 'standard':
            x = self.transform_standard(X)
        elif self.mode == 'min_max':
            x = self.transform_min_max(X)
        elif self.mode == 'damy':
            x = self.transform_damy(X)

        if x_len is not None:
            x = pad_zero(x, x_len)
        return x

    def fit_transform(self, x, x_len=None):
        self.fit(x)
        return self.transform(x, x_len=x_len)

    def inverse_transform_standard(self, X):
        x = X.clone()
        x = x * self.std
        ans = x + self.mean
        return ans

    def inverse_transform_min_max(self, X):
        x = X.clone()
        x = (x - self.min) * (self.x_max -
                              self.x_min) / (self.max - self.min) + self.x_min
        return x

    def transform_out_standard(self, X):
        x = X.clone()
        x = (x - self.mean_o) / self.var_o
        return x

    def inverse_transform_out_standard(self, X):
        x = X.clone()
        x = x * self.var_o.to(x)
        x = x + self.mean_o.to(x)
        return x.to(x)

    def transform_out(self, x):
        if self.mode == 'standard':
            self.mean_o = torch.tensor([
                self.mean[19], self.mean[3], self.mean[1], self.mean[1],
                torch.mean(torch.tensor(self.mean[25], self.mean[27])),
                torch.mean(torch.tensor(self.mean[11], self.mean[21]))
            ])
            self.var_o = torch.tensor([
                self.var[19], self.var[3], self.var[1], self.var[1],
                torch.mean(torch.tensor(self.var[25], self.var[27])),
                torch.mean(torch.tensor(self.var[11], self.var[21]))
            ])
            return self.transform_out_standard(x)
        else:
            sys.stderr('not defined transform, so use standard mode')
            return self.transform_out_standard(x)

    def inverse_transform_out(self, x):
        if self.mode == 'standard':
            self.mean_o = torch.tensor([
                self.mean[19], self.mean[3], self.mean[1], self.mean[1],
                torch.mean(torch.tensor((self.mean[25], self.mean[27])).to(x)),
                torch.mean(torch.tensor((self.mean[11], self.mean[21])).to(x))
            ])
            self.var_o = torch.tensor([
                self.var[19], self.var[3], self.var[1], self.var[1],
                torch.mean(torch.tensor((self.var[25], self.var[27])).to(x)),
                torch.mean(torch.tensor((self.var[11], self.var[21])).to(x))
            ])
            return self.inverse_transform_out_standard(x).to(x)
        else:
            sys.stderr('not defined inverse_transform, so use standard mode')
            return self.inverse_transform_out_standard(x).to(x)

    def inverse_transform_damy(self, X):
        return X.clone()

    def inverse_transform(self, x, x_len=None):
        if not self.isfit:
            print('not fit')
            return
        if self.mode == 'standard':
            x = self.inverse_transform_standard(x)
        elif self.mode == 'min_max':
            x = self.inverse_transform_min_max(x)
        elif self.mode == 'damy':
            x = self.inverse_transform_damy(x)

        if x_len is not None:
            x = pad_zero(x, x_len)
        return x


def pad_zero(X, x_len, batch_first=True):
    x = X.clone()
    total_len = x.size(1)
    x = pack_padded_sequence(x, x_len, batch_first=batch_first)
    x, _ = pad_packed_sequence(x,
                               batch_first=batch_first,
                               total_length=total_len)
    return x


def seq2out(y, length):
    # all have to think about sequenth legth?
    le = length.clone().to(torch.int)
    out = torch.zeros(len(y), 6).to(y)
    # NOTE : minimum's probrem (whith is better)
    # min([1,2,3,4,5,6,7,0,0,0,0])=0
    # min([1,2,3,4,5,6])=1
    for i, l in enumerate(le):
        p = y[i, :l, :]
        out[i, 0] = p[p[:, 19].pow(2).argmax(), 19]
        out[i, 1] = p[p[:, 3].pow(2).argmax(), 3]
        out[i, 2] = p[:, 1].max()
        out[i, 3] = p[:, 1].min()
        out[i, 4] = torch.min(p[:, 25].min(), p[:, 27].min())
        out[i, 5] = torch.min(p[:, 11].min(), p[:, 21].min())
        # out[i, 0] = p[p[:, 19].pow(2).argmax(), 19]
        # out[i, 1] = p[p[:, 3].pow(2).argmax(), 3]
        # out[i, 2] = p[:, 1].max()
        # out[i, 3] = p[:, 1].min()
        # out[i, 4] = torch.min(p[:, 25].min(), p[:, 27].min())
        # out[i, 5] = torch.min(p[:, 11].min(), p[:, 21].min())
    return out
