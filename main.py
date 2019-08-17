# %%
import argparse
import os
import pickle as pkl
import subprocess as sp
import sys
import time
from argparse import RawTextHelpFormatter
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import tensorboardX as tbx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from logzero import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn.utils.rnn import *
import dataloader as DL
import network
import utils

NoNE = 'NoNE'

# argument parser
parser = argparse.ArgumentParser(usage='%(prog)s {train,test} [options]',
                                 description='''dataset directry tree:
    (dataset_path) --- train --- in.pickle   (parameter of car. ex. spring constant)
                    |         |- out.pickle  (output of simulation ex. max(beta))
                    |         |- swa.pickle  (stearing wheel angle.)
                    |         |- mask.pickle (mask of data.)
                    |         `- seq.pickle  (sequence data ex. x,y,z,beta...)
                    |
                    |- test  --- in.pickle
                    |         |-   ...
                    |         `- seq.pickle
                    |
                    `- valid --- in.pickle
                              |-   ...
                              `- seq.pickle''',
                                 epilog='end',
                                 add_help=True,
                                 formatter_class=RawTextHelpFormatter)

parser.add_argument('mode',
                    help='select mode',
                    type=str,
                    choices=['train', 'test'])

parser.add_argument('--epochs',
                    help='num epochs (default : 50)',
                    required=False,
                    type=int,
                    default=50)
parser.add_argument('--batch_size',
                    help='batch size (default : 16)',
                    required=False,
                    type=int,
                    default=16)
parser.add_argument('--device',
                    help='select device (default : cpu)',
                    required=False,
                    type=str,
                    default='cpu')
parser.add_argument('--tbpath',
                    help='tensor board path',
                    required=False,
                    type=str,
                    default='runs/')
parser.add_argument('--dataset_path',
                    help='dataset path',
                    required=False,
                    type=str,
                    default='data/rnn/')
parser.add_argument('--scale',
                    help='scale type',
                    required=False,
                    type=str,
                    default='standard',
                    choices=['standard', 'min_max', 'damy'])
parser.add_argument('--times',
                    help='number of model',
                    required=False,
                    type=int,
                    default=3)
parser.add_argument('--save_dir',
                    help='save location of model\'s weight',
                    type=str,
                    required=False,
                    default=NoNE)
parser.add_argument('--model',
                    help='model\'s weight path',
                    type=str,
                    required=False,
                    default=NoNE)
parser.add_argument('--numbering',
                    required=False,
                    default=False,
                    action='store_true')

# network parameter
parser.add_argument('--sim_input_layer',
                    help='layer of simulation part\'s input layer',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('--sim_rnn_layer',
                    help='layer of simulation part\'s rnn layer',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('--sim_output_layer',
                    help='layer of simulation part\'s rnn layer',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('--cal_rnn_layer',
                    help='layer of calculation part\'s rnn layer',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('--cal_output_layer',
                    help='layer of calculation part\'s output layer',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('--sim_input_drop',
                    help='dropout input of simulation pard',
                    required=False,
                    type=float,
                    default=0.0)
parser.add_argument('--sim_output_drop',
                    help='dropout output of simulation part',
                    required=False,
                    type=float,
                    default=0.0)
parser.add_argument('--cal_input_drop',
                    help='dropout input of calculation part',
                    required=False,
                    type=float,
                    default=0.0)
parser.add_argument('--cal_output_drop',
                    help='dropout output of calculation part',
                    required=False,
                    type=float,
                    default=0.0)

parser.add_argument('--unit_size',
                    help='unit size of layer',
                    required=False,
                    type=int,
                    default=512)

parser.add_argument('--loss_out',
                    help='weight of output loss',
                    required=False,
                    type=float,
                    default=0.1)
parser.add_argument('--loss_seq',
                    help='weight of sequence loss',
                    required=False,
                    type=float,
                    default=0.1)
parser.add_argument('--l2',
                    help='weight of l2 regularization',
                    required=False,
                    type=float,
                    default=0.0)

args = parser.parse_args()

# epoch
# epochs = 50
# epochs=1
epochs = args.epochs

# batch size
# batch_size = 16
batch_size = args.batch_size

device = torch.device(args.device)

# setting optimizer
optimizer = optim.Adam

# set tensor board path
path = args.tbpath
# path = 'runs/cat_swa_in/forpaper__input_layer__rnn_layer__output_layer__olayer__last_rnn_layer__mse__l2__oloss__output_drop__last_drop__nn_drop/'

save_flag = (args.save_dir != NoNE)
load_flag = (args.model != NoNE)

number = args.numbering

# if you want to make dir automatically, uncomment
# sp.call('mkdir -p {}'.format(path).split())

# set dataset path
dataset_prefix = args.dataset_path
train_dir = os.path.join(dataset_prefix, 'train/')
test_dir = os.path.join(dataset_prefix, 'test/')
valid_dir = os.path.join(dataset_prefix, 'valid/')

modes = [train_dir, test_dir, valid_dir]

iname = 'in.pickle'
oname = 'out.pickle'
mname = 'mask.pickle'
swaname = 'swa.pickle'
seqname = 'seq.pickle'

names = [iname, swaname, seqname, mname, oname]

# load dataset
data = []
for m in modes:
    tmp = []
    for n in names:
        with open(os.path.join(m, n), mode='rb') as f:
            d = pkl.load(f)
        tmp.append(d)
    data.append(tmp)

train_data = data[0]
test_data = data[1]
valid_data = data[2]


def sort_data(x, y, end, reverse=False):
    return zip(*sorted(zip(x, y, end), key=lambda x: x[2], reverse=reverse))


def data2xy(param, swa, seq, mask):
    ends = []
    x = []
    y = []
    seq = np.array(seq)
    for i, m in enumerate(mask):
        ends.append(np.min(np.sum(m, axis=1)))
        x_ = []
        y_ = []
        for j in range(len(swa[i])):
            s = swa[i][j]
            # print(i,j)
            if s != 0:
                s = np.concatenate((np.array([s]), np.array(param[i])))
            else:
                s = np.concatenate((np.array([s]), np.zeros(len(param[i]))))
            x_.append(s)
            y_.append(seq[i, :, j])
        y.append(y_)
        x.append(x_)
    # x, y, ends = sort_data(x, y, ends, reverse=False)
    return np.array(x), np.array(y), np.array(ends)


# parse data
xtrain, ytrain, endstrain = data2xy(train_data[0], train_data[1],
                                    train_data[2], train_data[3])
xtest, ytest, endstest = data2xy(test_data[0], test_data[1], test_data[2],
                                 test_data[3])
xvalid, yvalid, endsvalid = data2xy(valid_data[0], valid_data[1],
                                    valid_data[2], valid_data[3])

# data convert to tensor
X_train = torch.from_numpy(xtrain).to(device=device, dtype=torch.float)
y_train = torch.from_numpy(ytrain).to(device=device, dtype=torch.float)
X_test = torch.from_numpy(xtest).to(device=device, dtype=torch.float)
y_test = torch.from_numpy(ytest).to(device=device, dtype=torch.float)
X_valid = torch.from_numpy(xvalid).to(device=device, dtype=torch.float)
y_valid = torch.from_numpy(yvalid).to(device=device, dtype=torch.float)

ends_train = torch.from_numpy(endstrain).to(device=device, dtype=torch.float)
ends_test = torch.from_numpy(endstest).to(device=device, dtype=torch.float)
ends_valid = torch.from_numpy(endsvalid).to(device=device, dtype=torch.float)

mlen = X_train.size(1)
len_train = mlen - ends_train
len_test = mlen - ends_test
len_valid = mlen - ends_valid

out_train = torch.tensor(train_data[4]).to(device)
out_test = torch.tensor(test_data[4]).to(device)
out_valid = torch.tensor(valid_data[4]).to(device)

le, x_, y_, out_ = zip(*sorted(zip(len_train, X_train, y_train, out_train),
                               reverse=True,
                               key=lambda x: x[0]))
X_train = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), x_)), dim=0)
len_train = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), le)))
y_train = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), y_)), dim=0)
out_train = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), out_)))

le, x_, y_, out_ = zip(*sorted(
    zip(len_test, X_test, y_test, out_test), reverse=True, key=lambda x: x[0]))
X_test = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), x_)), dim=0)
len_test = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), le)))
y_test = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), y_)), dim=0)
out_test = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), out_)))

le, x_, y_, out_ = zip(*sorted(zip(len_valid, X_valid, y_valid, out_valid),
                               reverse=True,
                               key=lambda x: x[0]))
X_valid = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), x_)), dim=0)
len_valid = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), le)))
y_valid = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), y_)), dim=0)
out_valid = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), out_)))


class damy():
    def __init__(self):
        pass

    def fit(self, x):
        return x

    def transform(self, x):
        pass

    def fit_transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


# get output from sequence data
out_train = utils.seq2out(y_train, len_train)
out_test = utils.seq2out(y_test, len_test)
out_valid = utils.seq2out(y_valid, len_valid)

# scale data
# mode: standard, min_max, damy(not scale)
sc_in = utils.RnnDataNorm(mode=args.scale)
sc_seq = utils.RnnDataNorm(mode=args.scale)
# if args.scale=='standard':
#     sc_out = StandardScaler()
# elif args.scale=='min_max':
#     sc_out=MinMaxScaler()
# else:
#     sc_out=damy()
sc_out = damy()

X_train = sc_in.fit_transform(X_train, x_len=len_train)
X_test = sc_in.transform(X_test, x_len=len_test)
X_valid = sc_in.transform(X_valid, x_len=len_valid)

y_train = sc_seq.fit_transform(y_train, x_len=len_train)
y_test = sc_seq.transform(y_test, x_len=len_test)
y_valid = sc_seq.transform(y_valid, x_len=len_valid)

out_test_denorm = out_test.clone()

# get output from sequence data
out_train = utils.seq2out(y_train, len_train)
out_test = utils.seq2out(y_test, len_test)
out_valid = utils.seq2out(y_valid, len_valid)

# out_train = torch.tensor(sc_out.fit_transform(
#     out_train.cpu().numpy())).to(device)
# out_valid = torch.tensor(sc_out.transform(out_valid.cpu().numpy())).to(device)
# out_test = torch.tensor(sc_out.transform(out_test.cpu().numpy())).to(device)

# make y0
y0_train = y_train[:, 0, :]
y_train = y_train[:, 1:, :]
y0_test = y_test[:, 0, :]
y_test = y_test[:, 1:, :]
y0_valid = y_valid[:, 0, :]
y_valid = y_valid[:, 1:, :]

# make y1 (equal to y0)
y1_train = y0_train.clone()
y1_test = y0_test.clone()
y1_valid = y0_valid.clone()
len_train -= 1
len_test -= 1
len_valid -= 1
X_train = X_train[:, :X_train.size(1) - 1, :]
X_test = X_test[:, :X_test.size(1) - 1, :]
X_valid = X_valid[:, :X_valid.size(1) - 1, :]

y_test_denorm = sc_seq.inverse_transform(y_test, len_test)

# data transform to torch format
dataset = torch.utils.data.TensorDataset(X_train, y_train, len_train, y0_train,
                                         y1_train, out_train)
##########################################################################

# define loss function


class my_loss(nn.modules.loss._Loss):
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
                 k=1,
                 oloss=0.):
        super(my_loss, self).__init__(size_average, reduce, reduction)
        self.rnn = utils.RnnLoss(MSE, L1, L2, Output, Curriculum,
                                 curriculum_num, curriculum_rnd, size_average,
                                 reduce, reduction, topk, k)
        self.outloss = nn.MSELoss()
        self.oloss = oloss

    def forward(self, y, y_pred, length, params, out, out_pred):
        loss = self.rnn(y, y_pred, length, params)
        loss += self.outloss(out, out_pred).to(out) * self.oloss
        # smoothing term

        return loss


def train(model,
          loss_func,
          optimizer,
          device,
          X_train,
          y_train,
          length,
          y0,
          y1,
          out,
          label=None,
          batch_first=False,
          p=0.0):
    model.train()
    y_train = y_train.clone()
    optimizer.zero_grad()
    y_pred, out_pred = model(X_train, length, y0, y1, label=label, p=p)
    loss = loss_func(y_pred, y_train, length, model.parameters(), out,
                     out_pred)
    loss.backward()
    optimizer.step()
    return loss.item()


def valid(model, loss_func, device, X_test, y_test, length, y0, y1, out):
    model.eval()
    X_test = X_test.clone()
    y_test = y_test.clone()
    y_pred, out_pred = model(X_test, length, y0, y1)
    val_loss = loss_func(y_pred, y_test, length, model.parameters(), out,
                         out_pred)
    return val_loss.item(), y_pred


def test(model, X_test, length, y0, y1):
    model.eval()
    y_pred, out_pred = model(X_test, length, y0, y1)
    return y_pred, out_pred


def mse_out(o, o_pred):
    return (o - o_pred).pow(2).mean(dim=0)


def rmse_out(o, o_pred):
    return mse_out(o, o_pred).sqrt()


# define network
class my_network(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 last_size,
                 sim_input_layer=1,
                 sim_rnn_layer=1,
                 cal_rnn_layer=1,
                 sim_output_layer=1,
                 cal_output_layer=1,
                 batch_first=False,
                 sim_input_dropout=0,
                 sim_rnn_dropout=0,
                 sim_output_dropout=0,
                 cal_input_dropout=0.0,
                 cal_rnn_dropout=0.0,
                 cal_output_dropout=0.,
                 lstm=False,
                 bidirectional=False,
                 device=torch.device('cpu'),
                 in_label=False,
                 poly=1):
        super(my_network, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sim_input_layer = sim_input_layer
        self.sim_rnn_layer = sim_rnn_layer
        self.sim_output_layer = sim_output_layer
        self.device = device
        self.lstm = lstm
        self.in_label = in_label
        self.poly = poly
        self.cal_rnn_layer = cal_rnn_layer

        self.drop_last = nn.Dropout(cal_input_dropout)
        self.simnet = network.MyRNN_skip(input_size, hidden_size, output_size,
                                         sim_input_layer, sim_rnn_layer,
                                         sim_output_layer, batch_first,
                                         sim_input_dropout, sim_rnn_dropout,
                                         sim_output_dropout, lstm,
                                         bidirectional, device, in_label, poly)
        if lstm:
            self.rnn = nn.LSTM(output_size + 1,
                               hidden_size,
                               cal_rnn_layer,
                               batch_first=batch_first,
                               dropout=cal_rnn_dropout,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(output_size + 1,
                              hidden_size,
                              cal_rnn_layer,
                              batch_first=batch_first,
                              dropout=cal_rnn_dropout,
                              bidirectional=bidirectional)

        self.outlayer = network.NeuralNet(
            hidden_size + input_size - 2 * output_size - 1, hidden_size,
            last_size, cal_output_layer, cal_output_dropout)
        self.module_list = nn.ModuleList(
            [self.simnet, self.rnn, self.outlayer])

    def forward(self, x, x_len, y_0, y_1, label=None, p=0.0):
        # simulator part
        seq, _ = self.simnet(x, x_len, y_0, y_1, label, p)

        # init state to zero
        h_0 = torch.zeros(self.cal_rnn_layer, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()
        total_len = x.size(1)

        # convert data to sequence type
        s = pack_padded_sequence(torch.cat((seq, x[:, :, 0].unsqueeze(-1)),
                                           dim=-1),
                                 x_len,
                                 batch_first=self.batch_first)

        # compress sequence data to feature vector
        if self.lstm:
            # x_, (h, c) = self.rnn(x_, (h, c))
            out, (h, c) = self.rnn(s, (h_0, c_0))
        else:
            # x_, h = self.rnn(x_, h)
            out, h = self.rnn(s, h_0)
        out, _ = pad_packed_sequence(out,
                                     batch_first=self.batch_first,
                                     total_length=total_len)
        out = out[:, x_len].diagonal(dim1=0, dim2=1)
        out = out.transpose(0, 1)
        # concatenate feature of simulation part's output and car parameter
        out = torch.cat((out, x[:, 0, 1:]), dim=-1)
        # dropout input to calculation part
        out = self.drop_last(out)
        # fully connected layer
        out = self.outlayer(out)
        return seq, out


class my_network_noskip(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 last_size,
                 sim_input_layer=1,
                 sim_rnn_layer=1,
                 cal_rnn_layer=1,
                 sim_output_layer=1,
                 cal_output_layer=1,
                 batch_first=False,
                 sim_input_dropout=0,
                 sim_rnn_dropout=0,
                 sim_output_dropout=0,
                 cal_input_dropout=0.0,
                 cal_rnn_dropout=0.0,
                 cal_output_dropout=0.,
                 lstm=False,
                 bidirectional=False,
                 device=torch.device('cpu'),
                 in_label=False,
                 poly=1):
        super(my_network, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sim_input_layer = sim_input_layer
        self.sim_rnn_layer = sim_rnn_layer
        self.sim_output_layer = sim_output_layer
        self.device = device
        self.lstm = lstm
        self.in_label = in_label
        self.poly = poly
        self.cal_rnn_layer = cal_rnn_layer

        self.drop_last = nn.Dropout(cal_input_dropout)
        self.simnet = network.MyRNN_skip(input_size, hidden_size, output_size,
                                         sim_input_layer, sim_rnn_layer,
                                         sim_output_layer, batch_first,
                                         sim_input_dropout, sim_rnn_dropout,
                                         sim_output_dropout, lstm,
                                         bidirectional, device, in_label, poly)
        if lstm:
            self.rnn = nn.LSTM(output_size + 1,
                               hidden_size,
                               cal_rnn_layer,
                               batch_first=batch_first,
                               dropout=cal_rnn_dropout,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(output_size + 1,
                              hidden_size,
                              cal_rnn_layer,
                              batch_first=batch_first,
                              dropout=cal_rnn_dropout,
                              bidirectional=bidirectional)

        self.outlayer = network.NeuralNet(hidden_size, hidden_size, last_size,
                                          cal_output_layer, cal_output_dropout)
        self.module_list = nn.ModuleList(
            [self.simnet, self.rnn, self.outlayer])

    def forward(self, x, x_len, y_0, y_1, label=None, p=0.0):
        # simulator part
        seq, _ = self.simnet(x, x_len, y_0, y_1, label, p)

        # init state to zero
        h_0 = torch.zeros(self.cal_rnn_layer, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()
        total_len = x.size(1)

        # convert data to sequence type
        s = pack_padded_sequence(seq, x_len, batch_first=self.batch_first)

        # compress sequence data to feature vector
        if self.lstm:
            # x_, (h, c) = self.rnn(x_, (h, c))
            out, (h, c) = self.rnn(s, (h_0, c_0))
        else:
            # x_, h = self.rnn(x_, h)
            out, h = self.rnn(s, h_0)
        out, _ = pad_packed_sequence(out,
                                     batch_first=self.batch_first,
                                     total_length=total_len)
        out = out[:, x_len].diagonal(dim1=0, dim2=1)
        out = out.transpose(0, 1)
        # concatenate feature of simulation part's output and car parameter
        # out = torch.cat((out, x[:, 0, 1:]), dim=-1)
        # dropout input to calculation part
        out = self.drop_last(out)
        # fully connected layer
        out = self.outlayer(out)
        return seq, out


################################################################
Input_size = X_train.size(-1) + y0_train.size(-1) + y1_train.size(-1)
# Input_size=X_train.size(-1)+y0_train.size(-1)
output_size = y_train.size(-1)
last_size = 6

in_label = True
shuffle = True

times = args.times


def main_test():
    if args.output == NoNE:
        sys.stderr('error: test mode needs \"output\" option.')
        return 1
    out_preds = torch.empty(0)
    for t in range(times):
        net = my_network(Input_size,
                         args.unit_size,
                         output_size,
                         last_size,
                         sim_input_layer=args.sim_input_layer,
                         sim_rnn_layer=args.sim_rnn_layer,
                         sim_output_layer=args.sim_output_layer,
                         cal_rnn_layer=args.cal_rnn_layer,
                         cal_output_layer=args.cal_output_layer,
                         lstm=True,
                         batch_first=True,
                         device=device,
                         in_label=in_label,
                         cal_input_dropout=args.cal_input_drop,
                         cal_output_dropout=args.cal_output_drop,
                         sim_input_dropout=args.sim_input_drop,
                         sim_output_dropout=args.sim_output_drop)
        net = net.to(device)
        if load_flag and os.path.exists(
                os.path.join(args.model, '{}'.format(t))):
            net.load_state_dict(
                torch.load(os.path.join(args.model, '{}'.format(t))))
        y_pred, out_pred = test(net, X_test, len_test.tolist(), y0_test,
                                y1_test)
        out_preds = torch.cat((out_preds, out_pred.unsqueeze(0)), dim=0)
    y = y_pred.detach().cpu().numpy()
    # y = sc_seq.inverse_transform(y_pred)
    o = out_preds.mean(0).detach().cpu().numpy()
    # o = sc_seq.inverse_transform_out(o)
    with open(os.path.join(args.output, 'seq'), 'wb') as f:
        pkl.dump(y, f)
    with open(os.path.join(args.output, 'output'), 'wb') as f:
        pkl.dump(o, f)


def main_train():
    writer_mean = tbx.SummaryWriter(log_dir=os.path.join(args.tbpath, 'mean'))
    rmses = torch.zeros(times, epochs, y_test.size(0), 6).cpu()
    for t in range(times):
        loss_func = my_loss(MSE=args.loss_seq, L2=args.l2, oloss=args.loss_out)
        net = my_network(Input_size,
                         args.unit_size,
                         output_size,
                         last_size,
                         sim_input_layer=args.sim_input_layer,
                         sim_rnn_layer=args.sim_rnn_layer,
                         sim_output_layer=args.sim_output_layer,
                         cal_rnn_layer=args.cal_rnn_layer,
                         cal_output_layer=args.cal_output_layer,
                         lstm=True,
                         batch_first=True,
                         device=device,
                         in_label=in_label,
                         cal_input_dropout=args.cal_input_drop,
                         cal_output_dropout=args.cal_output_drop,
                         sim_input_dropout=args.sim_input_drop,
                         sim_output_dropout=args.sim_output_drop)
        net = net.to(device)

        if load_flag and os.path.exists(
                os.path.join(args.model, '{}'.format(t))):
            net.load_state_dict(
                torch.load(os.path.join(args.model, '{}'.format(t))))
        optimizer = optim.Adam(net.parameters())
        # define tensorboard path and file name

        writer = tbx.SummaryWriter(
            log_dir=os.path.join(args.tbpath, '{}'.format(t)))
        # training ################################################
        for epoch in range(epochs):
            start = time.time()
            train_itr = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle)
            all_loss = []
            all_val = []

            for i, tr in enumerate(train_itr):
                X, y, length, y0, y1, out = tr
                le, tmp, y_, y0_, y1_, out_ = zip(
                    *sorted(zip(length, X, y, y0, y1, out),
                            reverse=True,
                            key=lambda x: x[0]))
                X = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), tmp)),
                              dim=0)
                length = torch.cat(
                    list(map(lambda x: torch.unsqueeze(x, 0), le)))
                y1 = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), y1_)))
                y0 = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), y0_)))
                y = torch.cat(list(map(lambda x: torch.unsqueeze(x, 0), y_)),
                              dim=0)
                out = torch.cat(
                    list(map(lambda x: torch.unsqueeze(x, 0), out_)))
                if epoch / epochs * 3 < 1:
                    p = 1
                elif epoch / epochs * 3 < 2:
                    p = 0.5
                else:
                    p = 0
                # p = min(1, 1.5-epoch/epochs)
                # p=0.5
                p=1
                loss = train(net,
                             loss_func,
                             optimizer,
                             device,
                             X,
                             y,
                             length.tolist(),
                             y0,
                             y1,
                             out,
                             label=y,
                             batch_first=True,
                             p=p)
                all_loss.append(loss)

            timer = time.time() - start
            # for i, (X, y,ends) in enumerate(valset):
            #     val_loss, _ = network.valid(net, loss_func, device, X, y)
            #     all_val.append(val_loss)
            val_loss, _ = valid(net, loss_func, device, X_valid, y_valid,
                                len_valid.tolist(), y0_valid, y1_valid,
                                out_valid)

            mloss = mean(all_loss)
            # mval = mean(all_val)

            logger.info('epoch: {:3} loss: {:>6} valid: {:>6}'.format(
                epoch, mloss, val_loss))

            writer.add_scalar('train/loss', mloss, epoch)
            writer.add_scalar('train/time', timer, epoch)
            writer.add_scalar('valid/loss', val_loss, epoch)
            ###########################################################

            # test ####################################################

            y_pred, out_pred = test(net, X_test, len_test.tolist(), y0_test,
                                    y1_test)
            count = (y_test != 0).sum(dim=0).sum(dim=0)

            diff = y_test - y_pred
            rate = diff / y_test
            rate[torch.isfinite(rate) == 0] = 0

            diffm = diff.abs().sum(dim=0).sum(dim=0) / count.to(
                dtype=torch.float)
            diffm[torch.isfinite(diffm) == 0] = 0
            ratem = rate.abs().sum(dim=0).sum(dim=0) / count.to(
                dtype=torch.float)
            ratem[torch.isfinite(ratem) == 0] = 0
            # print('diff',diff.size())
            # print('rate',rate.size())
            # print('diffm',diffm.size())
            # print('ratem',ratem.size())

            for i, (d, r) in enumerate(zip(diffm, ratem)):
                writer.add_scalar('test/error/{:02}'.format(i), d, epoch)
                writer.add_scalar('test/error_rate/{:02}'.format(i), r, epoch)

            m = ratem.mean()
            writer.add_scalar('test/error_avg', m, epoch)

            # output error
            diff_out = (out_test - out_pred)
            rate_out = (diff_out / out_test)

            diff_out = diff_out.abs().mean(dim=0)
            rate_out = rate_out.abs().mean(dim=0)

            for i, v in enumerate(diff_out):
                writer.add_scalar('out/error/{:02}'.format(i), v, epoch)
            for i, v in enumerate(rate_out):
                writer.add_scalar('out/error_rate/{:02}'.format(i), v, epoch)

            writer.add_scalar('out/error_avg', diff_out.abs().mean(), epoch)
            writer.add_scalar('out/error_rate_avg',
                              rate_out.abs().mean(), epoch)

            # mse,rmse of out
            out_mse_std = mse_out(out_test, out_pred).cpu()
            out_rmse_std = rmse_out(out_test, out_pred).cpu()

            y_pred_denorm = sc_seq.inverse_transform(y_pred, len_test)
            out_pred_denorm = torch.tensor(
                sc_seq.inverse_transform_out(out_pred).to(device))
            out_mse = mse_out(out_test_denorm, out_pred_denorm).cpu()
            out_rmse = rmse_out(out_test_denorm, out_pred_denorm).cpu()
            rmses[t, epoch] = out_rmse_std.detach().cpu()
            for i in range(out_mse.size(-1)):
                writer.add_scalar('mse/{:02}'.format(i), out_mse[i], epoch)
                writer.add_scalar('mse_std/{:02}'.format(i), out_mse_std[i],
                                  epoch)
                writer.add_scalar('rmse/{:02}'.format(i), out_rmse[i], epoch)
                writer.add_scalar('rmse_std/{:02}'.format(i), out_rmse_std[i],
                                  epoch)
            writer.add_scalar('mse/mean', out_mse.mean(), epoch)
            writer.add_scalar('mse_std/mean', out_mse_std.mean(), epoch)
            writer.add_scalar('rmse/mean', out_rmse.mean(), epoch)
            writer.add_scalar('rmse_std/mean', out_rmse_std.mean(), epoch)

        if save_flag:
            sp.call('mkdir -p {}'.format(args.save_dir).split())
            torch.save(net.state_dict(),
                       os.path.join(args.save_dir, '{}'.format(t)))

    rmses = rmses.mean(dim=0).mean(dim=-2)
    for e, rmse in enumerate(rmses):
        for i, r in enumerate(rmse):
            writer_mean.add_scalar('eval/{}'.format(i), r, e)
        writer_mean.add_scalar('eval/mean', rmse.mean(dim=-1), e)
    return 0


if __name__ == "__main__":
    if args.mode == 'train':
        sys.exit(main_train())
    elif args.mode == 'test':
        sys.exit(main_test())
