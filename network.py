import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.utils.rnn import *
from sklearn.preprocessing import StandardScaler
import utils
# in  19
# out 6


def train_cat(model, loss_func, optimizer, device, X_train, y_train, length, y0, label=None, batch_first=False, p=0.0):
    model.train()

    # X_train = torch.tensor(X_train, device=device)
    y_train = y_train.clone()

    optimizer.zero_grad()
    y_pred = model(X_train, length, y0, label=label, p=p)

    # packed = pack_padded_sequence(y_train, length, batch_first=batch_first)
    # y_train, _ = pad_packed_sequence(packed, batch_first=batch_first)

    loss = loss_func(y_pred, y_train, length, model.parameters())

    loss.backward()
    optimizer.step()

    return loss.item()


def valid_cat(model, loss_func, device, X_test, y_test, length, y0, batch_first=False):
    model.eval()

    X_test = X_test.clone()
    y_test = y_test.clone()

    y_pred = model(X_test, length, y0)

    # packed = pack_padded_sequence(y_test, length, batch_first=batch_first)
    # y_test, _ = pad_packed_sequence(packed, batch_first=batch_first)

    val_loss = loss_func(y_pred, y_test, length, model.parameters())

    # TODO : accuracy

    return val_loss.item(), y_pred


def test_cat(model, X_test, length, y0):
    model.eval()

    y_pred = model(X_test, length, y0)

    return y_pred


def train_rnn(model, loss_func, optimizer, device, X_train, y_train, length, batch_first=False):
    model.train()

    # X_train = torch.tensor(X_train, device=device)
    y_train = y_train.clone()

    optimizer.zero_grad()
    y_pred = model(X_train, length)

    # packed = pack_padded_sequence(y_train, length, batch_first=batch_first)
    # y_train, _ = pad_packed_sequence(packed, batch_first=batch_first)

    loss = loss_func(y_pred, y_train, length, model.parameters())

    loss.backward()
    optimizer.step()

    return loss.item()


def valid_rnn(model, loss_func, device, X_test, y_test, length, batch_first=False):
    model.eval()

    X_test = torch.tensor(X_test, device=device)
    y_test = torch.tensor(y_test, device=device)

    y_pred = model(X_test, length)

    # packed = pack_padded_sequence(y_test, length, batch_first=batch_first)
    # y_test, _ = pad_packed_sequence(packed, batch_first=batch_first)

    val_loss = loss_func(y_pred, y_test, length, model.parameters())

    # TODO : accuracy

    return val_loss.item(), y_pred


def test_rnn(model, X_test, length):
    model.eval()

    y_pred = model(X_test, length)

    return y_pred


def train(model, loss_func, optimizer, device, X_train, y_train):
    model.train()

    X_train = torch.tensor(X_train, device=device)
    y_train = torch.tensor(y_train, device=device)

    optimizer.zero_grad()

    y_pred = model(X_train)

    loss = loss_func(y_pred, y_train, model.parameters())

    loss.backward()
    optimizer.step()

    return loss.item()


def valid(model, loss_func, device, X_test, y_test):
    model.eval()

    X_test = torch.tensor(X_test, device=device)
    y_test = torch.tensor(y_test, device=device)

    y_pred = model(X_test)

    val_loss = loss_func(y_pred, y_test, model.parameters())

    # TODO : accuracy

    return val_loss.item(), y_pred


def test(model, X_test):
    model.eval()

    X_test = torch.tensor(X_test)

    y_pred = model(X_test)

    return y_pred


class KFoldCrossValidation():

    def __init__(self, k, dataset, model_method, model_args, device, epoch, loss_func, optim_method, optim_param=[], batch_size=1, shuffle=False, normalize=True):
        self.k = k
        self.dataset = dataset
        self.model_method = model_method
        self.model_args = model_args
        self.optim_method = optim_method
        self.optim_param = optim_param
        self.models = [self.model_method(
            *model_args).to(device=device) for i in range(k)]
        self.optimizer = optim_method(
            self.models[0].parameters(), *optim_param)
        self.loss_func = loss_func
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._SplitDataset()
        self.index_itr = -1
        self.index_dat = -1
        self.index_opt = -1
        self.normalize = normalize

    def __repr__(self):
        return self.model_method(*self.model_args).__repr__()

    def name(self):
        return __class__.__name__

    def _SplitDataset(self):
        num = self.dataset.__len__()
        q, mod = divmod(num, self.k)

        l = []

        for _ in range(self.k):
            if mod > 0:
                l.append(q+1)
                mod -= 1
            else:
                l.append(q)

        self.data = torch.utils.data.random_split(self.dataset, l)
        self.list = l

    def MakeDataset(self, num):
        if self.index_dat != num:
            if self.k > 1:
                self.train_data = torch.utils.data.ConcatDataset(
                    list(filter(lambda x: not x == self.data[num], self.data)))
                self.valid_data = self.data[num]
            else:
                self.train_data = self.dataset
                self.valid_data = self.dataset
            self.index_dat = num
        else:
            pass

    def MakeOptim(self, num):
        if self.index_opt != num:
            self.optimizer = self.optim_method(
                self.models[num].parameters(), *self.optim_param)
            self.index_opt = num
        else:
            pass

    def MakeItr(self, num):
        self.MakeDataset(num)
        if self.index_itr != num:
            self.train_iterater = torch.utils.data.DataLoader(
                self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
            self.index_itr = num
        else:
            pass

    def train_itr(self, num):
        self.MakeItr(num)
        self.MakeOptim(num)

        all_loss = []

        for _, tr in enumerate(self.train_iterater):
            X, y = tr
            loss = train(self.models[num], self.loss_func,
                         self.optimizer, self.device, X, y)

            all_loss.append(loss)
        return all_loss

    def valid(self, num, loss_func, device):
        valid_iterater = torch.utils.data.DataLoader(
            self.valid_data, drop_last=True)
        all_val = []
        y_preds = []
        for data in valid_iterater:
            vX, vy = data
            val, y_pred = valid(self.models[num], loss_func, device, vX, vy)
            all_val.append(val)
            y_preds.append(y_pred)

        return all_val, y_preds

    def test_each(self, num, X_test):
        return test(self.models[num], X_test)

    def test(self, X_test):
        for i, mdl in enumerate(self.models):
            if i == 0:
                res = test(mdl, X_test)
                res = res.reshape(1, len(res))
            else:
                tmp = test(mdl, X_test)
                res = torch.cat((res, tmp.reshape(1, len(tmp))))

        return res.mean(dim=0)


class NeuralNet(nn.Module):
    def __init__(self, in_unit, hidden_unit, out_unit, hidden_layer, is_debug=False,dropout=0.0):
        self.debug = is_debug
        self.hidden_layer = hidden_layer
        self.hidden_unit = hidden_unit
        self.in_unit = in_unit
        self.out_unit = out_unit

        super(NeuralNet, self).__init__()
        # for debug
        
        self.input = nn.Linear(self.in_unit, self.hidden_unit)
        self.activate_input = nn.ReLU()
        self.hidden = [nn.Linear(self.hidden_unit, self.hidden_unit)
                        for i in range(hidden_layer)]
        # self.drop =[nn.Dropout() for i in range(hidden_layer)]
        self.activate_hidden = [nn.ReLU() for i in range(hidden_layer)]
        self.output = nn.Linear(self.hidden_unit, self.out_unit)
        self.activate_output = nn.ReLU()

        self.list = [self.input, self.activate_input]
        for i in range(self.hidden_layer):
            self.list.append(self.hidden[i])
            self.list.append(self.activate_hidden[i])
            # self.list.append(self.drop[i])
        self.dropout=nn.Dropout(dropout)
        self.list.extend([self.dropout,self.output])

        self.module_list = nn.ModuleList(self.list)

    def name(self):
        return self.__class__.__name__

    def forward(self, x):
        # for test
        for l in self.module_list:
            x = l(x)

        return x


class NNDrop(nn.Module):
    def __init__(self, in_unit, hidden_unit, out_unit, hidden_layer, drop_in=0.5, drop_out=0.5):
        self.hidden_layer = hidden_layer
        self.hidden_unit = hidden_unit
        self.in_unit = in_unit
        self.out_unit = out_unit

        super(NNDrop, self).__init__()
        self.droprate_in = drop_in
        self.droprate_out = drop_out

        self.input = nn.Linear(self.in_unit, self.hidden_unit)
        self.activate_input = nn.ReLU()
        self.drop_in = nn.Dropout(self.droprate_in)

        self.hidden = [nn.Linear(self.hidden_unit, self.hidden_unit)
                       for i in range(hidden_layer)]
        # self.drop =[nn.Dropout() for i in range(hidden_layer)]
        self.activate_hidden = [nn.ReLU() for i in range(hidden_layer)]

        self.drop_out = nn.Dropout(self.droprate_out)

        self.output = nn.Linear(self.hidden_unit, self.out_unit)
        self.activate_output = nn.ReLU()

        self.list = [self.input, self.activate_input, self.drop_in]
        for i in range(self.hidden_layer):
            self.list.append(self.hidden[i])
            self.list.append(self.activate_hidden[i])
            # self.list.append(self.drop[i])
        self.list.extend([self.drop_out, self.output])

        self.module_list = nn.ModuleList(self.list)

    def name(self):
        return self.__class__.__name__

    def forward(self, x):
        for l in self.module_list:
            x = F.relu(l(x))
        return x


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size, drop=0.0):
        super(LinearRegression, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.drop = nn.Dropout(drop)
        self.layer = nn.Linear(self.input_size, self.output_size)

    def name(self):
        return self.__class__.__name__

    def forward(self, x):
        x = self.drop(x)
        x = self.layer(x)
        return x


class encoder_decoder(nn.Module):
    def __init__(self, input_size, output_size, drop_in=0.0, drop_out=0.0):
        super(encoder_decoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        # self.hidden_size=1024
        # self.latent_size=32

        self.input = nn.Linear(self.input_size, 1024)
        self.drop_in = nn.Dropout(drop_in)
        self.hidden1 = nn.Linear(1024, 256)
        self.hidden2 = nn.Linear(256, 64)
        self.latent = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64+64, 256)
        self.hidden4 = nn.Linear(256+256, 1024)
        self.drop_out = nn.Dropout(drop_out)
        self.output = nn.Linear(1024+1024, output_size)

    def forward(self, x):
        if len(x.size()) <= 1:
            d = 0
        else:
            d = 1
        x1 = F.relu(self.input(x))
        x1 = self.drop_in(x1)
        x2 = F.relu(self.hidden1(x1))
        x3 = F.relu(self.hidden2(x2))
        x4 = F.relu(self.latent(x3))
        out = F.relu(self.hidden3(torch.cat((x4, x3), dim=d)))
        out = F.relu(self.hidden4(torch.cat((out, x2), dim=d)))
        out = self.drop_out(out)
        out = F.relu(self.output(torch.cat((out, x1), dim=d)))

        return out

# class Embed(nn.Module):
#     def __init__


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extract_layer=1, rnn_layer=1, output_layer=1, batch_first=False, input_dropout=0, rnn_dropout=0, output_dropout=0, lstm=False, bidirectional=False, device=torch.device('cpu')):
        super(MyRNN, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.extract_layer = extract_layer
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer
        self.device = device
        self.lstm = lstm
        # self.embed = NeuralNet(input_size, hidden_size,
        #                        hidden_size, embed_layer)
        self.indrop = nn.Dropout(input_dropout)
        self.input = nn.Linear(input_size, input_size)
        if lstm:
            self.rnn = nn.LSTM(input_size, hidden_size,
                               rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(input_size, hidden_size,
                              rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.outdrop = nn.Dropout(output_dropout)
        self.out = NeuralNet(hidden_size, hidden_size,
                             output_size, output_layer)

    def forward(self, x, x_len):
        # x=self.embed(x)
        # TODO : embedding (extract feature)
        if self.bidirectional:
            direction = 2
        else:
            direction = 1
        h_0 = torch.zeros(self.rnn_layer*direction, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()

        total_len = x.shape[1]
        x = self.indrop(x)
        x = pack_padded_sequence(
            x, x_len, batch_first=self.batch_first)
        if self.lstm:
            x, (_, _) = self.rnn(x, (h_0, c_0))
        else:
            x, _ = self.rnn(x, h_0)
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=total_len)

        x = self.relu(x)
        x = self.outdrop(x)
        x = self.out(x)
        return x


class MyRNN_cat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extract_layer=1, rnn_layer=1, output_layer=1, batch_first=False, input_dropout=0, rnn_dropout=0, output_dropout=0, lstm=False, bidirectional=False, device=torch.device('cpu'), in_label=False):
        super(MyRNN_cat, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.extract_layer = extract_layer
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer
        self.device = device
        self.lstm = lstm
        self.in_label = in_label

        self.extract = NeuralNet(input_size, hidden_size,
                                 hidden_size, extract_layer)
        self.indrop = nn.Dropout(input_dropout)

        if lstm:
            self.rnn = nn.LSTM(hidden_size, hidden_size,
                               rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size,
                              rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.outdrop = nn.Dropout(output_dropout)
        self.out = NeuralNet(hidden_size, hidden_size,
                             output_size, output_layer)

        self.mod = nn.ModuleList([self.extract, self.rnn, self.out])

    def forward(self, x, x_len, y_0, label=None, p=0.0):
        # x=self.embed(x)
        # TODO : embedding (extract feature)
        if self.bidirectional:
            direction = 2
        else:
            direction = 1
        h_0 = torch.zeros(self.rnn_layer*direction, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()

        max_len = int(max(x_len))
        total_len = x.size(1)

        for i in range(max_len):
            if i == 0:
                y = y_0
                h = h_0
                c = c_0
            else:
                if self.in_label and self.training:
                    sampler = torch.distributions.bernoulli.Bernoulli(p)
                    y = (y, label[:, i, :].unsqueeze(1))[
                        int(sampler.sample().item())]

            x_ = torch.cat((x[:, i, :], y.squeeze()), dim=1)

            # cat power
            x_ = torch.cat((x_, x_**2, x_**3, x_**4, x_**5), dim=-1)
            x_ = x_.unsqueeze(1)

            # x_ = self.extract(x_)
            x_ = self.mod[0](x_)
            x_ = F.relu(x_)
            x_ = self.indrop(x_)

            if self.lstm:
                # x_, (h, c) = self.rnn(x_, (h, c))
                x_, (h, c) = self.mod[1](x_, (h, c))
            else:
                # x_, h = self.rnn(x_, h)
                x_, h = self.rnn(x_, h)

            x_ = self.relu(x_)
            x_ = self.outdrop(x_)
            # x_ = self.out(x_)
            x_ = self.mod[2](x_)
            y = x_.clone()
            if i == 0:
                ys = y.clone()
            else:
                ys = torch.cat((ys, y.clone()), dim=1)

        ys = pack_padded_sequence(ys, x_len, batch_first=self.batch_first)
        ys, _ = pad_packed_sequence(
            ys, batch_first=self.batch_first, total_length=total_len)

        return ys


class MyRNN_cat2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extract_layer=1, rnn_layer=1, output_layer=1, batch_first=False, input_dropout=0, rnn_dropout=0, output_dropout=0, lstm=False, bidirectional=False, device=torch.device('cpu'), in_label=False,poly=1):
        super(MyRNN_cat2, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.extract_layer = extract_layer
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer
        self.device = device
        self.lstm = lstm
        self.in_label = in_label
        self.poly=poly
        
        self.extract = NeuralNet(input_size, hidden_size,
                                 hidden_size, extract_layer)
        self.indrop = nn.Dropout(input_dropout)

        if lstm:
            self.rnn = nn.LSTM(hidden_size, hidden_size,
                               rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size,
                              rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.outdrop = nn.Dropout(output_dropout)
        self.out = NeuralNet(hidden_size, hidden_size,
                             output_size, output_layer)

        self.mod = nn.ModuleList([self.extract, self.rnn, self.out])

    def forward(self, x, x_len, y_0, y_1, label=None, p=0.0):
        # x=self.embed(x)
        # TODO : embedding (extract feature)
        if self.bidirectional:
            direction = 2
        else:
            direction = 1
        h_0 = torch.zeros(self.rnn_layer*direction, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()

        max_len = int(max(x_len))
        total_len = x.size(1)
        sampler_1 = torch.distributions.bernoulli.Bernoulli(p)
        a=int(sampler_1.sample().item())
        for i in range(max_len):
            if i == 0:
                y_ = y_0
                y = y_1
                h = h_0
                c = c_0
            else:
                if self.in_label and self.training:
                    sampler = torch.distributions.bernoulli.Bernoulli(p)
                    # y = (y, label[:, i, :].unsqueeze(1))[
                        # int(sampler.sample().item())]
                    y = (y, label[:, i, :].unsqueeze(1))[a]


            x_ = torch.cat((x[:, i, :], y.squeeze(), y_.squeeze()), dim=1)
            
            # cat power
            x_ = torch.cat(([*[x_*i for i in range(1,self.poly+1)]]), dim=-1)
            x_ = x_.unsqueeze(1)

            # x_ = self.extract(x_)
            x_ = self.mod[0](x_)
            x_ = F.relu(x_)
            x_ = self.indrop(x_)

            if self.lstm:
                # x_, (h, c) = self.rnn(x_, (h, c))
                x_, (h, c) = self.mod[1](x_, (h, c))
            else:
                # x_, h = self.rnn(x_, h)
                x_, h = self.rnn(x_, h)

            x_ = self.relu(x_)
            x_ = self.outdrop(x_)
            # x_ = self.out(x_)
            x_ = self.mod[2](x_)
            y = x_.clone()
            y_=y.clone()
            if i == 0:
                ys = y.clone()
            else:
                ys = torch.cat((ys, y.clone()), dim=1)

        ys = pack_padded_sequence(ys, x_len, batch_first=self.batch_first)
        ys, _ = pad_packed_sequence(
            ys, batch_first=self.batch_first, total_length=total_len)

        return ys


class MyRNN_skip(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extract_layer=1, rnn_layer=1, output_layer=1, batch_first=False, input_dropout=0, rnn_dropout=0, output_dropout=0, lstm=False, bidirectional=False, device=torch.device('cpu'), in_label=False, poly=1):
        super(MyRNN_skip, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.extract_layer = extract_layer
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer
        self.device = device
        self.lstm = lstm
        self.in_label = in_label
        self.poly = poly

        self.extract = NeuralNet(input_size, hidden_size,
                                         hidden_size, extract_layer)
        self.indrop = nn.Dropout(input_dropout)

        if lstm:
            self.rnn = nn.LSTM(hidden_size, hidden_size,
                               rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size,
                              rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.outdrop = nn.Dropout(output_dropout)
        # self.out = NeuralNet(hidden_size*2, hidden_size*2,
        #                              output_size, output_layer)
        self.out = NeuralNet(hidden_size, hidden_size,
                             output_size, output_layer)
        self.mod = nn.ModuleList([self.extract, self.rnn, self.out])

    def forward(self, x, x_len, y_0, y_1, label=None, p=0.0):
        if self.bidirectional:
            direction = 2
        else:
            direction = 1
        h_0 = torch.zeros(self.rnn_layer*direction, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()

        max_len = int(max(x_len))
        total_len = x.size(1)

        sampler_1 = torch.distributions.bernoulli.Bernoulli(p)
        a = int(sampler_1.sample().item())

        for i in range(max_len):
            if i == 0:
                y_ = y_0
                y = y_1
                h = h_0
                c = c_0
            else:
                if self.in_label and self.training:
                    # sampler = torch.distributions.bernoulli.Bernoulli(p)
                    # y = (y, label[:, i, :].unsqueeze(1))[
                    #     int(sampler.sample().item())]
                    y = (y, label[:, i, :].unsqueeze(1))[a]

            x_ = torch.cat((x[:, i, :], y.squeeze(), y_.squeeze()), dim=1)
            # x_ = torch.cat((x[:, i, :], y_.squeeze()), dim=1)

            # cat power
            x_ = torch.cat(([*[x_*i for i in range(1, self.poly+1)]]), dim=-1)
            x_ = x_.unsqueeze(1)

            # x_ = self.extract(x_)
            x_ = self.mod[0](x_)
            x_skip = x_.clone()
            x_ = F.relu(x_)
            x_ = self.indrop(x_)

            if self.lstm:
                # x_, (h, c) = self.rnn(x_, (h, c))
                x_, (h, c) = self.mod[1](x_, (h, c))
            else:
                # x_, h = self.rnn(x_, h)
                x_, h = self.mod[1](x_, h)

            x_ = self.relu(x_)
            x_ = self.outdrop(x_)
            # x_ = self.out(x_)

            # skip connection
            # x_ = torch.cat((x_, x_skip), dim=-1)

            x_ = self.mod[2](x_)
            y = x_.clone()
            y_ = y.clone()
            if i == 0:
                ys = y.clone()
                x_skips=x_skip.clone()
            else:
                ys = torch.cat((ys, y.clone()), dim=1)
                x_skips=torch.cat((x_skips,x_skip.clone()),dim=1)
        
        ys = pack_padded_sequence(ys, x_len, batch_first=self.batch_first)
        ys, _ = pad_packed_sequence(
            ys, batch_first=self.batch_first, total_length=total_len)
        x_skips = pack_padded_sequence(x_skips, x_len, batch_first=self.batch_first)
        x_skips, _ = pad_packed_sequence(
            x_skips, batch_first=self.batch_first, total_length=total_len)
        
        return ys,x_skips


class MyRNN_skip_nocat(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extract_layer=1, rnn_layer=1, output_layer=1, batch_first=False, input_dropout=0, rnn_dropout=0, output_dropout=0, lstm=False, bidirectional=False, device=torch.device('cpu'), in_label=False, poly=1):
        super(MyRNN_skip_nocat, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.extract_layer = extract_layer
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer
        self.device = device
        self.lstm = lstm
        self.in_label = in_label
        self.poly = poly

        self.extract = NeuralNet(input_size, hidden_size,
                                 hidden_size, extract_layer)
        self.indrop = nn.Dropout(input_dropout)

        if lstm:
            self.rnn = nn.LSTM(hidden_size, hidden_size,
                               rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(hidden_size, hidden_size,
                              rnn_layer, batch_first=batch_first, dropout=rnn_dropout, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.outdrop = nn.Dropout(output_dropout)
        self.out = NeuralNet(hidden_size, hidden_size,
                             output_size, output_layer)

        self.mod = nn.ModuleList([self.extract, self.rnn, self.out])

    def forward(self, x, x_len,  label=None, p=0.0):
        if self.bidirectional:
            direction = 2
        else:
            direction = 1
        h_0 = torch.zeros(self.rnn_layer*direction, len(x),
                          self.hidden_size).to(self.device)
        c_0 = h_0.clone()

        max_len = int(max(x_len))
        total_len = x.size(1)

        sampler_1 = torch.distributions.bernoulli.Bernoulli(p)
        a = int(sampler_1.sample().item())

        for i in range(max_len):
            if i == 0:
                h = h_0
                c = c_0
            else:
                if self.in_label and self.training:
                    # sampler = torch.distributions.bernoulli.Bernoulli(p)
                    # y = (y, label[:, i, :].unsqueeze(1))[
                    #     int(sampler.sample().item())]
                    y = (y, label[:, i, :].unsqueeze(1))[a]

            # x_ = torch.cat((x[:, i, :], y.squeeze(), y_.squeeze()), dim=1)
            x_=x[:,i,:]
            # cat power
            # x_ = torch.cat(([*[x_*i for i in range(1, self.poly+1)]]), dim=-1)
            x_ = x_.unsqueeze(1)

            # x_ = self.extract(x_)
            x_ = self.mod[0](x_)
            x_skip = x_.clone()
            x_ = F.relu(x_)
            x_ = self.indrop(x_)

            if self.lstm:
                # x_, (h, c) = self.rnn(x_, (h, c))
                x_, (h, c) = self.mod[1](x_, (h, c))
            else:
                # x_, h = self.rnn(x_, h)
                x_, h = self.mod[1](x_, h)

            x_ = self.relu(x_)
            x_ = self.outdrop(x_)
            # x_ = self.out(x_)

            # skip connection
            # x_ = torch.cat((x_, x_skip), dim=-1)

            x_ = self.mod[2](x_)
            y = x_.clone()
            y_ = y.clone()
            if i == 0:
                ys = y.clone()
                x_skips = x_skip.clone()
            else:
                ys = torch.cat((ys, y.clone()), dim=1)
                x_skips = torch.cat((x_skips, x_skip.clone()), dim=1)

        ys = pack_padded_sequence(ys, x_len, batch_first=self.batch_first)
        ys, _ = pad_packed_sequence(
            ys, batch_first=self.batch_first, total_length=total_len)
        x_skips = pack_padded_sequence(
            x_skips, x_len, batch_first=self.batch_first)
        x_skips, _ = pad_packed_sequence(
            x_skips, batch_first=self.batch_first, total_length=total_len)

        return ys, x_skips
