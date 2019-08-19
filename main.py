# Author: Soheil Khorram
# License: Simplified BSD

# A pytorch implimentation of a GAN-based speech enhancement
# To run this code you need to write the Dataset class which is a simple iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

FEAT_T = 2500
FEAT_DIM = 256
GEN_LAYER_NUM = 7
GEN_CHANNEL_IN = 256
GEN_CHANNEL_OUT = 256
GEN_KER_SIZE = 7
DIS_LAYER_NUM = 5
DIS_CHANNEL_IN = 256
DIS_CHANNEL_OUT = 256
DIS_KER_SIZE = 5
USE_GPU = True
SEED = 0
BATCH_SIZE = 64
TRAIN_DATA_DIR = '../data/train'
TEST_DATA_DIR = '../data/test'
LR = 0.0001
EPOCH_NUM = 50
GEN_SUBEPOCH_NUM = 1
DIS_SUBEPOCH_NUM = 4
DIS_LOSS_WEIGHT = 0.5
GEN_LOSS_WEIGHT = 0.5


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.conv_layers = [
            nn.Conv1d(
                in_channels=(GEN_CHANNEL_IN if (layer != 1) else FEAT_DIM),
                out_channels=(GEN_CHANNEL_OUT if (layer != GEN_LAYER_NUM - 1) else FEAT_DIM),
                kernel_size=GEN_KER_SIZE,
                padding=GEN_KER_SIZE//2
            ) for layer in xrange(GEN_LAYER_NUM)]

    def forward(self, x):
        inp = x
        for layer in xrange(GEN_LAYER_NUM):
            x = self.conv_layers[layer](x)
            if layer != GEN_LAYER_NUM - 1:
                x = F.relu(x)
        outp = x * inp
        return outp


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv_layers = [
            nn.Conv1d(
                in_channels=(DIS_CHANNEL_IN if (layer != 1) else FEAT_DIM * 2),
                out_channels=(DIS_CHANNEL_OUT if (layer != GEN_LAYER_NUM - 1) else 1),
                kernel_size=DIS_KER_SIZE,
                padding=DIS_KER_SIZE//2
            ) for layer in xrange(DIS_LAYER_NUM)]

    def forward(self, x):
        for layer in xrange(DIS_LAYER_NUM):
            x = self.conv_layers[layer](x)
            if layer != DIS_LAYER_NUM - 1:
                x = F.relu(x)
        outp = F.sigmoid(x)
        return outp


def gen_train(gen_model, dis_model, device, train_loader, gen_optimizer, epoch):
    gen_model.train()
    dis_model.eval()
    for batch_idx, (noisy, clean) in enumerate(train_loader):
        noisy, clean = noisy.to(device), clean.to(device)
        gen_optimizer.zero_grad()
        gen_output = gen_model(noisy)
        dis_output = dis_model(torch.cat([gen_output, noisy], 1))
        mse_loss = torch.sum((gen_output - clean) ** 2)
        dis_loss = torch.sum((dis_output - (dis_output * 0 + 1)) ** 2)
        loss = DIS_LOSS_WEIGHT * dis_loss + GEN_LOSS_WEIGHT * dis_loss
        loss.backward()
        gen_optimizer.step()
        print('Train Generator Epoch: {}, mse_loss: {}, dis_loss: {}, loss: {}'.format(epoch, mse_loss, dis_loss, loss))


def dis_train(gen_model, dis_model, device, train_loader, dis_optimizer, epoch):
    gen_model.eval()
    dis_model.train()
    for batch_idx, (noisy, clean) in enumerate(train_loader):
        noisy, clean = noisy.to(device), clean.to(device)
        dis_optimizer.zero_grad()
        gen_output = gen_model(noisy)
        dis_output = dis_model(torch.cat([gen_output, noisy], 1))
        loss = torch.sum((dis_output - (dis_output * 0)) ** 2)
        loss.backward()
        dis_optimizer.step()
        dis_optimizer.zero_grad()
        dis_output = dis_model(torch.cat([clean, noisy], 1))
        loss = torch.sum((dis_output - (dis_output * 0 + 1)) ** 2)
        loss.backward()
        dis_optimizer.step()
        print('Train Discriminator Epoch: {}, loss: {}'.format(epoch, loss))


def test(gen_model, device, test_loader):
    gen_model.eval()
    test_loss = 0
    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            gen_output = gen_model(noisy)
            mse_loss = torch.sum((gen_output - clean) ** 2).item()
            test_loss += mse_loss
    print('test_loss = {}'.format(test_loss))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        pass

    def __getitem__(self, index):
        # returns pairs of (noisy, clean)
        # make sure to use torch.from_numpy() to convert np to torch
        pass

    def __len__(self):
        pass


def main():
    use_cuda = USE_GPU and torch.cuda.is_available()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader = torch.utils.data.DataLoader(Dataset(TRAIN_DATA_DIR), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(TEST_DATA_DIR), batch_size=BATCH_SIZE, shuffle=True)
    gen_model = GeneratorNet().to(device)
    dis_model = DiscriminatorNet().to(device)
    gen_optimizer = optim.Adam(gen_model.parameters(), lr=LR)
    dis_optimizer = optim.Adam(dis_model.parameters(), lr=LR)
    for epoch in xrange(EPOCH_NUM):
        for gen_epoch in range(GEN_SUBEPOCH_NUM):
            gen_train(gen_model, dis_model, device, train_loader, gen_optimizer, epoch)
        for dis_epoch in range(DIS_SUBEPOCH_NUM):
            dis_train(gen_model, dis_model, device, train_loader, dis_optimizer, epoch)
        test(gen_model, dis_model, device, test_loader)
