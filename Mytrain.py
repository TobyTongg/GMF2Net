import torch
from torch.autograd import Variable
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from libs.C2FNet import C2FNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, poly_lr, AvgMeter
import torch.nn.functional as F
import torch.nn as nn
from utils.AdaX import AdaXW



torch.cuda.is_available()


x_label = []
y_label = []

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()



def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, edge = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            #edges = Variable(edges).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                #edges = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            fout, pre4 = model(images)
            # ---- loss function ----
            loss_out = structure_loss(fout, gts)
            loss_pre4 = structure_loss(pre4, gts)

            # criteria = nn.BCEWithLogitsLoss()
            # loss_edge = criteria(edge_out, edges)

            loss = loss_out + loss_pre4


            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record1.update(loss_out.data, opt.batchsize)
                loss_record2.update(loss_pre4.data, opt.batchsize)


        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
               print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[loss: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss.data.item()))
    y_label.append(loss.data.item())
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if epoch  % 1 == 0 and epoch > 20 :

        torch.save(model.state_dict(), save_path + 'C2FNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'C2FNet-%d.pth'% epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=60, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path tol train dataset')
    parser.add_argument('--train_save', type=str,
                        default='C2FNet_3APFGM_RFBtest')
    opt = parser.parse_args()

    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = C2FNet().cuda()

    params = model.parameters()
    optimizer = AdaXW(params, opt.lr)

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training...l")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
        x_label.append(epoch)
    plt.title("train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x_label, y_label)
    plt.show()

