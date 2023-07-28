import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
import imageio

from libs.C2FNet import C2FNet
from utils.dataloader import test_dataset

torch.cuda.is_available()

print(torch.cuda.current_device())

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/C2FNet_3APF_GM/C2FNet-37.pth')

for _data_name in ['CAMO','CHAMELEON','COD10K']: #'CAMO','CHAMELEON','COD10K'
    data_path = './data/TestDataset/{}/'.format(_data_name)
    res_save_path = './results/test37/{}/'.format(_data_name)
    edge_save_path = './results/C2FNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = C2FNet()
    # model = torch.nn.DataParallel(model)
    # torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    model.load_state_dict(torch.load(opt.pth_path,map_location='cpu'))
    model.cuda()
    model.eval()

    os.makedirs(res_save_path, exist_ok=True)
    os.makedirs(edge_save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res , pre3 = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        res *= 255
        res = res.astype(np.uint8)
        imageio.imsave(res_save_path+name, res)




        # ppm = F.upsample(ppm, size=gt.shape, mode='bilinear', align_corners=False)
        # ppm = ppm.sigmoid().data.cpu().numpy().squeeze()
        # ppm = (ppm - ppm.min()) / (ppm.max() - ppm.min() + 1e-8)
        #
        # ppm *= 255
        # ppm = ppm.astype(np.uint8)
        # imageio.imsave(edge_save_path + name, ppm)
