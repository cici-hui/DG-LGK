import sys
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from util.loader.CityLoader import CityLoader
from util.loader.GTA5Loader import GTA5Loader
from util.loader.augmentations import Compose, RandomHorizontallyFlip, RandomSized_and_Crop, \
    RandomCrop, ColorChange, BrightChange, SharpChange, ContrastChange
from util.metrics import runningScore
from util.loss import cross_entropy2d, cross_entropy_loss2d
from model.model_deviation import SharedEncoder
from util.utils import adjust_learning_rate, save_models

# Data-related
LOG_DIR = './log'
GEN_IMG_DIR = './generated_imgs'

GTA5_DATA_PATH = '/workspace/lustre/data/GTA5'
CITY_DATA_PATH = '/workspace/lustre/data/Cityscapes'
DATA_LIST_PATH_GTA5 = './util/loader/gta5_list/train_modified.txt'
DATA_LIST_PATH_VAL_IMG = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL = './util/loader/cityscapes_list/val_label.txt'

# Hyper-parameters
CUDA_DIVICE_ID = '0'

parser = argparse.ArgumentParser(description='LGK')
parser.add_argument('--dump_logs', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='the path to where you save plots and logs.')
parser.add_argument('--gen_img_dir', type=str, default=GEN_IMG_DIR,
                    help='the path to where you save translated images and segmentation maps.')
parser.add_argument('--gta5_data_path', type=str, default=GTA5_DATA_PATH, help='the path to GTA5 dataset.')
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to Cityscapes dataset.')
parser.add_argument('--data_list_path_gta5', type=str, default=DATA_LIST_PATH_GTA5)
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)

parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)

args = parser.parse_args()

print('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.exists(args.gen_img_dir):
    os.makedirs(args.gen_img_dir)

if args.dump_logs == True:
    old_output = sys.stdout
    sys.stdout = open(os.path.join(args.log_dir, 'output.txt'), 'w')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

num_classes = 19
source_input_size = [640, 640]
target_input_size = [512, 1024]
batch_size = 2

max_epoch = 150
num_steps = 200000
num_calmIoU = 1000
learning_rate_seg = 2.5e-4
power = 0.9
weight_decay = 0.0005

synthia_data_aug = Compose([BrightChange(),
                            SharpChange(),
                            ContrastChange(),
                            ColorChange(),
                            RandomSized_and_Crop([640, 640])
                            ])

city_data_aug = Compose([RandomHorizontallyFlip(),
                         RandomCrop([256, 512])
                         ])

# ==== DataLoader ====
gta5_set = GTA5Loader(args.gta5_data_path, args.data_list_path_gta5,
                            max_iters=num_steps * batch_size, crop_size=source_input_size, transform=synthia_data_aug,
                            mean=IMG_MEAN)
source_loader = torch_data.DataLoader(gta5_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

val_set = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None,
                     crop_size=[512, 1024], mean=IMG_MEAN, set='val')
val_loader = torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

sourceloader_iter = enumerate(source_loader)

cty_running_metrics = runningScore(num_classes)

model_dict = {}

# Setup Model
print('building models ...')

enc_shared = SharedEncoder().cuda()

model_dict['enc_shared'] = enc_shared

enc_shared_opt = optim.SGD(enc_shared.optim_parameters(learning_rate_seg), lr=learning_rate_seg, momentum=0.9,
                           weight_decay=weight_decay)

seg_opt_list = []

seg_opt_list.append(enc_shared_opt)
cudnn.enabled = True
cudnn.benchmark = True

mse_loss = nn.MSELoss(size_average=True).cuda()
sg_loss = cross_entropy2d
boundary_loss = cross_entropy_loss2d

upsample_256 = nn.Upsample(size=[640, 640], mode='bilinear')
upsample_512 = nn.Upsample(size=[512, 1024], mode='bilinear')

i_iter_tmp = []
epoch_tmp = []

loss_sim_sg_tmp = []
loss_boundary_tmp = []
loss_similarity_tmp = []
City_tmp = []

enc_shared.train()

best_iou = 0
best_iter = 0

for i_iter in range(num_steps):
    print(i_iter)
    sys.stdout.flush()

    enc_shared.train()
    adjust_learning_rate(seg_opt_list, base_lr=learning_rate_seg, i_iter=i_iter, max_iter=num_steps, power=power)

    idx_t, source_batch = next(sourceloader_iter)

    img, notran_data, target_label, sboundary = source_batch

    img = Variable(img).cuda()
    notran_data = Variable(notran_data).cuda()
    slabelv = Variable(target_label).cuda()
    sboundary_gt = Variable(sboundary).cuda()

    class3_contrastive, class_contrastive, s_pred2, t_pred2, s_boundary, boundary_fuse, boundary_4, boundary_3, boundary_2, boundary_1 = enc_shared(
        img, slabelv)

    class3_contrastive_o, class_contrastive_o, _, _, _, _, _, _, _, _ = enc_shared(notran_data, slabelv)

    class3_contrastive = torch.squeeze(class3_contrastive, dim=0)
    class3_contrastive_o = torch.squeeze(class3_contrastive_o, dim=0)

    class3_contrastive_1 = (class3_contrastive ** 2).sum(dim=-1)
    class3_contrastive_1 = torch.unsqueeze(class3_contrastive_1, dim=1)
    class3_contrastive_2 = class3_contrastive_1.permute(1, 0)

    class3_contrastive_o_1 = (class3_contrastive_o ** 2).sum(dim=-1)
    class3_contrastive_o_1 = torch.unsqueeze(class3_contrastive_o_1, dim=1)
    class3_contrastive_o_2 = class3_contrastive_o_1.permute(1, 0)

    class3_contrastive_ab = torch.mm(class3_contrastive_1, class3_contrastive_2)
    class3_contrastive_o_ab = torch.mm(class3_contrastive_o_1, class3_contrastive_o_2)

    class3_contrastive_o = class3_contrastive_o.permute(1, 0)
    contrastive_metric3 = torch.mm(class3_contrastive, class3_contrastive_o)
    contrastive_metric3 = contrastive_metric3 / (
                torch.sqrt(torch.sqrt(class3_contrastive_ab * class3_contrastive_o_ab + 1e-10) + 1e-10) + 1e-10)

    pos3 = torch.diagonal(torch.exp(contrastive_metric3 / 0.2)).sum()
    neg3 = torch.exp(contrastive_metric3 / 0.2).sum()
    loss_similarity3 = -torch.log(pos3 / neg3)

    class_contrastive = torch.squeeze(class_contrastive, dim=0)
    class_contrastive_o = torch.squeeze(class_contrastive_o, dim=0)

    class_contrastive_1 = (class_contrastive ** 2).sum(dim=-1)
    class_contrastive_1 = torch.unsqueeze(class_contrastive_1, dim=1)
    class_contrastive_2 = class_contrastive_1.permute(1, 0)

    class_contrastive_o_1 = (class_contrastive_o ** 2).sum(dim=-1)
    class_contrastive_o_1 = torch.unsqueeze(class_contrastive_o_1, dim=1)
    class_contrastive_o_2 = class_contrastive_o_1.permute(1, 0)

    class_contrastive_ab = torch.mm(class_contrastive_1, class_contrastive_2)
    class_contrastive_o_ab = torch.mm(class_contrastive_o_1, class_contrastive_o_2)

    class_contrastive_o = class_contrastive_o.permute(1, 0)
    contrastive_metric = torch.mm(class_contrastive, class_contrastive_o)
    contrastive_metric = contrastive_metric/(torch.sqrt(torch.sqrt(class_contrastive_ab * class_contrastive_o_ab + 1e-10)+ 1e-10) + 1e-10)
    pos = torch.diagonal(torch.exp(contrastive_metric/0.2)).sum()
    neg = torch.exp(contrastive_metric/0.2).sum()
    loss_similarity4 = -torch.log(pos / neg)
    loss_similarity = loss_similarity3 + loss_similarity4

    s_pred2 = upsample_256(s_pred2)
    t_pred2 = upsample_256(t_pred2)

    loss_s_sg2 = sg_loss(s_pred2, slabelv)
    loss_t_sg2 = sg_loss(t_pred2, slabelv)
    loss_sim_sg = 0.5 * loss_s_sg2 + loss_t_sg2
    
    loss_boundary = torch.zeros(1).cuda()
    for o in s_boundary:
        loss_boundary = loss_boundary + boundary_loss(o, sboundary_gt)

    pred_s = F.softmax(s_pred2, dim=1).data.max(1)[1].cpu().numpy()
    pred_t = F.softmax(t_pred2, dim=1).data.max(1)[1].cpu().numpy()

    map_s = gta5_set.decode_segmap(pred_s)
    map_t = gta5_set.decode_segmap(pred_t)

    gt_s = slabelv.data.cpu().numpy()
    gt_t = slabelv.data.cpu().numpy()
    gt_s = gta5_set.decode_segmap(gt_s)
    gt_t = gta5_set.decode_segmap(gt_t)

    total_loss = 1.0 * loss_sim_sg + 1.5 * loss_boundary + 0.5 * loss_similarity

    enc_shared_opt.zero_grad()

    total_loss.backward()

    enc_shared_opt.step()

    if i_iter % 25 == 0:
        i_iter_tmp.append(i_iter)
        print('Best Iter : ' + str(best_iter))
        print('Best mIoU : ' + str(best_iou))

        plt.title('segmentation_loss')
        loss_sim_sg_tmp.append(loss_sim_sg.item())
        plt.plot(i_iter_tmp, loss_sim_sg_tmp, label='loss_sim_sg')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'segmentation_loss.png'))
        plt.close()

        plt.title('mIoU')
        plt.plot(epoch_tmp, City_tmp, label='City')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'mIoU.png'))
        plt.close()

    if i_iter % 500 == 0:
        imgs_s = torch.cat(((img[:, [2, 1, 0], :, :].cpu() + 1) / 2,
                            Variable(torch.Tensor((map_s.transpose((0, 3, 1, 2))))),
                            Variable(torch.Tensor((gt_s.transpose((0, 3, 1, 2)))))), 0)
        imgs_s = vutils.make_grid(imgs_s.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_s = np.clip(imgs_s * 255, 0, 255).astype(np.uint8)
        imgs_s = imgs_s.transpose(1, 2, 0)
        imgs_s = Image.fromarray(imgs_s)
        filename = '%05d_source.jpg' % i_iter
        imgs_s.save(os.path.join(args.gen_img_dir, filename))

    if i_iter % num_calmIoU == 0:
        enc_shared.eval()
        print('evaluating models ...')
        precision = 0
        recall = 0
        for i_val, (images_val, labels_val, edge_label) in tqdm(enumerate(val_loader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val, volatile=True)

            pred = enc_shared(images_val)
            pred = upsample_512(pred)
            pred = pred.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            cty_running_metrics.update(gt, pred)
            gt_val_show = gta5_set.decode_segmap(gt)
            pred_val_show = gta5_set.decode_segmap(pred)

            imgs_val = torch.cat(((images_val[:, [2, 1, 0], :, :].cpu() + 1) / 2,
                                  Variable(torch.Tensor((gt_val_show.transpose((0, 3, 1, 2))))),
                                  Variable(torch.Tensor((pred_val_show.transpose((0, 3, 1, 2)))))), 0)
            imgs_val = vutils.make_grid(imgs_val.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
            imgs_val = np.clip(imgs_val * 255, 0, 255).astype(np.uint8)
            imgs_val = imgs_val.transpose(1, 2, 0)
            imgs_val = Image.fromarray(imgs_val)
            filename = '%05d_val.jpg' % (i_iter)
            imgs_val.save(os.path.join(args.gen_img_dir, filename))

        cty_score, cty_class_iou = cty_running_metrics.get_scores()

        for k, v in cty_score.items():
            print(k, v)

        cty_running_metrics.reset()
        City_tmp.append(cty_score['Mean IoU : \t'])
        epoch_tmp.append(i_iter)

        if cty_score['Mean IoU : \t'] > best_iou:
            best_iter = i_iter
            best_iou = cty_score['Mean IoU : \t']
            save_models(model_dict, './weight/')
