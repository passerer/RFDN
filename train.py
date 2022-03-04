import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from model.IMDN import IMDN, KD
from model.RFDN import RFDN
from model.repconv import ac_conv_layer, expand_conv_layer
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description="RFDN")
parser.add_argument("--name", type=str, help="name to record")
parser.add_argument("--batch_size", type=int, default=64,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=2000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=5e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--test-every", type=int, default=1000)
parser.add_argument("--root", type=str, default="Train_Datasets/DIV2K_decoded/",
                    help='dataset directory')
parser.add_argument("--extra-root", type=str, default="Train_Datasets/Flickr2K/",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=256,
                    help="hr image size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--isY", action="store_true", default=False)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--tb_logger", action="store_true", default=True)

args = parser.parse_args()
print(args)
if args.tb_logger:
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.name)

#torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset1 = DIV2K.div2k(args)
trainset2 = DIV2K.Flickr(args,root=args.extra_root)
trainset = DIV2K.RepeatDataset(ConcatDataset([trainset1,trainset2]))
testset = Set5_val.DatasetFromFolderVal("DIV2k_val/DIV2K_valid_HR/",
                                       "DIV2k_val/DIV2K_valid_LR_bicubic/X{}/".format(args.scale),
                                       args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print("===> Building models")
args.is_train = True
# teacher = IMDN(nc=64)
#
# if args.pretrained:
#     if os.path.isfile(args.pretrained):
#         print("===> loading models '{}'".format(args.pretrained))
#         checkpoint = torch.load(args.pretrained)
#         new_state_dcit = OrderedDict()
#         for k, v in checkpoint.items():
#             if 'module' in k:
#                 name = k[7:]
#             else:
#                 name = k
#             new_state_dcit[name] = v
#         model_dict = teacher.state_dict()
#         pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}
#
#         for k, v in model_dict.items():
#             if k not in pretrained_dict:
#                 print(k)
#         # for k, v in pretrained_dict.items():
#         #     if 'conv' in k and 'weight' in k:
#         #         vv = []
#         #         vv.append(v)
#         #         for i in range(1, 4):
#         #             vv.append((v+torch.roll(v,i,0))/2)
#         #         vv = torch.cat(vv,dim=0)
#         #         pretrained_dict[k] = vv
#
#         teacher.load_state_dict(pretrained_dict, strict=False)
#
#     else:
#         print("===> no models found at '{}'".format(args.pretrained))

#student = IMDN(nc=48)
#model = IMDN(nc=48)
model = RFDN(nf=50)
#model = KD(teacher=teacher,student=student)
print("===> Setting GPU")
if cuda:
    model = model.to(device)
if torch.cuda.device_count()>1:
    model = torch.nn.DataParallel(model)
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):
        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        #sr_tensor, loss_feat  = model(lr_tensor)
        sr_tensor  = model(lr_tensor)
        if epoch <= 1000:
            loss_pixel = F.l1_loss(sr_tensor, hr_tensor)
        else:
            loss_pixel = F.mse_loss(sr_tensor, hr_tensor)
        #loss_sr = loss_pixel + loss_feat*min(1,epoch/200)
        loss_sr = loss_pixel
        loss_sr.backward()
        # param_l1 = 0
        # prune_ratio = 0
        # for name, param in model.named_parameters():
        #     if 'logits' in name:
        #         param.grad = param.grad + param.abs()*optimizer.param_groups[0]['lr']*1e-2
        #         param_l1 += param.abs().sum()
        #         prune_ratio = (param.abs()<1e-5).sum()/len(param)
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f} Loss_feat: {:.5f}".format\
                      (epoch, iteration, len(training_data_loader),loss_pixel.item(),0))
        #    print("Param_l1: {:.0f} prune_ratio:{:.2f}".format(param_l1.item(),prune_ratio.item()))
            if args.tb_logger:
                tb_logger.add_scalar('loss', loss_sr.item(), (epoch-1)*len(training_data_loader)+iteration)
            #    tb_logger.add_scalar('flops', flops.item(), (epoch - 1) * len(training_data_loader) + iteration)


def valid(epoch):
    model.eval()
    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0]) # tensor -> ndarray
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size) # get rid of border pixels
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True: # whether to calculate psnr and ssim in ycbcr format
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    avg_psnr = avg_psnr / len(testing_data_loader)
    avg_ssim = avg_ssim / len(testing_data_loader)
    #scheduler.step(avg_psnr)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr, avg_ssim))
    if args.tb_logger:
        tb_logger.add_scalar('psnr', avg_psnr, (epoch - 1) )
    return avg_psnr

def save_checkpoint(epoch):
    model_folder = args.name+"_checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # for layer in model.modules():
    #     if isinstance(layer, ac_conv_layer) or isinstance(layer, expand_conv_layer):
    #         layer.rep()
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    from summery import get_model_activation, get_model_flops
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv2d = get_model_activation(net, input_dim)
    print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations / 10 ** 6))
    print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))
    flops = get_model_flops(net, input_dim, False)
    print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops / 10 ** 9))
    num_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    print('{:>16s} : {:<.4f} [K]'.format('#Params', num_parameters / 10 ** 3))


print("===> Training")
#print_network(model.student.model)
for epoch in range(args.start_epoch, args.nEpochs + 1):
    if epoch % 20 == 0:
        psnr = valid(epoch)
    train(epoch)
save_checkpoint(args.nEpochs)
