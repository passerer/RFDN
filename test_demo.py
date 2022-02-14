import os.path
import logging
import time
from collections import OrderedDict
import torch

import utils as util
from model.IMDN import IMDN
from model.RFDN import RFDN
from tqdm import tqdm

def main():

    # --------------------------------
    # basic settings
    # --------------------------------
    # testsets = 'DIV2K'
    testsets = '/home/cc/RFDN/DIV2k_val'
    testset_L = 'DIV2K_valid_LR_bicubic'
    testset_H = 'DIV2K_valid_HR'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join('RFDN_checkpoint_x4', 'epoch_600.pth')
    #model = IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=4)
    model = RFDN(deploy=True)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('Params number: {} K'.format(number_parameters/1e3))

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L, 'X4')
    H_folder = os.path.join(testsets, testset_H)
  #  E_folder = os.path.join(testsets, testset_L+'_results')
  #  util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []
    test_results['psnr'] = []
    test_results['ssim'] = []

    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for lr_img in tqdm(util.get_image_paths(L_folder)):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
    #    img_name, ext = os.path.splitext(os.path.basename(lr_img))
        img_L = util.imread_uint(lr_img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        
        img_H = util.imread_uint(os.path.join(H_folder, os.path.basename(lr_img).replace("x4","")), n_channels=3)
        img_H = util.modcrop(img_H, 4)

        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)
        img_H = util.shave(img_H,4)
        img_E = util.shave(img_E,4)
        test_results['psnr'] .append(util.calculate_psnr(img_H, img_E))
        test_results['ssim'] .append(util.calculate_ssim(img_H, img_E))

    #    util.imsave(img_E, os.path.join(E_folder, img_name[:4]+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    print('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))
    print('------> Average runtime psnr is : {:.2f}, ssim is : {:.4f}'.format(ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()