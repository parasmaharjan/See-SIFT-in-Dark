import os, time
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from skimage import filters, color
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
import glob
import rawpy
from SFNet import SFNet, MBSFNet18
import torch.optim as optim
import h5py
import torch.utils.data as data
import cv2
#import pytorch_ssim
from model import SeeInDark

# import easydict
# args = easydict.EasyDict({
#         "n_resblocks": 32,
#         "n_feats": 64,
#         "patch": 256,
#         "sigma": 1.6,
#         "assumed_sigma": 0.5,
#         "num_interval": 3,
#         "octav": 3,
#         "res_scale": 1,
#         "scale": 2,
#         "n_colors": 4,
#         "o_colors": 6
# })

#from se_modle import SELayer
save_freq = 1
num_epochs = 2001
patch_size = 512
#batch_size = 4
learning_rate = 0.0001
#data_path = './train.h5'#'./sid_train_patch.h5'
input_dir = '../dataset/Sony/short/'
gt_dir = '../dataset/Sony/long/'
result_dir = './ssid_result'
model_dir = './ssid_model'

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

def blur(image, sigma=1.6, assumed_blur = 0.5):
    # 1. octave
    num_octaves = 3
    num_intervals = 3

    # 2. compute gaussian kernels
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma
    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)

    # 3. base iamge
    image = image.data.cpu()
    image = image.numpy() * 255.0
    b, l, h, w = image.shape
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    x1 = np.zeros((b, 6, h * 2, w * 2))
    x2 = np.zeros((b, 6, h, w))
    x3 = np.zeros((b, 6, int(h / 2), int(w / 2)))
    for i in range(b):
        img = cv2.resize(image[i,0,:,:], (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        base_image = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

        # octave 1
        x1[i,0,:,:] = base_image
        # plt.figure();
        # plt.subplot(231);
        # plt.imshow(x1[i, 0, :, :], cmap='gray')
        for j in range(1,6):
            base_image = cv2.GaussianBlur(base_image, (0, 0), sigmaX=gaussian_kernels[j], sigmaY=gaussian_kernels[j])
            x1[i, j, :, :] = base_image
            # plt.subplot(231 + j);
            # plt.imshow(x1[i, j, :, :], cmap='gray')

        # octave 2
        base_image = x1[i, 3, :, :]
        base_image = cv2.resize(base_image, (int(base_image.shape[1] / 2), int(base_image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        x2[i, 0, :, :] = base_image
        # plt.figure();
        # plt.subplot(231);
        # plt.imshow(x2[i, 0, :, :], cmap='gray')
        for j in range(1, 6):
            base_image = cv2.GaussianBlur(base_image, (0, 0), sigmaX=gaussian_kernels[j], sigmaY=gaussian_kernels[j])
            x2[i, j, :, :] = base_image
            # plt.subplot(231 + j);
            # plt.imshow(x2[i, j, :, :], cmap='gray')

        # octave 3
        base_image = x2[i, 3, :, :]
        base_image = cv2.resize(base_image, (int(base_image.shape[1] / 2), int(base_image.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
        x3[i, 0, :, :] = base_image
        # plt.figure();
        # plt.subplot(231);
        # plt.imshow(x3[i, 0, :, :], cmap='gray')
        for j in range(1, 6):
            base_image = cv2.GaussianBlur(base_image, (0, 0), sigmaX=gaussian_kernels[j], sigmaY=gaussian_kernels[j])
            x3[i, j, :, :] = base_image
            # plt.subplot(231 + j);
            # plt.imshow(x3[i, j, :, :], cmap='gray')

    return torch.from_numpy(x1/255.0).float().to(device), torch.from_numpy(x2/255.0).float().to(device), torch.from_numpy(x3/255.0).float().to(device)

def dog(img):
    b,l,h,w = img[0].shape
    x1 = torch.zeros((b,l-1,h,w)).cuda()
    temp = torch.clamp(img[0], max=1.0, min=0.0)
    for i in range(b):
        cnt = 0
        for first_image, second_image in zip(temp[i,:,:,:], temp[i, 1:, :, :]):
            x1[i,cnt, :,:] = (torch.sub(second_image, first_image))
            cnt += 1
    b,l,h,w = img[1].shape
    x2 = torch.zeros((b,l-1,h,w)).cuda()
    temp = torch.clamp(img[1], max=1.0, min=0.0)
    for i in range(b):
        cnt = 0
        for first_image, second_image in zip(temp[i,:,:,:], temp[i, 1:, :, :]):
            x2[i,cnt, :,:] = (torch.sub(second_image, first_image))
            cnt += 1
    b,l,h,w = img[2].shape
    x3 = torch.zeros((b,l-1,h,w)).cuda()
    temp = torch.clamp(img[2], max=1.0, min=0.0)
    for i in range(b):
        cnt = 0
        for first_image, second_image in zip(temp[i,:,:,:], temp[i, 1:, :, :]):
            x3[i,cnt, :,:] = (torch.sub(second_image, first_image))
            cnt += 1
    return x1, x2, x3

def pack_raw(im):
    # pack Bayer image to 4 channels
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


test_name = 'unet_upscale_blur/' #'edsr-lerelu-ps-'+str(args.patch_size)+'-b-'+str(args.n_resblocks)+'/'

# get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

ps = args.patch_size  # patch size for training
save_freq = 1

def display(images, index = 0):
    for oct in images:
        _,l,_,_ = oct.shape
        plt.figure()
        for i in range(l):
            plt.subplot(231+i)
            plt.imshow(oct[index,i,:,:].data.cpu(),cmap='gray')

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SeeInDark()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.cuda()

opt = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(lastepoch, 4001):
    if os.path.isdir("result/%04d" % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        in_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2).cuda()
        gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).cuda()

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        g_loss[ind] = loss.data

        print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_freq == 0:
            #if not os.path.isdir(result_dir + '%04d' % epoch):
            #    os.makedirs(result_dir + '%04d' % epoch)
            if not os.path.isdir(model_dir + test_name):
                os.makedirs(model_dir + test_name)
            torch.save(model.state_dict(), model_dir + test_name + 'unet_upscale_blur_e%04d.pth' % epoch)



# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = MBSFNet18(args1).to(device)
# #model.load_state_dict(torch.load(model_dir + '/model_sid_raw_sony_mfsr_combine_18blur0_5_e0016.pth'))
# opt = optim.Adam(model.parameters(), lr=learning_rate)
# #ssim_loss = pytorch_ssim.SSIM(window_size=11)
# los = []
# losn = []
#
# for epoch in range(num_epochs):
#     model.train()
#
#     for j, (input_patch, gt_patch) in enumerate(train_loader):
#
#         input_patch = input_patch.permute(0, 3, 1, 2).to(device)
#         gt_patch = gt_patch.permute(0, 3, 1, 2)
#         gt_patch = blur(gt_patch)
#         # Forward + Backward + Optimize
#         opt.zero_grad()
#         outputs = model(input_patch)
#         out_dog = dog(outputs)
#         gt_dog = dog(gt_patch)
#         #s_loss = 1 - ssim_loss(outputs, gt_patch)
#         l1_loss = reduce_mean(outputs[0], gt_patch[0]) + reduce_mean(outputs[1], gt_patch[1]) + reduce_mean(outputs[2], gt_patch[2])
#         dog_loss = reduce_mean(out_dog[0], gt_dog[0])+reduce_mean(out_dog[1], gt_dog[1])+reduce_mean(out_dog[2], gt_dog[2])
#         loss = l1_loss + dog_loss
#
#         loss.backward()
#         opt.step()
#         losn.append(loss.item())
#         print('Epoch [%d/%d], Iter: %d,  L1: %.8f  ' % (epoch + 1, num_epochs, j+1, losn[-1]))
#         torch.cuda.empty_cache()
#     if epoch % save_freq == 0:
#         # plt.plot(losn)
#         # plt.show()
#         if not os.path.isdir(model_dir):
#             os.makedirs(model_dir)
#         torch.save(model.state_dict(), model_dir + '/model_sid_raw_sony_mfsr_combine_18blur0_5_e%04d.pth' % (epoch))
#         print('mean l1 loss: ', np.mean(losn[-1000:]))
# np.save(model_dir +"/mfsr_L1_sigma_combine_18blur0_5", losn)
# print("Done...")
#
# outputs = torch.clamp(outputs, max=1.0, min=0.0)
# out_img = outputs.permute(0, 2, 3, 1).cpu().data.numpy()
# gt_img = gt_patch.permute(0, 2, 3, 1).cpu().data.numpy()
# plt.figure(); plt.imshow(out_img[3,:,:,0], cmap='gray')
# plt.figure(); plt.imshow(gt_img[3,:,:,0], cmap='gray')
# plt.show()
