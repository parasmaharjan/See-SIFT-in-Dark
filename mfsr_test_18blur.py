import os, time
import numpy as np
import torch
import skimage.measure as skm
from skimage import color, filters
import matplotlib.pyplot as plt
import glob
import rawpy
from SFNet import SFNet, MBSFNet18
from PIL import Image
import cv2
from PythonSIFT2 import pysift

import easydict
args1 = easydict.EasyDict({
        "n_resblocks": 8,
        "n_feats": 64,
        "patch": 32,
        "res_scale": 1,
        "scale": 2,
        "n_colors": 4,
        "o_colors": 6
})

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

def display(images, index = 0):
    for oct in images:
        _,l,_,_ = oct.shape
        plt.figure()
        for i in range(l):
            plt.subplot(231+i)
            plt.imshow(oct[index,i,:,:].data.cpu(),cmap='gray')

sigma = [1.6, 2.01587, 2.53984, 3.2, 4.03174, 5.07968] #1.6, 2.01587, 2.53984, 3.20, 4.03174, 5.07968
input_dir = '../dataset/Sony/short/'
gt_dir = '../dataset/Sony/long/'
model_dir = './mfsr_model'

test_fns = glob.glob(gt_dir + '1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MBSFNet18(args1).to(device)
model.load_state_dict(torch.load(model_dir + '/model_sid_raw_sony_mfsr_combine_18blur0_5_e0007.pth'))

psnr = []
dog_psnr = []
dog_mse = []
xx = 1000
yy = 100
ps = 256
with torch.no_grad():
    cnt = 0
    for test_id in test_ids:
        # test the first image in each sequence
        test_id = 10125
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
            input_full = np.minimum(input_full, 1.0)
            input_full = input_full[:,yy:yy+ps,xx:xx+ps,:]

            gt_raw = rawpy.imread(gt_path)
            #gt4ch_images = np.expand_dims(pack_raw(gt_raw), axis=0)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = color.rgb2gray(np.float32(im / 65535.0))
            gt_full = gt_full[yy*2:yy*2+ps*2, xx*2:xx*2+ps*2]
            ori_blur, ori_dog = pysift.computeBlurDoG(gt_full)
            gt_full = np.expand_dims(np.expand_dims(gt_full,axis=0),axis=3)
            #gt_full = filters.gaussian(gt_full, sigma, truncate=4)

            # h, w = gt_full.shape
            # l = len(sigma)
            #blur_patch = blur(gt_full)

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)
            gt_img = torch.from_numpy(gt_full).permute(0, 3, 1, 2).to(device)
            gt_img = blur(gt_img)
            model.eval()
            st = time.time()
            out_img = model(in_img)
            print("Time: ", time.time() - st)
            out_dog = dog(out_img)
            gt_dog = dog(gt_img)

            out_img = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            out_dog = out_dog.permute(0, 2, 3, 1).cpu().data.numpy()
            gt_dog = gt_dog.permute(0, 2, 3, 1).cpu().data.numpy()
            #out_img = out_img[0,:,:,:]


            temp = []
            for i in range(6):
                temp.append(skm.compare_psnr(blur_patch[0, :, :, i], out_img[0, :, :, i]))
            psnr.append(temp)

            temp1 = []
            temp2 = []
            for i in range(5):
                temp1.append(skm.compare_psnr(out_dog[0, :, :, i], gt_dog[0, :, :, i]))
                temp2.append(skm.compare_mse(out_dog[0, :, :, i], gt_dog[0, :, :, i]))
            dog_psnr.append(temp1)
            dog_mse.append(temp2)
            cnt += 1
            print(cnt, " psnr: ", psnr[-1], "ratio:", ratio)
            torch.cuda.empty_cache()

print('mean psnr:', np.mean(psnr, axis=0))
print("Done...")

# result = Image.fromarray((out_img * 255).astype(np.uint8))
# result.save('test4.png')

plt.figure(); plt.imshow(out_img[0,:,:,0], cmap='gray')
plt.figure(); plt.imshow(blur_patch[0,:,:,0], cmap='gray')
plt.show()

gaussian_images = []
for octave_index in range(3):
    gaussian_images_in_octave = []
    cnt = 1
    for i in range(6):
        # image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
        # plt.subplot(231 + cnt)
        # plt.imshow(image, cmap='gray')
        # cnt += 1
        to = out_img[octave_index][0,i,:,:].data.cpu().numpy()
        to = np.maximum(np.minimum(to, 1.0),0.0)
        gaussian_images_in_octave.append(to)
    gaussian_images.append(gaussian_images_in_octave)
gaussian_images = np.array(gaussian_images)

np.save('./PythonSIFT2/pred_blur.npy', gaussian_images)