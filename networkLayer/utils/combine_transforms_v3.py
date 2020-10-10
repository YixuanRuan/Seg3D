import torch
import random
import numpy as np
from torchvision.transforms.functional import normalize

from PIL import Image, ImageOps
import pdb


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img_in = sample['I']
        img_seg = sample['S']

        img_in = normalize(img_in, self.mean, self.std)

        return {'I': img_in,
                'S': img_seg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
        Fossil
    """

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = sample['I']
        img_seg = sample['S']
        imgseq = sample['I_seq']
        segseq = sample['S_seq']

        img_seg = np.array(img_seg)
        img_in = np.array(img_in).astype(np.float32) / 255.
        imgseq = [np.array(img).astype(np.float32) / 255. for img in imgseq]
        segseq = [np.array(img) for img in segseq]

        img_shape = img_in.shape

        if len(img_shape) == 3:
            img_in = img_in.transpose((2, 0, 1))
            imgseq = np.concatenate(imgseq, axis=-1).transpose((2, 0, 1))

            img_in = torch.from_numpy(img_in).float()
            imgseq = torch.from_numpy(imgseq).float()
            img_seg = torch.from_numpy(img_seg).long()
        else:
            img_in = torch.from_numpy(img_in).float().unsqueeze(0)
            imgseq = [img.unsqueeze(0) for img in imgseq]
            imgseq = np.concatenate(imgseq, axis=0)
            imgseq = torch.from_numpy(imgseq).float()
            img_seg = torch.from_numpy(img_seg).long()

        return {'I': img_in,
                'S': img_seg,
                'I_seq': imgseq,
                'S_seq': segseq}


class RandomHorizontalFlip(object):
    """
    Fossil train data
    """
    def __call__(self, sample):
        img_in = sample['I']
        img_seg = sample['S']
        imgseq = sample['I_seq']
        segseq = sample['S_seq']
        if random.random() < 0.5:
            img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
            img_seg = img_seg.transpose(Image.FLIP_LEFT_RIGHT)
            imgseq = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgseq]
            segseq = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in segseq]

        return {'I': img_in,
                'S': img_seg,
                'I_seq': imgseq,
                'S_seq': segseq}


class RandomHorizontalFlip_bk(object):
    """
    MPI train data
    """
    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        if random.random() < 0.5:
            img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
            img_bg = img_bg.transpose(Image.FLIP_LEFT_RIGHT)
            img_rf = img_rf.transpose(Image.FLIP_LEFT_RIGHT)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        rotate_degree = random.uniform(-1*self.degree, self.degree)

        img_in = img_in.rotate(rotate_degree, Image.BILINEAR)
        img_bg = img_bg.rotate(rotate_degree, Image.BILINEAR)
        img_rf = img_rf.rotate(rotate_degree, Image.BILINEAR)

        return {'input': img_in,
                'background': img_bg,
                'reflection': img_rf}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img_in = ImageOps.expand(img_in, border=(0, 0, padw, padh), fill=0)
            img_bg = ImageOps.expand(img_bg, border=(0, 0, padw, padh), fill=0)
            img_rf = ImageOps.expand(img_rf, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img_in.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomScaleCrop_refine(object):
    """ for MPI:
        1. crop without black edges: scale factor [0.8,1.2]
            shorter edge is 436, while crop window is 336. 336/436=0.7706
            [0.6, 1.5]for 256 pratch
        2. use mask images as crop reference

        for Fossil:
        1. img size (W,H) = (1570, 1536)
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.w = None
        self.h = None
        self.x1 = None
        self.y1 = None

    def __call__(self, sample):
        img_in = sample['I']
        img_seg = sample['S']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.68),
                                    int(self.base_size * 1.75))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # check if the patch is valid

        x1 = random.randint(0, ow - self.crop_size)
        y1 = random.randint(0, oh - self.crop_size)

        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_seg = img_seg.resize((ow, oh), Image.NEAREST)

        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_seg = img_seg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'S': img_seg}


class RandomScaleCrop_refine_v2(object):
    """ for MPI:
        1. crop without black edges: scale factor [0.8,1.2]
            shorter edge is 436, while crop window is 336. 336/436=0.7706
            [0.6, 1.5]for 256 pratch
        2. use mask images as crop reference

        for Fossil:
        1. img size (W,H) = (1570, 1536)
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.w = None
        self.h = None
        self.x1 = None
        self.y1 = None

    def __call__(self, sample):
        img_in = sample['I']
        img_seg = sample['S']
        imgseq = sample['I_seq']
        segseq = sample['S_seq']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.68),
                                    int(self.base_size * 1.75))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_seg = img_seg.resize((ow, oh), Image.NEAREST)
        imgseq = [img.resize((ow, oh), Image.BILINEAR) for img in imgseq]
        segseq = [img.resize((ow, oh), Image.NEAREST) for img in segseq]

        # check if the patch is valid
        resample_flag = True
        loop_n = 0
        mask_num = 0
        while resample_flag:
            if loop_n >= 10 and loop_n % 10 == 0:
                # print('-----resample valid patch...loop times:', loop_n)
                if loop_n >= 100:
                    print('\t---final valid proportion: ', mask_num / (ow*oh))
                    break
            else:
                pass
            loop_n += 1
            img_mask = segseq
            self.x1 = random.randint(0, ow - self.crop_size)
            self.y1 = random.randint(0, oh - self.crop_size)
            # pdb.set_trace()
            patch_mask = [np.array(mask.crop((self.x1, self.y1, self.x1 + self.crop_size, self.y1 + self.crop_size))) for mask in img_mask]
            assert np.max(np.array(patch_mask)) <= 1
            mask_idx = np.array(patch_mask, dtype=np.float32) == 1
            mask_num = mask_idx.sum().astype(np.float32)
            resample_flag = mask_num == 0  # mask_num < 0.00005 * mask_idx.size

        x1 = self.x1
        y1 = self.y1

        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_seg = img_seg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        imgseq = [img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) for img in imgseq]
        segseq = [img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size)) for img in segseq]

        return {'I': img_in,
                'S': img_seg,
                'I_seq': imgseq,
                'S_seq': segseq}


class RandomScaleCrop_refine_bk(object):
    """ for MPI:
        1. crop without black edges: scale factor [0.8,1.2]
            shorter edge is 436, while crop window is 336. 336/436=0.7706
            [0.6, 1.5]for 256 pratch
        2. use mask images as crop reference
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.w = None
        self.h = None
        self.x1 = None
        self.y1 = None

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        img_mask = sample['M']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.78),
                                    int(self.base_size * 1.3))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # check if the patch is valid
        resample_flag = True
        loop_n = 0
        while resample_flag:
            if loop_n >= 10 and loop_n % 10 == 0:
                print('resample valid patch...loop times:', loop_n)
            else:
                pass
            loop_n += 1
            img_mask = img_mask.resize((ow, oh), Image.NEAREST)
            self.w, self.h = img_mask.size
            self.x1 = random.randint(0, w - self.crop_size)
            self.y1 = random.randint(0, h - self.crop_size)
            patch_mask = img_mask.crop((self.x1, self.y1,
                                       self.x1 + self.crop_size, self.y1 + self.crop_size))
            mask_idx = np.array(patch_mask, dtype=np.float32) == 0
            mask_num = mask_idx.sum().astype(np.float32)
            resample_flag = mask_num > 0.05 * mask_idx.size
        # pdb.set_trace()

        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        # pad crop
        # random crop crop_size
        #w, h = img_in.size
        #x1 = random.randint(0, w - self.crop_size)
        #y1 = random.randint(0, h - self.crop_size)
        x1 = self.x1
        y1 = self.y1
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class RandomScaleCrop_refine_RD(object):
    """ for MPI:
        1. crop without black edges: scale factor [0.8,1.2]
            shorter edge is 436, while crop window is 336. 336/436=0.7706
            [0.6, 1.5]for 256 pratch
        2. use mask images as crop reference
        for MPI-RD: do not use masks
    """
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.w = None
        self.h = None
        self.x1 = None
        self.y1 = None

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        img_mask = sample['M']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.78),
                                    int(self.base_size * 1.3))
        w, h = img_in.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # check if the patch is valid
        resample_flag = False
        loop_n = 0
        while resample_flag:
            if loop_n >= 10 and loop_n % 10 == 0:
                print('resample valid patch...loop times:', loop_n)
            else:
                pass
            loop_n += 1
            img_mask = img_mask.resize((ow, oh), Image.NEAREST)
            self.w, self.h = img_mask.size
            self.x1 = random.randint(0, w - self.crop_size)
            self.y1 = random.randint(0, h - self.crop_size)
            patch_mask = img_mask.crop((self.x1, self.y1,
                                       self.x1 + self.crop_size, self.y1 + self.crop_size))
            mask_idx = np.array(patch_mask, dtype=np.float32) == 0
            mask_num = mask_idx.sum().astype(np.float32)
            resample_flag =  mask_num > 0.05 * mask_idx.size
        # pdb.set_trace()

        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        # pad crop
        # random crop crop_size
        w, h = img_in.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        w, h = img_in.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img_in.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img_in = img_in.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_bg = img_bg.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        img_rf = img_rf.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        img_in = img_in.resize(self.size, Image.BILINEAR)
        img_bg = img_bg.resize(self.size, Image.BILINEAR)
        img_rf = img_rf.resize(self.size, Image.BILINEAR)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class ScaleWidth(object):
    def __init__(self, size):
        self.target_width = size

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        oh, ow = img_in.size
        if ow == self.target_width:
            return {'I': img_in,
                    'B': img_bg,
                    'R': img_rf}

        w = self.target_width
        h = int(self.target_width * oh / ow)

        img_in = img_in.resize((w, h), Image.BICUBIC)
        img_bg = img_bg.resize((w, h), Image.BICUBIC)
        img_rf = img_rf.resize((w, h), Image.BICUBIC)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedRescale(object):
    """
    Fossil, test data
    """
    def __init__(self, scale):
        self.scale = scale  # scale: float

    def __call__(self, sample):
        img_in = sample['I']
        img_seg = sample['S']
        imgseq = sample['I_seq']
        segseq = sample['S_seq']
        w, h = img_in.size
        ow = int(w * self.scale)
        oh = int(h * self.scale)
        if 224 < oh < 256:
            oh = 256
        else:
            #oh = 224
            pass
        self.size = (ow, oh)
        img_in = img_in.resize(self.size, Image.BILINEAR)
        img_seg = img_seg.resize(self.size, Image.NEAREST)
        imgseq = [img.resize(self.size, Image.BILINEAR) for img in imgseq]
        segseq = [img.resize(self.size, Image.NEAREST) for img in segseq]

        return {'I': img_in,
                'S': img_seg,
                'I_seq': imgseq,
                'S_seq': segseq}


class FixedRescale_bk(object):
    """
    MPI RD test data
    """
    def __init__(self, scale):
        self.scale = scale  # scale: float

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']
        w, h = img_in.size
        ow = int(w * self.scale)
        oh = int(h * self.scale)
        if 224 < oh < 256:
            oh = 256
        else:
            #oh = 224
            pass
        self.size = (ow, oh)
        img_in = img_in.resize(self.size, Image.BILINEAR)
        img_bg = img_bg.resize(self.size, Image.BILINEAR)
        img_rf = img_rf.resize(self.size, Image.BILINEAR)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}


class FixedScalePadding(object):
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, sample):
        img_in = sample['I']
        img_bg = sample['B']
        img_rf = sample['R']

        w, h = img_in.size
        if h < w:
            ow = self.size
            oh = int(1.0 * h * ow / w)
        else:
            oh = self.size
            ow = int(1.0 * w * oh / h)
        img_in = img_in.resize((ow, oh), Image.BILINEAR)
        img_bg = img_bg.resize((ow, oh), Image.BILINEAR)
        img_rf = img_rf.resize((ow, oh), Image.BILINEAR)

        pad_h = self.size - oh if oh < self.size else 0
        pad_w = self.size - ow if ow < self.size else 0

        img_in = ImageOps.expand(img_in, border=(0, 0, pad_w, pad_h), fill=0)
        img_bg = ImageOps.expand(img_bg, border=(0, 0, pad_w, pad_h), fill=0)
        img_rf = ImageOps.expand(img_rf, border=(0, 0, pad_w, pad_h), fill=0)

        return {'I': img_in,
                'B': img_bg,
                'R': img_rf}
