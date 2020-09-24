import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import os, os.path
from networkLayer.utils import combine_transforms_v3 as ctr
from PIL import Image
from torchvision import transforms
import glob
import tqdm

import pdb


###############################################################################
# train/test split files generation
###############################################################################
def rename_imgs(data_root):
    file_dict = glob.glob(os.path.join(data_root, 'masks', '*.jpg'))
    # pdb.set_trace()
    for file in file_dict:
        fn = file.split('/')[-1].split('.')[0].split('-')[1]
        fn_new = '{:05d}'.format(int(fn)) + '.' + file.split('/')[-1].split('.')[1]
        os.rename(file, os.path.join(data_root, 'masks', fn_new+'_mask.jpg'))
        # os.rename(os.path.join(data_root, 'seg', fn+'_mask.jpg'), os.path.join(data_root, 'seg', fn_new+'_mask.jpg'))


def rename_imgs_frac(data_root):
    """for 16B imgs_1000, .0-->.00"""
    file_dict = glob.glob(os.path.join(data_root, 'masks', '*.jpg'))
    # pdb.set_trace()
    for file in file_dict:
        fn = file.split('/')[-1].split('_')[0].split('.')[0]
        decimal = file.split('/')[-1].split('_')[0].split('.')[1]
        if len(decimal) < 2:
            decimal = decimal + '0'
        fn_new = '{:05d}'.format(int(fn)) + '.' + decimal
        # os.rename(file, os.path.join(data_root, 'imgs_1000', fn_new+'.jpg'))
        fn_mask = file.split('/')[-1]
        os.rename(os.path.join(data_root, 'masks', fn_mask), os.path.join(data_root, 'masks', fn_new+'_mask.jpg'))


def get_fn_list(data_root, is_train_set=True):
    file_dict = sorted(glob.glob(os.path.join(data_root, 'masks', '*.jpg')))
    fn_list = []
    for file in file_dict:
        fn = file.split('/')[-1].split('_')[0]
        fn_list.append(fn)
    fnum = len(fn_list)
    tbar = tqdm.tqdm(range(fnum))
    tbar.set_description('gen_fn_list')
    invalid_img = []
    fn_list_mi = []  # multi-input
    for i in tbar:
        f_base = [fn_list[i]]
        f_former = [fn_list[max(i-3, 0)], fn_list[max(i-2, 0)], fn_list[max(i-1, 0)]]
        f_after = [fn_list[min(i+1, fnum-1)], fn_list[min(i+2, fnum-1)], fn_list[min(i+3, fnum-1)]]
        # pdb.set_trace()
        f_seq = f_former + f_base + f_after
        """ check if valid """
        masks = []
        for file in f_seq:
            seg_mask = np.array(cv2.imread(os.path.join(data_root, 'masks', file+'_mask.jpg'), 0) // 255)
            masks.append(seg_mask)
        if np.sum(np.asarray(masks)) == 0:
            invalid_img.append(*f_base)
            if is_train_set:
                continue
        fs = '{} {} {} {} {} {} {}'.format(*f_seq[:])
        fn_list_mi.append(fs)
    print('invalid_imgs are ', invalid_img)
    return fn_list_mi


def get_fn_list_valid(data_root):
    """
    remove images without label
    data_root is seg folder
    """
    file_dict = glob.glob(os.path.join(data_root, 'seg', '*.jpg'))
    fn_list = []
    for file in file_dict:
        if np.sum(np.array(cv2.imread(file, 0) // 255)) == 0:
            continue
        fn = file.split('/')[-1].split('_')[0]
        fn_list.append(fn)
    return fn_list


def gen_split_files(fn_list, save_root, fossil_name, test_pile=None):
    """
    :param fn_list: data file list
    :param save_root: split file save root
    :param test_pile: test file number
    :return: None
    """
    train_fn = 'train_v4_' + fossil_name + '.txt'
    test_fn = 'test_v4_' + fossil_name + '.txt'
    idx = np.arange(len(fn_list), dtype=np.int32)
    np.random.shuffle(idx)
    if test_pile == None:
        test_pile = int(len(fn_list) * 0.5)  #  0.5 (2020.02) --> 0.6 (2020.04)

    with open(os.path.join(save_root, train_fn), 'w+') as f:
        for i in range(len(idx) - test_pile):
            st = fn_list[idx[i]]
            f.write(st+'\n')

    with open(os.path.join(save_root, test_fn), 'w+') as f:
        for i in range(len(idx) - test_pile, len(idx)):
            st = fn_list[idx[i]]
            f.write(st+'\n')


def gen_test_file_from_folder(fn_list, save_root):
    """
    :param fn_list: data file list
    :param save_root: split file save root
    :param test_pile: test file number
    :return: None
    """
    test_fn = 'test_sequence.txt'

    with open(os.path.join(save_root, test_fn), 'w+') as f:
        for i in range(len(fn_list)):
            st = fn_list[i]
            f.write(st+'\n')


###############################################################################
# Dataset sets
###############################################################################
class DatasetIdFossil(data.Dataset):
    def __init__(self, data_opt, is_train=None, cropped=None, test_full_data=False):
        self.opt = data_opt
        self.root = data_opt.data_root
        self.is_train = data_opt.is_train if is_train is None else is_train
        self.fossil_name = data_opt.fossil_name

        self.I_paths = []
        self.S_paths = []
        self.I_sequences = []
        self.S_sequences = []

        for fossil in self.fossil_name:
            root_i = os.path.join(self.root, fossil, 'imgs_1000')
            root_s = os.path.join(self.root, fossil, 'masks')

            if self.is_train:
                fname = 'train_v4_' + fossil + '.txt'
            else:
                if not test_full_data:
                    fname = 'test_v4_' + fossil + '.txt'
                else:
                    fname = 'test_v3_' + fossil + '.txt'

            with open(os.path.join(self.root, fname), 'r') as fid:
                lines = fid.readlines()
                for line in lines:
                    line = line.strip()  # sequence
                    fs = line.split(' ')
                    fn = fs[3]
                    self.I_paths.append(os.path.join(root_i, fn+'.jpg'))
                    self.S_paths.append(os.path.join(root_s, fn+'_mask.jpg'))
                    self.I_sequences.append([os.path.join(root_i, f + '.jpg') for f in fs])
                    self.S_sequences.append([os.path.join(root_s, f + '_mask.jpg') for f in fs])
                    # print(self.I_sequences)

        if self.is_train:
            self.transform = get_combine_transform(name=data_opt.preprocess, load_size=data_opt.load_size,
                                                   new_size=data_opt.new_size, is_train=self.is_train,
                                                   no_flip=data_opt.no_flip,
                                                   image_mean=data_opt.image_mean, image_std=data_opt.image_std,
                                                   use_norm=data_opt.use_norm)
        else:
            self.transform = fixed_scale_transform(new_size=1.0, image_mean=data_opt.image_mean,
                                                   image_std=data_opt.image_std, use_norm=data_opt.use_norm)

    def __len__(self):
        return len(self.I_paths)

    def _transform_image(self, image, seed=None):
        if self.transform is not None:
            seed = np.random.randint(100, 500000) if seed is None else seed
            torch.manual_seed(seed)
            random.seed = seed
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        seed = np.random.randint(100, 500000)
        if not self.opt.serial_batches:
            if self.opt.unpaired:
                # unpaired
                random.shuffle(self.I_paths)
                random.shuffle(self.S_paths)
            else:
                # paired
                idx = np.arange(len(self.I_paths))
                random.shuffle(idx)
                # pdb.set_trace()
                self.I_paths = np.array(self.I_paths)[idx]
                self.S_paths = np.array(self.S_paths)[idx]
                self.I_sequences = np.array(self.I_sequences)[idx]
                self.S_sequences = np.array(self.S_sequences)[idx]
            seed = None

        ret_dict_tmp = {
            'I': Image.open(self.I_paths[index]).convert('RGB'),
            'S': Image.fromarray(cv2.imread(self.S_paths[index], 0) // 255),
            'I_seq': [Image.open(fpath).convert('RGB') for fpath in self.I_sequences[index]],
            'S_seq': [Image.fromarray(cv2.imread(fpath, 0) // 255) for fpath in self.S_sequences[index]],
            # 'S': Image.open(self.S_paths[index]).convert('1')
        }

        ret_dict = self.transform(ret_dict_tmp)

        ret_dict['name'] = os.path.join(self.I_paths[index])

        # print([torch.max(ret_dict['S']), torch.min(ret_dict['S'])])

        return ret_dict


###############################################################################
# Fast functions
###############################################################################

def get_combine_transform(name, load_size=300, new_size=256, is_train=True, no_flip=False,
                          image_mean=(0., 0., 0.), image_std=(1.0, 1.0, 1.0), use_norm=True):
    transform_list = []
    if name == 'resize_and_crop_refine':
        transform_list.append(ctr.RandomScaleCrop_refine(load_size, new_size))
    elif name == 'resize_and_crop_refine_v2':
        transform_list.append(ctr.RandomScaleCrop_refine_v2(load_size, new_size))
    elif name == 'resize_and_crop':
        # o_size = [load_size, load_size]
        # transform_list.append(transforms.Scale(o_size, Image.BICUBIC))
        transform_list.append(ctr.RandomScaleCrop(load_size, new_size))
    elif name == 'crop':
        transform_list.append(ctr.RandomScaleCrop(load_size, new_size))
    elif name == 'scale_width':
        transform_list.append(ctr.ScaleWidth(new_size))
    elif name == 'scale_width_and_crop':
        transform_list.append(ctr.ScaleWidth(load_size))
        transform_list.append(transforms.RandomCrop(new_size))

    if is_train and not no_flip:
        transform_list.append(ctr.RandomHorizontalFlip())

    transform_list += [ctr.ToTensor()]

    # if use_norm:
    #     transform_list += [ctr.Normalize(image_mean, image_std)]
    return transforms.Compose(transform_list)


def fixed_scale_transform(new_size=0.45,
                          image_mean=(0., 0., 0.), image_std=(1., 1., 1.), use_norm=True):
    transform_list = []
    # transform_list.append(ctr.FixedScalePadding(size=new_size))
    transform_list.append(ctr.FixedRescale(scale=new_size))
    transform_list += [ctr.ToTensor()]
    # if use_norm:
    #     transform_list += [ctr.Normalize(image_mean, image_std)]

    return transforms.Compose(transform_list)


def test_id_dataset():
    from configs.fossil import opt
    from utils import tensor2img, show_image
    dataset = DatasetIdFossil(opt.data)
    for i in range(len(dataset)):
        data = dataset[i]
        img_i = data['I']
        img_s = torch.tensor(data['S'], dtype=torch.float32)
        img_i = tensor2img(img_i, mean=opt.data.image_mean, std=opt.data.image_std)
        img_s = tensor2img(img_s, mean=opt.data.image_mean, std=opt.data.image_std)
        if not show_image(cv2.hconcat([img_i[:, :, ::-1], img_s[:, :]])):
            break


if __name__ == '__main__':
    # # format img files
    fossil_n = ['XSC-03A', 'XSC-03B', 'XSC-16B']
    data_root = '../../Data'

    # # rename mask images
    # for i in range(2, len(fossil_n)):
    #     fossil_name = fossil_n[i]
    #     print('processing fossil: '+fossil_name)
    #     # rename_imgs(os.path.join(data_root, fossil_name))
    #     rename_imgs_frac(os.path.join(data_root, fossil_name))

    # # get file lists and save into test_v3_full.txt; test_v4
    for i in range(0, len(fossil_n)):
        fossil_name = fossil_n[i]
        print('processing fossil: '+fossil_name)
        fn_list = get_fn_list(os.path.join(data_root, fossil_name), is_train_set=True)
        # pdb.set_trace()
        gen_split_files(fn_list, data_root, fossil_name)
