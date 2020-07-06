import numpy as np
import re, os, cv2, shutil
from collections import Counter

class Utils2D:
    """ Utils2D
    
    Deal with the following type of 2d img
    
        1. img - original - gray
        2. img - original origin-mask-blended - rgb
        3. mask - ground truth - gray
        4. mask - predict names - gray
        5. mask - gt-predict-blended - rgb


    processing status diagram:  start from status 0, top to bottom

                            Status 0: Directory Check
                                        |
                 Status 1: Directory File Names/Indexes Get (and Sort)
                                        |
                                Status 2: File Read
                                        |
                 Status 3: File Transform (Binary, RGB, Resize etc.)
                                        |
                 Status 4: File Visualize (Bounding Box, UnderAndOverSegVisual)
                                        |
                            Status 5: File Saver

    ##############################
    Directory Check Functions
    ##############################
        check_dir
        remove_dir

    ##############################
    Path Getter Functions
    ##############################

        ### indexes getter ###

            getSortedIndexes(root, pattern, splitter, index)
                1. add file name which matches the pattern to a list
                2. split the file name and integer it
                3. sort the integer

        ### img name getter ###

            # get one img name by index
            get2DNameByIndex(root, index, tail, fixed=True, fixed_num=4)
                :cat root index tail together

            
    """

    '''
    Directory Checker
    '''

    @staticmethod
    def check_dir(s_dir, force_clean=False):
        if force_clean:
            Utils2D.remove_dir(s_dir)
        if not os.path.exists(s_dir):
            os.makedirs(s_dir)

    @staticmethod
    def remove_dir(s_dir):
        if os.path.exists(s_dir):
            shutil.rmtree(s_dir)

    '''
    Names/Indexes Getter
    '''

    @staticmethod
    def getSortedIndexes(root, pattern, splitter, index):
        imgs_uncleaned = os.listdir(root)
        imgs = []
        for img in imgs_uncleaned:
            if re.match(pattern, img):
                imgs.append(img)
        imgs = np.array(imgs)
        imgs_int = np.zeros((len(imgs),), dtype=np.uint32)
        for i in range(len(imgs)):
            imgs_int[i] = int(imgs[i].split(splitter)[index])
        return (np.sort(imgs_int)).tolist()

    @staticmethod
    def get2DNameByIndex(root, index, tail, fixed=True, fixed_num=4):
        if fixed:
            img_path = root + str(index).zfill(fixed_num) + tail
        else:
            img_path = root + str(index) + tail
        return img_path

    @staticmethod
    def get2DNames(root_path,indexes,tail,fixed,fixed_num):
        img_paths = []
        nums = len(indexes)
        for i in range(nums):
            img_paths.append(Utils2D.get2DNameByIndex(root_path, indexes[i], tail, fixed=fixed, fixed_num=fixed_num))
        return img_paths
    '''
    Img Reader
    '''

    @staticmethod
    def readBinary(img_path, resize=False, H=1000, W=1000):
        gray = Utils2D.readGray(img_path, resize, H, W)
        return Utils2D.binary(gray)

    @staticmethod
    def readGray(img_path, resize=False, H=1000, W=1000):
        gray = cv2.imread(img_path, 0)
        if resize:
            gray = cv2.resize(gray, (W, H))
        return gray

    @staticmethod
    def readRGB(img_path, resize=False, H=1000, W=1000):
        gbr = cv2.imread(img_path)
        if resize:
            gbr = cv2.resize(gbr, (W, H))
        rgb = cv2.cvtColor(gbr,cv2.COLOR_BGR2RGB)
        return rgb

    '''
    Img Transformer
    '''

    @staticmethod
    def binary(image, maxScale=False, hint=False, threshhold=0.5):
        assert type(image) is np.ndarray
        if hint:
            print('###############Binarizing Image###############')
            print('Original Image info as follows')
            Utils2D.checkPicProperties(image)
            print()

        if np.max(image) <= 1:
            if hint:
                print('Image value already ranges within 0-1')
                print()
        elif maxScale:
            image = image / np.max(image)
            if hint:
                print('Binarizing image with image max value')
                print('Binarized image info:')
                Utils2D.checkPicProperties(image)
                print()
        else:
            image = image / 255
            if hint:
                print('Binarizing image with 255')
                print('Binarized image info:')
                Utils2D.checkPicProperties(image)
                print()

        if image.dtype != np.uint8:
            if hint:
                print('Changing image type to np.uint8')
                print('all pixels above threshold %.1f set to 1' % (threshhold))
                print('all pixels below threshold %.1f set to 0' % (threshhold))
                print()
            image[image > threshhold] = 1
            image = image.astype(np.uint8)
        return image

    '''
    Img Visualizer
    '''

    @staticmethod
    def showOnePic(image,winName='Pic'):
        assert type(image) is np.ndarray
        if np.max(image) <= 1:
            image = image * 255
        image = image.astype(np.uint8)
        cv2.imshow(winName, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def drawBoxByBRGImg(img, h_min, h_max, w_min, w_max, bgr=[0, 215, 255], width=5):
        assert len(img.shape) == 3
        H = img.shape[0]
        W = img.shape[1]
        coe = width // 2
        h_min_coe = np.max([0,h_min-coe])
        w_min_coe = np.max([0,w_min-coe])
        h_max_coe = np.min([H,h_max+coe])
        w_max_coe = np.min([H,w_max+coe])
        img[h_min_coe:h_max_coe, w_min_coe:w_min, :] = bgr
        img[h_min_coe:h_max_coe, w_max:w_max_coe, :] = bgr
        img[h_min_coe:h_min, w_min_coe:w_max_coe, :] = bgr
        img[h_max:h_max_coe, w_min_coe:w_max_coe, :] = bgr
        return img

    '''
    Others
    '''

    @staticmethod
    def checkArrProperties(arr):
        assert type(arr) is np.ndarray
        print('Array datatype is: %s' % (str(arr.dtype)))
        print('Array max value is: %.3f' % (np.max(arr)))
        print('Array min value is: %.3f' % (np.min(arr)))
        cnt = Counter(arr.flatten())
        cnt = list(cnt.items())
        print('Array value distribution is: %s' % (str(cnt)))
