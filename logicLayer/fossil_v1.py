from skimage import img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
from skimage.segmentation import clear_border
from utils import Utils3D
from utils import Metrics
import numpy as np

class Fossil:
    def __init__(self, binary_voxel, gray_voxel, labeled_id,
                 min_slice, max_slice, min_h, max_h, min_w, max_w):
        self.binary_voxel = binary_voxel
        self.gray_voxel = gray_voxel
        self.labeled_id = labeled_id
        self.min_slice = min_slice
        self.max_slice = max_slice
        self.min_h = min_h
        self.max_h = max_h
        self.min_w = min_w
        self.max_w = max_w

        self.three_view = None
        self.getThreeView()

        self.reseged_binary_voxel = None
        self.reseged_three_view = None

        self.gt_fossil = None
        self.metrics = None

    def getThreeView(self):
        # [0] left , [1] upper , [2] front
        self.three_view = Utils3D.getThreeViews(self.binary_voxel)

    def reseg(self,iteration):
        self.reseged_binary_voxel = np.zeros_like(self.gray_voxel)
        for i in range(self.gray_voxel.shape[0]):
            self.reseged_binary_voxel[i] = self._MorphACWE(self.gray_voxel[i],iteration)
        return self.reseged_binary_voxel

    def getNewThreeView(self,iteration):
        self.reseg(iteration)
        self.reseged_three_view = Utils3D.getThreeViews(self.reseged_binary_voxel)
        return self.reseged_three_view,  self.reseged_binary_voxel

    def getGT(self,gt_stone,pad):
        gt = Utils3D.getPadFossilFromStone(gt_stone, pad, self.min_slice, self.max_slice, self.min_h, self.max_h, self.min_w, self.max_w)
        if np.max(gt)>=1:
            gt = gt/np.max(gt)
        self.gt_fossil = gt
        return gt

    def getMetrics(self, initial = 0):
        if initial == 0:
            res = Metrics.computeMetrics(self.reseged_binary_voxel,self.gt_fossil)
        else:
            res = Metrics.computeMetrics(self.binary_voxel, self.gt_fossil)
        self.metrics = res
        return res

    def _MorphACWE(self, img,iteration):
        image = img_as_float(img)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self._store_evolution_in(evolution)
        ls = morphological_chan_vese(image, iteration, init_level_set=init_ls, smoothing=3,
                                     iter_callback=callback)
        # if np.sum(ls) > (ls.shape[0] * ls.shape[1] / 2):
        #     ls = 1 - ls
        ls = clear_border(ls)
        return ls

    def _store_evolution_in(self,lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store

if __name__ == '__main__':
    import numpy as np
    import cv2
    binary_voxel = np.zeros((100,100,100),dtype=np.uint8)
    binary_voxel[30:70,30:70,30:70] = 1
    fossil = Fossil(binary_voxel, binary_voxel, 1,1,1,1,1,1,1)
    cv2.imwrite('Front.jpg',fossil.three_view[0]*255)
