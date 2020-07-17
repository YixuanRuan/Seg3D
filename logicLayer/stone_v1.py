from skimage import img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
from skimage.segmentation import clear_border
import numpy as np
from .fossil_v1 import Fossil
from utils import Utils3D

class Stone:
    def __init__(self, stone, slice_list):
        self.stone = stone
        self.slice_list = slice_list
        self.morph_stone = None
        self.segmentStone()
        self.labeled_fossil_features = None
        self.labeled_morph_stone = None
        self.labelStone()

    def segmentStone(self):
        res = np.zeros_like(self.stone)
        for i in range(self.stone.shape[0]):
            res[i] = self._MorphACWE(self.stone[i])
        self.morph_stone = res
        return res

    def getThreeViewByIndexAndHWPosition(self, slice, h, w):
        label = self.labeled_morph_stone[slice,h,w]

        min_slice = self.labeled_morph_fossil_features[label, 4]
        max_slice = self.labeled_morph_fossil_features[label, 5]
        min_h = self.labeled_morph_fossil_features[label, 6]
        max_h = self.labeled_morph_fossil_features[label, 7]
        min_w = self.labeled_morph_fossil_features[label, 8]
        max_w = self.labeled_morph_fossil_features[label, 9]

        binary_voxel = Utils3D.getPadFossilFromStone(self.morph_stone, 1, min_slice, max_slice, min_h, max_h, min_w, max_w )
        gray_voxel = Utils3D.getPadFossilFromStone(self.stone, 100, min_slice, max_slice, min_h, max_h, min_w, max_w )

        fossil = Fossil(binary_voxel, gray_voxel, label,
                 min_slice, max_slice, min_h, max_h, min_w, max_w)

        return fossil

    def labelStone(self):
        features , vx, names, pts = Utils3D.gseperateMaskVoxelsGetFeaturesAndNamesAndPts(
            self.stone, self.slice_list, fixed_num=None,
            needPts=False, hint=True, needNamesOrPts=False
        )
        self.labeled_morph_fossil_features = features
        self.labeled_morph_stone = vx
        return features, vx

    def _MorphACWE(self, img):
        image = img_as_float(img)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self._store_evolution_in(evolution)
        ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                     iter_callback=callback)
        if np.sum(ls) > (ls.shape[0] * ls.shape[1] / 2):
            ls = 1 - ls
        ls = clear_border(ls)
        return ls

    def _store_evolution_in(self,lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store





