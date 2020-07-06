from skimage import img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  checkerboard_level_set)
from skimage.segmentation import clear_border
import numpy as np

class Stone:
    def __init__(self, stone, slice_list):
        self.stone = stone
        self.slice_list = slice_list
        self.segmentStone()

    def segmentStone(self):
        res = np.zeros_like(self.stone)
        for i in range(self.stone.shape[0]):
            res[i] = self._MorphACWE(self.stone[i])
        self.morph_stone = res
        return res

    def _MorphACWE(self, img):
        image = img_as_float(img)

        # Initial level set
        init_ls = checkerboard_level_set(image.shape, 6)
        # List with intermediate results for plotting the evolution
        evolution = []
        callback = self._store_evolution_in(evolution)
        ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3,
                                     iter_callback=callback)
        ls = clear_border(ls)
        return ls

    def _store_evolution_in(self,lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store





