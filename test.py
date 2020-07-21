from logicLayer import LogicLayerInterface
from utils import Utils2D
import numpy as np
import cv2
if __name__ == '__main__':

    stone = LogicLayerInterface.getStone('./samples/')
    cv2.imwrite('1.png',stone.morph_stone[0].astype(np.uint8) * 255)
    fossil = stone.getThreeViewByIndexAndHWPosition(0, 436, 663)
    Utils2D.showOnePic(fossil.three_view[2].astype(np.uint8) * 255, "Hello")
