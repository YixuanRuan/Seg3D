from logicLayer import LogicLayerInterface
from utils import Utils2D
import numpy as np
import cv2
if __name__ == '__main__':

    stone = LogicLayerInterface.getStone('./samples/')
    gt_stone = stone.getGTStone('./samples/')
    fossil = stone.getThreeViewByIndexAndHWPosition(0,100,200)
    fossil.getGT(gt_stone, pad=1)
    metrics = fossil.getMetrics(1)
    print(metrics)
    cv2.imwrite('1.png',stone.morph_stone[0].astype(np.uint8) * 255)

    # fossil = stone.getThreeViewByIndexAndHWPosition(0, 480, 725)
    # Utils2D.showOnePic(fossil.three_view[2].astype(np.uint8) * 255, "Hello")
    # cv2.imwrite('1.png', stone.morph_stone[0].astype(np.uint8) * 255)
    # a = cv2.imread("1.png",0)
    # Utils2D.showOnePic(a[465:495,710:740], "Hello")

