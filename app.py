import os
import sys
import numpy as np
import cv2
from skimage import measure, color
import matplotlib.pyplot as plt

from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QAction, QFileDialog, QDialog
from PySide2 import QtGui

from logicLayer import LogicLayerInterface
from utils import Metrics


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


class QSelectDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("inside or outside")
        self.resize(300, 100)

        self.dialogLayout = QVBoxLayout()
        self.lb = QLabel('inside or outside?')
        self.dialogLayout.addWidget(self.lb)
        self.inbnt = QPushButton('inside')
        self.inbnt.clicked.connect(self.accept)
        self.dialogLayout.addWidget(self.inbnt)
        self.outbnt = QPushButton('outside')
        self.outbnt.clicked.connect(self.reject)
        self.dialogLayout.addWidget(self.outbnt)

        self.setLayout(self.dialogLayout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.totSlice = 0
        self.curSlice = 0
        self.fossil = None

        self.setWindowTitle("Seg3D")
        self.resize(1500, 1500)
        self.setUi()

    def setUi(self):
        # Set Menu
        menuBar = self.menuBar()

        menuFile = menuBar.addMenu("File")
        openAction = QAction('&Open Image', self, triggered=self.openImage, shortcut='Ctrl+G')
        menuFile.addAction(openAction)
        quitAction = QAction('&Quit', self, triggered=self.quit, shortcut='Ctrl+Q')
        menuFile.addAction(quitAction)

        # Set layout
        mainLayout = QHBoxLayout()

        toolLayout = QHBoxLayout()
        self.lb0 = QLabel('IoU')
        toolLayout.addWidget(self.lb0)
        self.IoU = QLineEdit()
        self.IoU.setEnabled(False)
        toolLayout.addWidget(self.IoU)
        toolwg = QWidget()
        toolwg.setLayout(toolLayout)

        canvasLayout = QGridLayout()
        self.lb1 = QLabel('original image')
        self.ori_image = QLabel()
        self.lb2 = QLabel('mask image')
        self.mask_image = QLabel()
        self.lb3 = QLabel('front view')
        self.front_view = QLabel()
        self.lb4 = QLabel('left_view')
        self.left_view = QLabel()
        self.lb5 = QLabel('top_view')
        self.top_view = QLabel()
        canvasLayout.addWidget(self.lb1, 1, 1)
        canvasLayout.addWidget(self.ori_image, 2, 1)
        canvasLayout.addWidget(self.lb2, 1, 2)
        canvasLayout.addWidget(self.mask_image, 2, 2)
        canvasLayout.addWidget(self.lb3, 1, 3)
        canvasLayout.addWidget(self.front_view, 2, 3)
        canvasLayout.addWidget(self.lb4, 1, 4)
        canvasLayout.addWidget(self.left_view, 2, 4)
        canvasLayout.addWidget(self.lb5, 1, 5)
        canvasLayout.addWidget(self.top_view, 2, 5)
        canvaswg = QWidget()
        canvaswg.setLayout(canvasLayout)

        mainLayout.addWidget(toolwg)
        mainLayout.addWidget(canvaswg)
        mainwg = QWidget(self)
        mainwg.setLayout(mainLayout)
        self.setCentralWidget(mainwg)

    def openImage(self):
        filename = QFileDialog.getExistingDirectory(self, 'Open Image', 'C:\\')
        self.img_list = get_paths_from_images(filename)

        self.totSlice = len(self.img_list) / 2
        self.curSlice = 0

        # segmentation
        self.stone = LogicLayerInterface.getStone(filename)
        self.gt_stone = self.stone.getGTStone(filename)

        self.display()

    def quit(self):
        sys.exit(0)

    def wheelEvent(self, event:QtGui.QWheelEvent):
        if self.mask_image.underMouse():
            angle = event.angleDelta()
            if angle.y() > 0 and self.curSlice + 1 > 1:
                self.curSlice = self.curSlice - 1
            elif angle.y() < 0 and self.curSlice + 1 < self.totSlice:
                self.curSlice = self.curSlice + 1

            self.display()

    def mouseDoubleClickEvent(self, event:QtGui.QMouseEvent):
        if self.mask_image.underMouse():
            pos = self.mask_image.mapFromGlobal(event.globalPos())
            labels = measure.label(self.stone.morph_stone[self.curSlice], connectivity=2)

            pos_x = 193
            pos_y = 332

            maske_stone = np.nonzero(self.stone.morph_stone[self.curSlice])
            dist = np.sqrt(np.sum((np.array(maske_stone)- np.array([[pos_y],[pos_x]]))**2, axis=0)).tolist()
            near_p = np.array(maske_stone)[:, dist.index(np.min(dist))]
            num_morph = labels[near_p[0],near_p[1]]
            neighbor = np.array(np.where(labels == num_morph))
            min_y = np.min(neighbor[0, :]) -10
            max_y = np.max(neighbor[0, :]) +10
            min_x = np.min(neighbor[1, :]) -10
            max_x = np.max(neighbor[1, :]) +10
            # print(min_x, max_x, min_y, max_y)
            # print(pos.x(), pos.y())
            crop = self.stone.morph_stone[self.curSlice, min_y:max_y, min_x:max_x]
            crop_morph = self.stone.stone[self.curSlice, min_y:max_y, min_x:max_x]

            front_view = crop.astype(np.uint8) * 255
            front_view = QtGui.QImage(front_view, front_view.shape[1],
                                      front_view.shape[0], front_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.front_view.setPixmap(QtGui.QPixmap(front_view))

            scores = Metrics.computeMetrics( crop,self.gt_stone[self.curSlice, min_y:max_y, min_x:max_x])
            IoU = scores['iou']
            print('old IoU:', IoU)

            self.dialog = QSelectDialog()
            if self.dialog.exec_() == QDialog.Accepted:
                crop_resg = self.stone.reseg_in(crop_morph, pos_x - min_x, pos_y-min_y)
            else:
                crop_resg = self.stone.reseg_out(crop_morph, pos_x - min_x, pos_y-min_y)
            self.dialog.destroy()

            left_view = crop_resg.astype(np.uint8) * 255
            left_view = QtGui.QImage(left_view, left_view.shape[1],
                                     left_view.shape[0], left_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.left_view.setPixmap(QtGui.QPixmap(left_view))

            self.stone.morph_stone[self.curSlice, min_y:max_y, min_x:max_x] = crop_resg

            self.display()

            scores = Metrics.computeMetrics(crop_resg, self.gt_stone[self.curSlice, min_y:max_y, min_x:max_x])
            IoU = scores['iou']
            print('new IoU:', IoU)

    def display(self):
        stoneImage = self.stone.stone[self.curSlice]
        if np.max(stoneImage) <= 1:
            stoneImage = stoneImage * 255
        stoneImage = stoneImage.astype(np.uint8)
        img_stone = QtGui.QImage(stoneImage, stoneImage.shape[0],
                                 stoneImage.shape[1], QtGui.QImage.Format_Grayscale8)
        self.ori_image.setPixmap(QtGui.QPixmap(img_stone))

        maskImage = self.stone.morph_stone[self.curSlice]
        nonzero = np.nonzero(maskImage)
        segImage = cv2.cvtColor(stoneImage, cv2.COLOR_GRAY2RGB)
        segImage[nonzero] = [255, 0, 0]
        img_seg = QtGui.QImage(segImage, segImage.shape[0],
                               segImage.shape[1], segImage.shape[0] * 3, QtGui.QImage.Format_RGB888)
        self.mask_image.setPixmap(QtGui.QPixmap(img_seg))

        # if np.max(maskImage) <= 1:
        #     maskImage = maskImage * 255
        # maskImage = maskImage.astype(np.uint8)
        # cv2.imwrite('D:/workspace/Seg3D/results/'+ str(self.img_list[self.curSlice]), maskImage)


if __name__ == '__main__':
    app = QApplication()
    window = MainWindow()
    window.show()
    app.exec_()