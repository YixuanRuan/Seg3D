import os
import os.path as osp
import sys
import numpy as np
import cv2

from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QSplitter, QLabel, QLineEdit, QPushButton, QAction, QFileDialog
from PySide2 import QtGui
from PySide2 import QtCore

from logicLayer import LogicLayerInterface
from utils import Utils2D


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

        toolLayout = QVBoxLayout()
        form = QFormLayout()
        self.lb = QLabel('threshold')
        self.le = QLineEdit()
        self.le.setValidator(QtGui.QIntValidator())
        form.addRow(self.lb, self.le)
        toolLayout.addLayout(form)
        self.bnt = QPushButton('re-segment')
        self.bnt.clicked.connect(self.resegment)
        toolLayout.addWidget(self.bnt)
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

        self.totSlice = len(self.img_list)
        self.curSlice = 0

        # segmentation
        self.stone = LogicLayerInterface.getStone(filename)

        stoneImage = self.stone.stone[self.curSlice]
        print(stoneImage.shape)
        if np.max(stoneImage) <= 1:
            stoneImage = stoneImage * 255
        stoneImage = stoneImage.astype(np.uint8)
        img_stone = QtGui.QImage(stoneImage, stoneImage.shape[0],
                                 stoneImage.shape[1], QtGui.QImage.Format_Grayscale8)
        self.ori_image.setPixmap(QtGui.QPixmap(img_stone))

        maskImage = self.stone.morph_stone[self.curSlice]
        if np.max(maskImage) <= 1:
            maskImage = maskImage * 255
        maskImage = maskImage.astype(np.uint8)
        img_mask = QtGui.QImage(maskImage, maskImage.shape[0],
                                 maskImage.shape[1], QtGui.QImage.Format_Grayscale8)
        self.mask_image.setPixmap(QtGui.QPixmap(img_mask))

    def quit(self):
        sys.exit(0)

    def wheelEvent(self, event:QtGui.QWheelEvent):
        if self.mask_image.underMouse():
            angle = event.angleDelta()
            if angle.y() > 0 and self.curSlice + 1 > 1:
                self.curSlice = self.curSlice - 1
            elif angle.y() < 0 and self.curSlice + 1 < self.totSlice:
                self.curSlice = self.curSlice + 1

            stoneImage = self.stone.stone[self.curSlice]
            if np.max(stoneImage) <= 1:
                stoneImage = stoneImage * 255
            stoneImage = stoneImage.astype(np.uint8)
            img_stone = QtGui.QImage(stoneImage, stoneImage.shape[0],
                                     stoneImage.shape[1], QtGui.QImage.Format_Grayscale8)
            self.ori_image.setPixmap(QtGui.QPixmap(img_stone))

            maskImage = self.stone.morph_stone[self.curSlice]
            if np.max(maskImage) <= 1:
                maskImage = maskImage * 255
            maskImage = maskImage.astype(np.uint8)
            img_mask = QtGui.QImage(maskImage, maskImage.shape[0],
                                     maskImage.shape[1], QtGui.QImage.Format_Grayscale8)
            self.mask_image.setPixmap(QtGui.QPixmap(img_mask))


    def mouseDoubleClickEvent(self, event:QtGui.QMouseEvent):
        pos = self.mask_image.mapFromGlobal(event.globalPos())
        x = pos.x()
        y = pos.y()
        if self.mask_image.underMouse() and self.stone.morph_stone[self.curSlice][y, x] == 1:
            self.fossil = self.stone.getThreeViewByIndexAndHWPosition(self.curSlice, y, x)

            front_view = self.fossil.three_view[0].astype(np.uint8) * 255
            front_view = QtGui.QImage(front_view, front_view.shape[1],
                                     front_view.shape[0], front_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.front_view.setPixmap(QtGui.QPixmap(front_view))

            left_view = self.fossil.three_view[1].astype(np.uint8) * 255
            left_view = QtGui.QImage(left_view, left_view.shape[1],
                                      left_view.shape[0], left_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.left_view.setPixmap(QtGui.QPixmap(left_view))

            top_view = self.fossil.three_view[2].astype(np.uint8) * 255
            top_view = QtGui.QImage(top_view, top_view.shape[1],
                                      top_view.shape[0], top_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.top_view.setPixmap(QtGui.QPixmap(top_view))
        else:
            self.fossil = None

    def resegment(self):
        input = self.le.text()
        if len(input) != 0 and self.fossil != None:
            iter = int(input)
            self.reseged_three_view, self.reseged_binary_voxel = self.fossil.getNewThreeView(iter)

            front_view = self.reseged_three_view[0].astype(np.uint8) * 255
            front_view = QtGui.QImage(front_view, front_view.shape[1],
                                      front_view.shape[0], front_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.front_view.setPixmap(QtGui.QPixmap(front_view))

            left_view = self.reseged_three_view[1].astype(np.uint8) * 255
            left_view = QtGui.QImage(left_view, left_view.shape[1],
                                     left_view.shape[0], left_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.left_view.setPixmap(QtGui.QPixmap(left_view))

            top_view = self.reseged_three_view[2].astype(np.uint8) * 255
            top_view = QtGui.QImage(top_view, top_view.shape[1],
                                    top_view.shape[0], top_view.shape[1], QtGui.QImage.Format_Grayscale8)
            self.top_view.setPixmap(QtGui.QPixmap(top_view))


if __name__ == '__main__':
    app = QApplication()
    window = MainWindow()
    window.show()
    app.exec_()