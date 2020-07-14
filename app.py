import os
import os.path as osp
import sys
import numpy as np
import cv2

from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtWidgets import QWidget, QGridLayout, QLabel, QAction, QFileDialog
from PySide2 import QtGui
from PySide2 import QtCore


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

        self.setWindowTitle("Seg3D")
        self.resize(1000, 1000) # 设置窗口大小
        self.setUi()

    def setUi(self):
        #Set canvas
        mainWidget = QWidget()
        mainLayout = QGridLayout()
        mainWidget.setLayout(mainLayout)

        self.label1 = QLabel('image')
        self.label1.setFixedSize(400,20)
        self.origin_image_label = QLabel()
        self.origin_image_label.setFixedSize(400,400)
        self.origin_image_label.setScaledContents(True)

        label2 = QLabel('front view')
        label2.setFixedSize(400, 20)
        self.front_view_label = QLabel()
        self.front_view_label.setFixedSize(400, 400)
        self.front_view_label.setScaledContents(True)

        label3 = QLabel('left_view')
        label3.setFixedSize(400, 20)
        self.left_view_label = QLabel()
        self.left_view_label.setFixedSize(400, 400)
        self.left_view_label.setScaledContents(True)

        label4 = QLabel('top_view')
        label4.setFixedSize(400, 20)
        self.top_view_label = QLabel()
        self.top_view_label.setFixedSize(400, 400)
        self.top_view_label.setScaledContents(True)

        mainLayout.addWidget(self.label1, 1, 1)
        mainLayout.addWidget(self.origin_image_label, 2, 1)
        mainLayout.addWidget(label2, 1, 2)
        mainLayout.addWidget(self.front_view_label, 2, 2)
        mainLayout.addWidget(label3, 3, 1)
        mainLayout.addWidget(self.left_view_label, 4, 1)
        mainLayout.addWidget(label4, 3, 2)
        mainLayout.addWidget(self.top_view_label, 4, 2)

        self.setCentralWidget(mainWidget)

        # Set Menu
        menuBar = self.menuBar()

        menuFile = menuBar.addMenu("File")
        openAction = QAction('&Open Image', self, triggered=self.openImage, shortcut='Ctrl+G')
        menuFile.addAction(openAction)
        quitAction = QAction('&Quit', self, triggered=self.quit, shortcut='Ctrl+Q')
        menuFile.addAction(quitAction)

    def openImage(self):
        print('open image')
        filename = QFileDialog.getExistingDirectory(self, 'Open Image', 'C:\\')

        ori_img_list = get_paths_from_images(filename)
        ori_img = np.zeros(shape=(cv2.imread(ori_img_list[0]).shape[0], cv2.imread(ori_img_list[0]).shape[1],
                           len(ori_img_list)), dtype='uint8') # Numpy HWN

        for i, img_path in enumerate(ori_img_list):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            ori_img[:, :, i] = img

        """
        分割图像
        """

        self.seg_img_list = get_paths_from_images(filename[0:-3] + 'seg')
        # self.seg_img = np.zeros(shape=(ori_img.shape[0], ori_img.shape[1], 3, ori_img.shape[2]),
        #                    dtype='uint8')  # Numpy HW3*N
        #
        # for i, img_path in enumerate(seg_img_list):
        #     img = cv2.imread(img_path)
        #     self.seg_img[:, :, :, i] = img

        self.totSlice = len(self.seg_img_list)
        self.curSlice = 0
        self.origin_image_label.setPixmap(QtGui.QPixmap(self.seg_img_list[self.curSlice]))
        self.label1.setText('image                      '
                            + str(self.curSlice + 1) + '  of  ' + str(self.totSlice))


    def quit(self):
        print('quit')
        sys.exit(0)

    def wheelEvent(self, event:QtGui.QWheelEvent):
        if self.origin_image_label.underMouse():
            angle = event.angleDelta()
            if angle.y() > 0 and self.curSlice + 1 > 1:
                self.curSlice = self.curSlice - 1
                self.origin_image_label.setPixmap(QtGui.QPixmap(self.seg_img_list[self.curSlice]))
                self.label1.setText('image                      '
                                    + str(self.curSlice + 1) + '  of  ' + str(self.totSlice))
            elif angle.y() < 0 and self.curSlice + 1 < self.totSlice:
                self.curSlice = self.curSlice + 1
                self.origin_image_label.setPixmap(QtGui.QPixmap(self.seg_img_list[self.curSlice]))
                self.label1.setText('image                      '
                                    + str(self.curSlice + 1) + '  of  ' + str(self.totSlice))

    def mouseDoubleClickEvent(self, event:QtGui.QMouseEvent):
        if self.origin_image_label.underMouse():
            print(self.origin_image_label.mapFromGlobal(event.globalPos()))
            self.front_view_label.setPixmap(QtGui.QPixmap('F:\BaiduNetdiskDownload\\front.png'))
            self.left_view_label.setPixmap(QtGui.QPixmap('F:\BaiduNetdiskDownload\\left.png'))
            self.top_view_label.setPixmap(QtGui.QPixmap('F:\BaiduNetdiskDownload\\top.png'))



if __name__ == '__main__':
    app = QApplication()

    window = MainWindow()
    window.show()

    app.exec_()