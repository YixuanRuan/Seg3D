from utils import Utils3D

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

    def getThreeView(self):
        # [0] left , [1] upper , [2] front
        self.three_view = Utils3D.getThreeViews(self.binary_voxel)

if __name__ == '__main__':
    import numpy as np
    import cv2
    binary_voxel = np.zeros((100,100,100),dtype=np.uint8)
    binary_voxel[30:70,30:70,30:70] = 1
    fossil = Fossil(binary_voxel, binary_voxel, 1,1,1,1,1,1,1)
    cv2.imwrite('Front.jpg',fossil.three_view[0]*255)
