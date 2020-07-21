from .format_v1 import Format
from .utils2D_v1 import Utils2D
from .metrics_v1 import Metrics
import numpy as np
from tqdm import tqdm
import cv2, os, re, skimage, time
import easydict as ed
import h5py
from skimage import measure
from skimage.draw import ellipsoid
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # 这一句虽然显示灰色，但是去掉会报错。
import matplotlib
from collections import Counter

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Utils3D:
    """Utils3D Based On Utils3D

    deal with the following type of 3d controllers

    1. mesh - stl format
    2. point cloud - ply format
    3. voxel - h5 format

    ##############################
    Directory Filter Functions
    ##############################

        ### 3D controllers names getter
            get3DNamesUnsort (root, tl)
                :add file name which matches the "^pt[0-9]+.*" pattern
                    and ends with tl to a list

    """

    @staticmethod
    def getGrayStoneFromNamePaths(image_path, H=1000, W=1000):
        num_of_images = len(image_path)
        stone = np.zeros((num_of_images, H, W), dtype=np.uint8)
        for i in range(num_of_images):
                gray = cv2.imread(image_path[i], 0)
                stone[i] = cv2.resize(gray, (W,H))
        return stone

    @staticmethod
    def get3DNamesUnsort(root, tl, forbid_tl):
        assert root.endswith('/')
        stls_uncleaned = os.listdir(root)
        stls = []
        stls_without_root = []
        for stl in stls_uncleaned:
            if re.match("^pt[0-9]+.*", stl) and stl.endswith(tl) and not stl.endswith(forbid_tl):
                stls.append(root + stl)
                stls_without_root.append(stl)
        return stls

    @staticmethod
    def getVox(img_paths, use_num, L, H, W, binary=True, rgb=False, hint=True):
        start = time.clock()

        ''' Body Start '''
        nums = len(img_paths)

        if hint:
            Format.utilsHead('Creating Voxels')
            print('There are %d needed images' % nums)
            print('Using %d images to generate voxels ' % use_num)

        if rgb:
            voxels = np.zeros((use_num, H, W, 3), dtype=np.uint8)
        else:
            voxels = np.zeros((use_num, H, W), dtype=np.uint8)

        cnt = 0
        bar = tqdm(list(range(use_num)))
        for i in bar:
            if rgb:
                voxels[cnt] = Utils2D.readRGB(img_paths[i], True, H, W)
            else:
                if binary:
                    voxels[cnt] = Utils2D.readBinary(img_paths[i], True, H, W)
                else:
                    voxels[cnt] = Utils2D.readGray(img_paths[i], True, H, W)
            bar.set_description("Processing mask %d" % (i + 1))
            cnt += 1
        voxels = Utils3D.resizeVox(voxels, L, H, W, True)
        ''' Body End '''

        elapsed = (time.clock() - start)
        if hint:
            Format.tick(elapsed)
            Format.utilsTail()
        return voxels

    @staticmethod
    def resizeVox(vox, L=120, H=120, W=120, rgb=False, hint=True):
        if hint:
            Format.utilsHead('Resizing Voxels')
            print('Voxels will be resized to %d*%d*%d' % (L, H, W))
        vox = vox.astype(np.uint8)
        vox_len = len(vox)
        if rgb:
            res_vox_0 = np.zeros((vox_len, H, W), dtype=np.uint8)
        else:
            res_vox_0 = np.zeros((vox_len, H, W, 3), dtype=np.uint8)
        res_vox = np.zeros((L, H, W), dtype=np.uint8)

        bar = tqdm(list(range(vox_len)))
        for i in bar:
            res_vox_0[i] = cv2.resize(vox[i], (W, H))
            bar.set_description("First round resize, resizing slice %d" % (i + 1))

        bar2 = tqdm(list(range(W)))
        for i in bar2:
            temp = res_vox_0[:vox_len, :H, i]
            res_vox[:L, :H, i] = cv2.resize(temp, (H, L))
            bar2.set_description("Second round resize, resizing slice %d" % (i + 1))

        if hint:
            Format.utilsTail()
        return res_vox

    @staticmethod
    def getThreeViews(vox):
        # (silce, H, W)
        # left_view
        left_view = np.max(vox,2)
        upper_view = np.max(vox,1)
        front_view = np.max(vox,0)
        return left_view, upper_view, front_view

    @staticmethod
    def getPadFossilFromStone(vox, pad, min_slice, max_slice, min_h, max_h, min_w, max_w):
        real_min_slice = np.max([0,min_slice-pad]).astype(np.int64)
        real_min_h = np.max([0,min_h-pad]).astype(np.int64)
        real_min_w = np.max([0,min_w-pad]).astype(np.int64)
        real_max_slice = np.min([vox.shape[0]-1,max_slice+pad]).astype(np.int64)
        real_max_h = np.min([vox.shape[1]-1,max_h+pad]).astype(np.int64)
        real_max_w = np.min([vox.shape[2]-1,max_w+pad]).astype(np.int64)

        if(real_max_slice < real_min_slice):
            real_max_slice = real_min_slice + 1

        if (real_max_h < real_min_h):
            real_max_h = real_min_h + 1

        if(real_max_w < real_min_w):
            real_max_w = real_min_w + 1


        return vox[real_min_slice:real_max_slice+1,real_min_h:real_max_h+1,real_min_w:real_max_w+1]

    @staticmethod
    def seperateMaskVoxelsGetFeaturesAndNamesAndPts(voxels, originalIndexes, fixed_num=None,
                                                    needPts=True, hint=True, needNamesOrPts = True):
        assert type(voxels) is np.ndarray
        assert len(voxels.shape) == 3
        if hint:
            Format.utilsHead('Extracting Seperate Voxels Feature')
            print('Voxels shape is %s' % (str(voxels.shape)))
        vx, num = skimage.measure.label(voxels, connectivity=1, return_num=True)
        features = np.zeros((num, 10), dtype=np.float64)

        # Illustration for features
        features[:, 4] = 10000
        features[:, 6] = 10000
        features[:, 8] = 10000
        features[:, 5] = 0
        features[:, 7] = 0
        features[:, 9] = 0
        bar = tqdm(list(range(vx.shape[0])))

        # 0 slice average coordinates
        # 1 H average coordinates
        # 2 W average coordinates
        # 3 voxel sum number
        # 4 min slice
        # 5 max slice
        # 6 min H
        # 7 max H
        # 8 min W
        # 9 max W

        for i in bar:
            for j in range(vx.shape[1]):
                for k in range(vx.shape[2]):
                    if vx[i, j, k] != 0:
                        label = vx[i, j, k] - 1
                        features[label, 0] += i
                        features[label, 1] += j
                        features[label, 2] += k
                        features[label, 3] += 1
                        if i < features[label, 4]:
                            features[label, 4] = i
                        if i > features[label, 5]:
                            features[label, 5] = i

                        if j < features[label, 6]:
                            features[label, 6] = j
                        if j > features[label, 7]:
                            features[label, 7] = j

                        if k < features[label, 8]:
                            features[label, 8] = k
                        if k > features[label, 9]:
                            features[label, 9] = k
            bar.set_description('Processing voxels positional features %d' % (i + 1))
        for i in range(3):
            features[:, i] //= features[:, 3]

        names = []
        pts = []
        if needNamesOrPts:
            bar2 = tqdm(list(range(num)))
            for i in bar2:
                x_center_index = int(features[i][0] * len(originalIndexes) / vx.shape[0])
                index = originalIndexes[x_center_index]
                if fixed_num is not None:
                    center_slice_name = str(index).zfill(fixed_num)
                else:
                    center_slice_name = str(index)
                x_min = int(features[i][4])
                x_max = int(features[i][5])

                h_min = int(features[i][6])
                h_max = int(features[i][7])

                w_min = int(features[i][8])
                w_max = int(features[i][9])
                name = 'pt%d_x_center_%s_xMin_%d_xMax_%d_hMin_%d_hMax_%d_wMin_%d_wMax_%d_voxels_shape_%d*%d*%d' % \
                       (i, center_slice_name, x_min, x_max, h_min, h_max, w_min, w_max, vx.shape[0], vx.shape[1],
                        vx.shape[2])
                names.append(name)
                if needPts:
                    pts.append(Utils3D.voxels2ply(vx, label=i + 1, rgb=False, hint=False)[0])
                bar2.set_description('Creating point clouds from voxels and names')

        if hint:
            Format.utilsTail()
        return features, vx, names, pts

    @staticmethod
    def fuseVoxelsAndGetMetrics(predicted, predicted_labeled, gt,gt_labeled, position, position_com, names, names_com, use_gt = False, save_path=None, resize=False, L=60, H=60, W=60,
                                hint=True):
        if hint:
            Format.utilsHead('Saving voxels separately')
        if save_path is not None:
            Utils2D.check_dir(save_path)
        voxels_info = []
        bar = tqdm(list(range(len(position))))
        for i in bar:

            x_min = int(position[i][4])
            x_max = int(position[i][5])

            y_min = int(position[i][6])
            y_max = int(position[i][7])

            z_min = int(position[i][8])
            z_max = int(position[i][9])

            predicted_slice = predicted[x_min:x_max, y_min:y_max, z_min:z_max]
            if np.min(predicted_slice.shape) == 0:
                continue

            gt_slice = gt_labeled[x_min:x_max, y_min:y_max, z_min:z_max]
            result = Counter(gt_slice.flatten())
            d = sorted(result.items(), key=lambda x: x[1], reverse=True)
            sign = 0
            for j in range(len(d)):
                if d[j][0] != 0:
                    sign = d[j][0]
                    break

            if sign != 0:
                sign = sign - 1
                x_min = np.min([int(position[i][4]),int(position_com[sign][4])])
                x_max = np.max([int(position[i][5]),int(position_com[sign][5])])

                y_min = np.min([int(position[i][6]),int(position_com[sign][6])])
                y_max = np.max([int(position[i][7]),int(position_com[sign][7])])

                z_min = np.min([int(position[i][8]),int(position_com[sign][8])])
                z_max = np.max([int(position[i][9]),int(position_com[sign][9])])

                predicted_slice = predicted[x_min:x_max, y_min:y_max, z_min:z_max]
                gt_slice = gt[x_min:x_max, y_min:y_max, z_min:z_max]

            if resize:
                predicted_slice = Utils3D.resizeVox(predicted_slice, L, H, W)
                gt_slice = Utils3D.resizeVox(gt_slice, L, H, W)

            metrics = Metrics.computeMetrics(predicted_slice, gt_slice)
            metricsVox = Utils3D.getOverUnderSegVoxels(predicted_slice, gt_slice)
            sgl_info = {}
            sgl_info['predicted'] = predicted_slice
            sgl_info['gt'] = gt_slice
            sgl_info['visual'] = metricsVox
            sgl_info['metrics'] = metrics
            if use_gt:
                sgl_info['name'] = names_com[i]+'_name_gt'
            else:
                sgl_info['name'] = names[i]
            voxels_info.append(sgl_info)

            if save_path is not None:
                if use_gt:
                    h5_file_path = os.path.join(save_path, names_com[i] + "_name_gt.h5")
                else:
                    h5_file_path = os.path.join(save_path, names[i] + ".h5")
                f = h5py.File(h5_file_path, 'w')
                f['predicted'] = predicted_slice
                f['gt'] = gt_slice
                f['visual'] = metricsVox
                # f['metrics'] = metrics h5 不支持
                f.close()

            bar.set_description('Saving separate voxels')

        return voxels_info

    @staticmethod
    def parseName(ptName):
        ptInfo = ed.EasyDict()
        ptInfo.name = ptName
        names = ptName.split('/')[-1]
        names = names.split('_')
        ptInfo.ptIndex = names[0].split('pt')[1]
        ptInfo.center_slice = names[3]
        ptInfo.xMin = int(names[5])
        ptInfo.xMax = int(names[7])
        ptInfo.yMin = int(names[9])
        ptInfo.yMax = int(names[11])
        ptInfo.zMin = int(names[13])
        ptInfo.zMax = int(names[15])
        return ptInfo

    ply_header_with_color = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    \n
    '''

    ply_header_without_color = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    end_header
    \n
    '''

    @staticmethod
    def savePly(vertices, filename, colors=None):
        if colors is not None:
            colors = colors.reshape(-1, 3)
            vertices = np.hstack([vertices.reshape(-1, 3), colors])
            np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  # 必须先写入，然后利用write()在头部插入ply header
        else:
            vertices = vertices.reshape(-1, 3)
            np.savetxt(filename, vertices, fmt='%f %f %f')

        if colors is not None:
            head = Utils3D.ply_header_with_color % dict(vert_num=len(vertices))
        else:
            head = Utils3D.ply_header_without_color % dict(vert_num=len(vertices))
        with open(filename, 'r+') as f:
            old = f.read()
            f.seek(0)
            f.write(head)
            f.write(old)

    @staticmethod
    def getOverUnderSegVoxels(predicted, gt, hint=True):
        start = time.clock()
        if hint:
            Format.utilsHead('Generating Over&Under Seg For Voxels')
        assert type(predicted) == np.ndarray
        assert type(gt) == np.ndarray
        assert predicted.shape == gt.shape
        assert len(predicted.shape) == 3
        l = predicted.shape[0]
        h = predicted.shape[1]
        w = predicted.shape[2]

        color_pp = (200, 190, 200)  # pred exactly  (r,g,b)
        color_pn = (0, 255, 200)  # over segmented: pred more than GT
        color_np = (200, 50, 255)  # under segmented: failed to pred areas in GT
        color_nn = (0, 0, 0)  # pred exactly background

        blended_voxels = np.zeros((l, h, w, 3), dtype=np.uint8)

        blended_voxels[np.where(np.logical_and(predicted == True, gt == True) == True)] = color_pp
        blended_voxels[np.where(np.logical_and(predicted == True, gt == False) == True)] = color_pn
        blended_voxels[np.where(np.logical_and(predicted == False, gt == True) == True)] = color_np
        blended_voxels[np.where(np.logical_and(predicted == False, gt == False) == True)] = color_nn
        elapsed = (time.clock() - start)
        if hint:
            Format.tick(elapsed)
            Format.tail()
        return blended_voxels

    @staticmethod
    def voxels2ply(voxels, label=1, rgb=False, color=[0, 0, 0], hint=True):
        start = time.clock()
        if hint:
            Format.utilsHead('voxels2ply')
        points = []
        colors = []
        bar = tqdm(list(range(voxels.shape[0])))
        if rgb:
            assert len(voxels.shape) == 4 and voxels.shape[3] == 3
            for i in bar:
                for j in range(voxels.shape[1]):
                    for k in range(voxels.shape[2]):
                        if np.sum(voxels[i, j, k]) != 0:
                            points.append([i, j, k])
                            colors.append(voxels[i, j, k])
                bar.set_description('processing line %d' % (i + 1))
        else:
            points = np.asarray(np.where(voxels == label)).transpose((1, 0))
            colors = np.ones_like(points)
            colors = colors * np.asarray(color)

        points = np.asarray(points).astype(np.uint8)
        colors = np.asarray(colors).astype(np.uint8)
        elapsed = (time.clock() - start)
        if hint:
            Format.tick(elapsed)
            Format.tail()
        return points, colors

    @staticmethod
    def marchingCubes():

        # Generate a level set about zero of two identical ellipsoids in 3D
        ellip_base = ellipsoid(6, 10, 16, levelset=True)
        print(np.min(ellip_base))
        print(ellip_base.dtype)
        ellip_double = np.concatenate((ellip_base[:-1, ...],
                                       ellip_base[2:, ...]), axis=0)
        print(ellip_double.shape)
        # Use marching cubes to obtain the surface mesh of these ellipsoids
        verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_double, 0)

        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')

        ax.add_collection3d(mesh)

        ax.set_xlabel("x-axis: a = 6 per ellipsoid")
        ax.set_ylabel("y-axis: b = 10")
        ax.set_zlabel("z-axis: c = 16")

        ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
        ax.set_ylim(0, 20)  # b = 10
        ax.set_zlim(0, 32)  # c = 16

        plt.tight_layout()
        plt.show()



    @staticmethod
    def showVoxels(voxels):
        # notice this is extremely slow
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, edgecolor="k")
        plt.show()

'''
References:
    [1][使用matplotlab可视化体素voxel](https://blog.csdn.net/york1996/article/details/101538555)
'''
