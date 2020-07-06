from utils import Utils2D
from utils import Utils3D
from configs import opt_logic
from .stone_v1 import Stone

class Interface:
    @staticmethod
    def getStone(image_dir_path):
        image_manage = opt_logic.image_manage

        # Test 01 Start
        # 根据目录将图片序号提取出来并按顺序排列
        index_list = Utils2D.getSortedIndexes(image_dir_path, image_manage.pattern,
                                              image_manage.splitter, image_manage.index)
        # print(index_list)
        # Test 01 End

        # Test 02 Start
        # 根据序列获取图片路径序列
        image_path_name_list = Utils2D.get2DNames(image_dir_path, index_list,
                                                  image_manage.tail, fixed=False, fixed_num=4)
        # print(image_path_name_list)
        # Test 02 End

        # Test 03 Start
        gray_stone = Utils3D.getGrayStoneFromNamePaths(image_path_name_list, image_manage.scale_size_H,
                          image_manage.scale_size_W)
        # print(np.sum(gray_stone))
        # Test 03 End

        stone = Stone(gray_stone, index_list)

        return stone

if __name__ == '__main__':

    # If You Want To Run This Test Please change Line 4 to
    # from stone_v1 import Stone

    # Test 01-03 Start
    stone = Interface.getStone('../samples/')
    Utils2D.showOnePic(stone.morph_stone[0], "Hello")
    # Test 01-03 End

