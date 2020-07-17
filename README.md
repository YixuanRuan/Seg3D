## 算法层
使用说明
将configs，logicLayer, utils文件夹放入项目中
然后在要使用的文件中，加入
from logicLayer import LogicLayerInterface
然后
stone = LogicLayerInterface.getStone(dir_path)
通过Class::Stone的属性访问即可

默认放缩图片到 1000 * 1000

1. 获取三视图
使用Stone类创建实例stone后使用stone.getThreeViewByIndexAndHWPosition(slice, h, w)

slice；比如有255张图片，如果我点击的是第100张，这个参数输入99
h：高度，请输入0-999之间的数值，注意点击位置与1000\*1000图片的正确对应
w：宽度，请输入0-999之间的数值，注意点击位置与1000\*1000图片的正确对应

### 接口
#### 函数01 genStone

输入：文件夹路径
输出：Class::Stone
要求：

1. 文件夹内部必须只含化石切片图片
2. 图片命名格式要求 切片位置.jpg 如 803.jpg

### Class::Stone
属性：
    self.stone ：灰度图3D Array
    self.slice_list：图片index array，int
    self.morph_stone：MorphACWE Algorithm Segmentation Stone
方法：
    self.getThreeViewByIndexAndHWPosition

### Class::Fossil
属性：
    self.binary_voxel
    self.gray_voxel
    self.labeled_id
    self.min_slice
    self.max_slice
    self.min_h
    self.max_h
    self.min_w
    self.max_w


