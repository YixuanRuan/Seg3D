## 算法层
使用说明
将configs，logicLayer, utils文件夹放入项目中
然后在要使用的文件中，加入
from logicLayer import LogicLayerInterface
然后
stone = LogicLayerInterface.getStone(dir_path)
通过Class::Stone的属性访问即可

默认放缩图片到 1000 * 1000
ToDo：
根据Slice和点击位置，寻找连通域

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

