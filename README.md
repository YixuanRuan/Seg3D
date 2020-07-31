## 算法层
### 1. 使用说明
将configs，logicLayer, utils文件夹放入项目中
然后在要使用的文件中，加入
from logicLayer import LogicLayerInterface
然后
stone = LogicLayerInterface.getStone(dir_path)
通过Class::Stone的属性访问即可

默认放缩图片到 1000 * 1000

#### 01-01. 获取三视图

使用Stone类创建实例stone后使用stone.getThreeViewByIndexAndHWPosition(slice, h, w)

slice；比如有255张图片，如果我点击的是第100张，这个参数输入99  
h：高度，请输入0-999之间的数值，注意点击位置与1000\*1000图片的正确对应  
w：宽度，请输入0-999之间的数值，注意点击位置与1000\*1000图片的正确对应

#### 01-02. 局部化石分割

在上一步使用getThreeViewByIndexAndHWPosition后会获取一个Fossil对象
利用这个对象fossil.getNewThreeView(iteration)

iteration：正整数，1-正无穷，默认数值35，数值越大分割出来的化石越小（边缘越靠里面），缩小数值如输入20可以得到更大的化石

返回值有两个：self.reseged_three_view,  self.reseged_binary_voxel   
	self.reseged_three_view：新的三视图，在原来的基础上上下左右padding了100个像素后重新分割   
	self.reseged_binary_voxel：分割出来的3D化石矩阵


### 2. 接口
#### 函数01 genStone

输入：文件夹路径
输出：Class::Stone
要求：

1. 文件夹内部必须只含化石切片图片
2. 图片命名格式要求 切片位置.jpg 如 803.jpg

### 3. 主要类

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
    self.three_view   
    self.reseged_three_view  
    self.reseged_binary_voxel   
    




