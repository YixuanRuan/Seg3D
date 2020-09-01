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

#### 01-02-02. 局部化石分割02

```python
getNewThreeView2(iteration,origin,pad):
```

Origin: stone里面的 stone属性，代表的是石头的原始数据

Pad，手动选择重新分割在当前Fossil范围上扩大的范围

Iteration：iteration越大pad可以越小，因为iteration越大包围圈一般缩小更严重。

#### 01-02-03. 局部化石分割03

在调用下面这个函数之前需要调用

```python
getGT3(gt_stone,pad)
```
上面下面的pad需要一致
```python
getNewThreeView3(origin,pad)
```

Origin: stone里面的 stone属性，代表的是石头的原始数据

Pad，手动选择重新分割在当前Fossil范围上扩大的范围

使用了上面的函数后fossil.metrics_3储存了最新的metrics

#### 01-03. 获取化石分割Metrics

##### 关于数据

https://pan.baidu.com/s/1ih3aC8ma9MPDiq3qI18yVQ

提取码：UBu9

记得只能留.png和_mask_gt.png后缀的文件在文件夹里

##### 如何使用

参考test_.py
由于使用者不一定有gt image所以接口操作复杂了几步  

```python
stone = LogicLayerInterface.getStone('./samples/')
gt_stone = stone.getGTStone('./samples/')
fossil = gt_stone.getThreeViewByIndexAndHWPosition(0,100,200)
fossil.getGT(gt_stone, pad=0)
metrics = fossil.getMetrics(1)
```

1. 使用 gt_stone = stone.getGTStone('./samples/')获取gt石头
2. 使用 fossil.getGT(gt_stone,pad=n)获取fossil在gt stone 对应位置的gt fossil
3. 使用 fossil.getMetrics(initial=1)，如果initial是1代表没有进行重新分割，如果是0可以不输入参数，
因为0是默认值，默认对重新运行分割算法的区域进行metrics计算。
4. 根据返回的metrics进行信息获取就行，是个dictionary，滑动到readme最后看属性。

#### 01-03-02. 获取化石分割Metrics 02

获取gt_stone后进行下面操作

**Step 1**

```python
fossil.getGT2(gt_stone,pad)
```

Pad：注意这个pad必须和上面01-02-02. 局部化石分割02那个pad相同

**Step2**

```
fossil.getMetrics2( initial = 0):
```

没变

### 2. 接口

#### 函数01 getStone

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
    self.labeled_fossil_features = None  
    self.labeled_morph_stone = None  
    self.gt_stone  
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
	self.reseged_slices = None  
	self.reseged_binary_voxel_2 = None  
	self.reseged_three_view_2 = None    self.gt_fossil  
    self.metrics  

### Metrics::Dictionary
```python
picScore['accuracy'] = accuracy  
picScore['under_seg'] = under_seg  
picScore['over_seg'] = over_seg  
picScore['precision'] = precision  
picScore['recall'] = recall  
picScore['f1_score'] = f1_score  
picScore['iou'] = iou  
picScore['TP'] = TP  
picScore['FP'] = FP  
picScore['FN'] = FN  
picScore['TN'] = TN  
picScore['Total'] = total  
```


​    




