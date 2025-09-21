## OpenCV 学习笔记

### 一、OpenCV 基础介绍

OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉和机器学习软件库，它包含了数百个计算机视觉算法，广泛应用于图像识别、视频处理、目标检测等领域。

OpenCV 采用模块化结构，包含众多功能模块：

- **core**：核心基础数据结构模块，定义了基本的数据结构（如 `Mat` 矩阵，用于存储图像等数据）以及相关的基础操作函数。
- **imgproc**：图像处理模块，涵盖了图像滤波（如高斯滤波、中值滤波）、几何变换（如旋转、缩放）、形态学操作（如膨胀、腐蚀）等多种图像处理算法。
- **video**：视频分析模块，包含运动估计、背景减除、目标跟踪等视频处理相关的功能。
- **calib3d**：相机标定模块，用于相机的标定（如单目、双目相机标定）以及三维重建等任务。
- **objdetect**：目标检测模块，可用于检测预定义的目标（如人脸、眼睛、嘴巴等）。

### 二、OpenCV 基本操作

#### 1. 加载图像

- **函数**：`cv2.imread()`

- **参数**：

  - 第一个参数为图像路径，可以是相对路径（如当前文件夹下的 `lenna.jpg`）或绝对路径（如 `D:\OpenCV\samples\lenna.jpg`）。
  - 第二个参数为读取模式，常见的有：
    - `cv2.IMREAD_COLOR`：加载彩色图像，默认值为 `1`。
    - `cv2.IMREAD_GRAYSCALE`：加载灰度图像，默认值为 `0`。
    - `cv2.IMREAD_UNCHANGED`：加载包含透明通道的图像，默认值为 `-1`。

- **示例**：

  ```python
  import cv2
  import matplotlib.pyplot as plt
  # 加载彩色图像
  img = cv2.imread('data/fruits.jpg')
  # 由于 OpenCV 以 BGR 模式加载图像，而 matplotlib 以 RGB 模式显示，需转换
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img_rgb)
  plt.show()
  ```

- **注意事项**：若图像路径错误，`cv2.imread()` 不会报错，只是返回 `None`，可通过判断返回值是否为 `None` 来检查图像是否加载成功。

#### 2. 绘图功能

OpenCV 提供了多种在图像上绘图的函数：

- **绘制直线**：`cv2.line(img, pt1, pt2, color, thickness)`

  - `img`：要绘制直线的图像。
  - `pt1`：直线起点坐标。
  - `pt2`：直线终点坐标。
  - `color`：直线颜色（BGR 格式，如 `(0, 0, 255)` 表示红色）。
  - `thickness`：直线粗细，若为 `-1` 表示填充（用于闭合图形）。

- **绘制矩形**：`cv2.rectangle(img, pt1, pt2, color, thickness)`

  - `pt1`：矩形左上角坐标。
  - `pt2`：矩形右下角坐标。

- **绘制圆形**：`cv2.circle(img, center, radius, color, thickness)`

  - `center`：圆心坐标。
  - `radius`：圆的半径。

- **添加文字**：`cv2.putText(img, text, org, fontFace, fontScale, color, thickness)`

  - `text`：要添加的文字内容。
  - `org`：文字起始坐标（左下角为起点）。
  - `fontFace`：字体类型。
  - `fontScale`：字体缩放比例。

- **示例**：

  ```python
  import cv2
  import matplotlib.pyplot as plt
  img = cv2.imread('data/lena.jpg')
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # 绘制直线
  cv2.line(img_rgb, (0, 0), (500, 500), (255, 0, 0), 5)
  # 绘制矩形
  cv2.rectangle(img_rgb, (100, 100), (200, 200), (0, 255, 0), 3)
  # 绘制圆形
  cv2.circle(img_rgb, (250, 250), 75, (0, 0, 255), -1)
  # 添加文字
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img_rgb, 'hello', (100, 300), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
  plt.imshow(img_rgb)
  plt.show()
  ```

#### 3. 图像几何变换

- **缩放**：`cv2.resize(src, dsize, fx, fy, interpolation)`

  - `src`：输入图像。
  - `dsize`：输出图像尺寸。
  - `fx`、`fy`：水平和垂直方向的缩放因子。
  - `interpolation`：插值方法，常见的有 `cv2.INTER_LINEAR`（线性插值）、`cv2.INTER_NEAREST`（最近邻插值）等。

- **翻转**：`cv2.flip(src, flipCode)`

  - `flipCode`：翻转模式，`0` 表示沿 x 轴翻转，`1` 表示沿 y 轴翻转，`-1` 表示同时沿 x、y 轴翻转。

- **平移**：通过定义变换矩阵，再使用 `cv2.warpAffine()` 实现

  - 变换矩阵 `M` 为 `np.float32([[1, 0, tx], [0, 1, ty]])`，其中 `tx` 为 x 方向平移量，`ty` 为 y 方向平移量。

- **示例**：

  ```python
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  img = cv2.imread('data/fruits.jpg')
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # 缩放
  resized = cv2.resize(img_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
  # 翻转
  flipped = cv2.flip(img_rgb, 1)
  # 平移
  rows, cols, _ = img_rgb.shape
  M = np.float32([[1, 0, 100], [0, 1, 50]])
  translated = cv2.warpAffine(img_rgb, M, (cols, rows))
  plt.subplot(131), plt.imshow(resized), plt.title('Resized')
  plt.subplot(132), plt.imshow(flipped), plt.title('Flipped')
  plt.subplot(133), plt.imshow(translated), plt.title('Translated')
  plt.show()
  ```

#### 4. 其他基础操作

- **感兴趣区域（ROI）**：可通过数组切片的方式获取图像的特定区域并进行操作，例如 `roi = img[100:200, 150:250]`，表示获取图像中 y 坐标从 100 到 200，x 坐标从 150 到 250 的区域。

- **图像通道分割与合并**：

  - 分割：`b, g, r = cv2.split(img)`，将 BGR 图像分割为蓝、绿、红三个通道。
  - 合并：`merged = cv2.merge((b, g, r))`，将分割后的通道重新合并为 BGR 图像。

- **颜色空间转换**：

  - 常见的转换有 `cv2.COLOR_BGR2GRAY`（BGR 转灰度）、`cv2.COLOR_BGR2HSV`（BGR 转 HSV）等，例如 `gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`。

- **基于颜色的目标跟踪**：在 HSV 颜色空间中，通过设置目标颜色的范围，使用 `cv2.inRange()` 函数获取掩码，再对掩码进行操作来跟踪目标，例如跟踪蓝色物体：

  ```python
  import cv2
  import numpy as np
  import matplotlib.pyplot as plt
  img = cv2.imread('data/blue_object.jpg')
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # 蓝色的 HSV 范围
  lower_blue = np.array([110, 50, 50])
  upper_blue = np.array([130, 255, 255])
  mask = cv2.inRange(hsv, lower_blue, upper_blue)
  res = cv2.bitwise_and(img, img, mask=mask)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
  plt.subplot(131), plt.imshow(img_rgb), plt.title('Original')
  plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title('Mask')
  plt.subplot(133), plt.imshow(res_rgb), plt.title('Result')
  plt.show()
  ```

### 三、算术与逻辑操作

#### 1. 算术操作

- **加法**：`cv2.add(src1, src2, dst, mask, dtype)`，将两幅图像对应像素的灰度值或颜色分量相加，若和大于 255 则取 255。也可使用 `src1 + src2`，但这种方式是饱和操作（超过 255 则取模），与 `cv2.add` 效果不同。

- **减法**：`cv2.subtract(src1, src2, dst, mask, dtype)`，将 `src1` 对应像素值减去 `src2` 对应像素值。

- **乘法**：`cv2.multiply(src1, src2, dst, scale, dtype)`，将两幅图像对应像素值相乘，结果若大于 255 则取 255。

- **除法**：`cv2.divide(src1, src2, dst, scale, dtype)`，将 `src1` 对应像素值除以 `src2` 对应像素值。

- **示例（乘法）**：

  ```python
  import cv2
  import matplotlib.pyplot as plt
  img = cv2.imread('data/lenna.bmp')
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # 二值化，灰度值大于 127 为 255，小于 127 为 0
  ret, th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
  img_multiple = cv2.multiply(img_gray, th)
  titles = ["Original", "Binary", "Multiple"]
  images = [img_gray, th, img_multiple]
  for i in range(3):
      plt.subplot(1, 3, i + 1)
      plt.imshow(images[i], cmap='gray')
      plt.title(titles[i])
      plt.xticks([])
      plt.yticks([])
  plt.show()
  ```

#### 2. 逻辑操作

- **与（AND）**：`cv2.bitwise_and(src1, src2, dst, mask)`，对两幅图像的对应像素进行按位与操作。

- **或（OR）**：`cv2.bitwise_or(src1, src2, dst, mask)`，对两幅图像的对应像素进行按位或操作。

- **非（NOT）**：`cv2.bitwise_not(src, dst, mask)`，对图像的每个像素进行按位取反操作。

- **异或（XOR）**：`cv2.bitwise_xor(src1, src2, dst, mask)`，对两幅图像的对应像素进行按位异或操作（若两个值不同，结果为 1；若相同，结果为 0）。

- **示例**：

  ```python
  import cv2
  import matplotlib.pyplot as plt
  import numpy as np
  img_A = cv2.imread('data/images/A.jpg')
  img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
  img_A = cv2.resize(img_A, (300, 300))
  # 生成圆形掩码
  circle = np.zeros((300, 300), dtype="uint8")
  cv2.circle(circle, (150, 150), 100, 255, -1)
  # 逻辑操作
  bitwise_And = cv2.bitwise_and(img_A, circle)
  bitwise_Or = cv2.bitwise_or(img_A, circle)
  bitwise_Not = cv2.bitwise_not(img_A)
  bitwise_Xor = cv2.bitwise_xor(img_A, circle)
  # 显示结果
  images = [img_A, bitwise_And, bitwise_Or, bitwise_Not, bitwise_Xor]
  titles = ['Original', 'bitwise_And', 'bitwise_Or', 'bitwise_Not', 'bitwise_Xor']
  for i in range(5):
      plt.subplot(2, 3, i + 1)
      plt.imshow(images[i], cmap="gray")
      plt.title(titles[i])
      plt.xticks([])
      plt.yticks([])
  plt.show()
  ```

### 四、OpenCV 使用相机与加载视频

#### 1. 相关函数

- **`cv2.VideoCapture()`**：用于打开相机或加载视频文件，进行视频帧的读取操作。
- **`cv2.VideoWriter()`**：用于将处理后的视频帧写入新的视频文件。
- **`IPython.display.Video()`**：在 IPython 环境下用于显示视频文件。

#### 2. 加载视频

通过 `IPython.display.Video()` 可以直接在 IPython 环境中加载并显示视频文件：

```python
# load video
import IPython
IPython.display.Video("data/demo_video.mp4")
```

#### 3. 打开相机

使用 `cv2.VideoCapture(0)` 打开默认相机（参数 `0` 表示默认相机，若有多个相机可修改参数选择），然后在循环中不断读取相机捕获的帧，将彩色帧转换为灰度帧并显示，按下 `q` 键可退出循环。

> 注意：在 AIStudio 在线环境中无法打开相机，该功能可在 PyCharm 等本地环境中使用。

```python
# open camera
# while you can use this function in pycharm, because AIStudio on line CAN'T open camera!!!
import cv2
capture = cv2.VideoCapture(0)

while(True):
    # get one fram
    ret, frame = capture.read()
    # color to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
# 释放资源
capture.release()
cv2.destroyAllWindows()
```



### 颜色跟踪代码讲解

#### 1. 准备工作：导入库和读取图片

```python
import cv2  # 用于图像处理
import numpy as np  # 用于数值运算
import matplotlib.pyplot as plt  # 用于图像显示

# 读取图像
img = cv2.imread("data/cornfield.bmp")
if img is None:
    print("错误: 无法读取图像。")
    exit()
```

- 首先导入了必要的库，然后用`cv2.imread()`读取图片
- 这里加了一个判断，如果图片读取失败（比如路径错误），会提示错误并退出

#### 2. 颜色空间转换：BGR → HSV
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

- 这是整个颜色提取的关键步骤
- OpenCV 读取的图片默认是 BGR 格式（蓝绿红），而 HSV 格式更适合颜色检测
- HSV 分别代表：Hue（色相，颜色种类）、Saturation（饱和度，颜色深浅）、Value（明度，亮度）

#### 3. 定义颜色范围（核心中的核心）

```python
# 蓝色范围
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# 红色范围（特殊：红色在HSV中是两个区间）
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])
```

- 这是告诉程序 "什么是红色"、"什么是蓝色"
- 每个颜色在 HSV 空间中都有固定的数值范围：
  - 蓝色的色相（H）在 110-130 之间
  - 红色比较特殊，因为它在 HSV 环的两端，所以需要两个范围（0-10 和 170-180）
  - 后面两个数值是饱和度和明度的范围，确保只提取色彩鲜艳、不太暗的区域

#### 4. 创建颜色掩码（筛选过程）
```python
# 为每种颜色创建掩码
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
```

- `cv2.inRange()`函数会生成一个 "掩码"（黑白图像）
  - 在设定的颜色范围内的像素会变成白色（255）
  - 不在范围内的像素会变成黑色（0）
- 红色因为有两个范围，所以用`|`（或运算）把两个掩码合并

#### 5. 合并掩码并提取颜色

```python
# 合并红色和蓝色掩码
combined_mask = mask_blue | mask_red

# 将合并后的掩码应用于原始图像
result = cv2.bitwise_and(img, img, mask=combined_mask)
```

- `combined_mask`是同时包含红色和蓝色区域的掩码（白色部分是要保留的）
- `cv2.bitwise_and()`用掩码 "过滤" 原始图像：
  - 只保留掩码中白色区域对应的原始图像像素
  - 掩码中黑色区域对应的像素会被过滤掉（变成黑色）

#### 6. 显示结果
```python
# 转换为RGB格式（因为matplotlib需要RGB格式）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blue_rgb = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask_blue), cv2.COLOR_BGR2RGB)
red_rgb = cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask_red), cv2.COLOR_BGR2RGB)
combined_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# 显示四张图：原图、蓝色物体、红色物体、红蓝合并
```

- 因为 OpenCV 用 BGR，而 matplotlib 用 RGB，所以需要转换格式才能正确显示颜色
- 最终会展示四个结果，方便对比

#### 总结一下整个流程

1. 读入图片 → 2. 转换为 HSV 格式 → 3. 告诉程序要找的颜色范围 → 4. 生成只保留目标颜色的掩码 → 5. 用掩码从原图中 "扣出" 目标颜色 → 6. 展示结果

如果提取效果不好，主要是因为颜色范围需要调整，你可以尝试修改`lower_blue`、`upper_blue`等数组中的数值来优化。