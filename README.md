# python-note

个人 Python 学习笔记仓库，用代码记录 NumPy 与 OpenCV 的入门练习。内容会持续补充。

## 学习内容

### NumPy

- 用 `ndarray` 理解矩阵形状、`reshape`、广播与矩阵乘法
- 封装 `Numpy` 类：零矩阵 / 全 1 / 单位阵、随机矩阵、运算符重载
- 索引与切片：按列取值、布尔掩码筛选、`axis` 与 `mean` 等聚合

### OpenCV

- 图像读写、显示、ROI 切片；BGR 与灰度、HSV 及 `inRange` 颜色过滤
- 缩放、摄像头采集（练习代码中已封装）
- 二值化：固定阈值与自适应阈值
- 边缘与轮廓：Canny → 形态学（膨胀 / 腐蚀 / 开闭运算）→ `findContours` → 按面积过滤 → 外接矩形
- 将上述步骤串成 `pipeline`：灰度 → 边缘 → 形态学 → 轮廓检测与框选

## 目录结构

```
python-note/
├── main.py           # 入口，切换 numpy / opencv 测试
├── numpy_/
│   ├── common.py     # Numpy 封装
│   └── test.py
├── opencv/
│   ├── common.py     # Opencv 封装
│   └── test.py
└── requirements.txt
```

## 环境

- Python 3.x
- 依赖见 `requirements.txt`：`numpy`、`opencv-python`

```bash
pip install -r requirements.txt
python main.py
```

运行 OpenCV 示例前，请在 `opencv/test.py` 或 `Opencv(...)` 构造参数中改为本机图片路径。

## 说明

- 各模块 `test.py` 中有部分注释掉的实验代码，可按需取消注释逐步尝试。
- 仓库以练习与备忘为主，非完整教程或生产项目。
