# 本项目主要用于为yolov5-pytorch生成台标数据集
* [x] 支持多台标生成
* [x] 图像缩放
* [x] 图像加噪：椒盐、高斯
* [x] 图像旋转
* [x] 台标抠图
* **在运行`python3 build.py`之前，请先运行`python3 build.py --finetune`进行微调**
* **[./dataset/background](./dataset/background)和[./dataset/tvlogos](./dataset/tvlogos)的目录结构固定，不可改变；增加新的标签（签名）请在[./dataset/tvlogos](./dataset/tvlogos)目录之下新建对应的文件夹**
```
.
├── README.md
├── build.py
└── dataset
    ├── background  // 背景图文件夹
    ├── fakes       // fake台标文件夹
    └── tvlogos     // 目标台标文件夹
        ├── chengdu
        ├── hunan
        └── jiangsu
```