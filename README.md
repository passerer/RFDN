## RFDN

#### 文件说明
./model: 包含模型定义的文件

./data: 包含数据操作相关的文件

summery.py: 计算模型统计量，包括Flops/Activations/Number of conv

test_demo.py: 测试模型**精度 (psnr)** 和**耗时**

train.py: 训练

utils.py: 包含一些工具
#### 数据集
数据集[下载](https://drive.google.com/file/d/12hOYsMa8t1ErKj6PZA352icsx9mz1TwB/view?usp=drive_open)


建议数据集组织形式如下  
+ RFDN  
    + Train_Datasets
        + DIV2K
            + DIV2K_train_HR
            + DIV2K_train_LR_bicubic
                + X4
    + Test_Datasets
        + DIV2k_val
            + DIV2K_valid_HR
            + DIV2K_valid_LR_bicubic
                + X4

#### 参考指标
RFDN(nf=64) / params：704K / psnr:28.90 / runtime:113ms(on 1080Ti)
