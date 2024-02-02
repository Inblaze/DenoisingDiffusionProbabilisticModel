# Denoise Diffusion Probabilistic Model

## 数据集

使用Flower102训练集，共8192张图



## 模型

### Timestep Scheduler

timestep scheduler写了2个，参考了Improved DDPM那篇论文写了cosine scheduler，生成的图像均使用的是cosine scheduler

### Timestep Position Embedding

PE函数中的i代表的是channel的编号（一个channel对应一个词，一个channel的二维向量对应一个词向量），也就是说同一个图像中的同一个channel的所有像素嵌入的值是一样的



## 可能的问题

记录一些看别人代码和我的代码的不同之处。

1. 我的代码没有attention block

2. timestep position embedding方式各有不同

3. 最后忘了把像素值从[-1, 1]恢复成[0, 255]
   - 这条通过暴力clamp再乘以255解决了，但是发现图像大部分区域变黑，怀疑是网络最后output的最后一层卷积产生了许多像素值的绝对值超出1的数据，然后直接被clamp掉了，应该先BatchNorm一下的，但是来不及了

4. denoise的variance可能有问题

   

   