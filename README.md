# Denoise Diffusion Probabilistic Model

## 数据集

使用Flower102训练集，共8192张图



## 模型

### Timestep Scheduler

timestep scheduler写了2个，参考了Improved DDPM那篇论文写了cosine scheduler，生成的图像均使用的是cosine scheduler

### Timestep Position Embedding

PE函数中的i代表的是channel的编号（一个channel对应一个词，一个channel的二维向量对应一个词向量），也就是说同一个图像中的同一个channel的所有像素嵌入的值是一样的