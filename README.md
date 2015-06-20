implicitMF
==========
实现参考 [http://labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf](http://labs.yahoo.com/files/HuKorenVolinsky-ICDM08.pdf)
训练数据来源 [http://grouplens.org/datasets/movielens](http://grouplens.org/datasets/movielens) 

### 环境依赖
+ Python 2.x
+ numPy（1.9.2或以上）
+ sciPy（0.15.1或以上）

### 运行方式
`recommend.py` 可以通过一批用户的评分，然后为每个用户做推荐。`recommend.py`接受一下参数：
+ `traing_examples`: 存储用户评分的文件，每一行有3列，分别为用户、item、评分，必需参数。
+ `recommend_count`: 每个用户推荐 item 个数，必需参数。
+ `num_users`: 用户个数，必需参数。
+ `num_items`: item 个数，必需参数。
+ `num_factor`: 隐含特征维数，可选参数，默认40。
+ `num_iterations`: 训练是迭代次数，可选参数，默认30。
+ `reg_param`: 正则化系数，可选参数，默认0.8。
+ `conv_loss_value`: 当损失函数小于该值时停止训练迭代，可选参数，默认为0。因为过程中计算损失函数开销比较大，把`conv_loss_value`设为0，可以跳过这一步而只通过一定次数的迭代终止训练过程。

运行示例：
```
python recommend.py \
    "training_examples" \
    10 \
    1000 \
    500 \
    > output
```

另外，`mf.py` 中的 `ImplicitMF` 类包含了主要的训练和推荐逻辑，可以通过导入这个包来自定义训练和推荐过程：
```
import mf
imcf = mf.ImplicitMF(ratings, num_factors, num_iterations, conv_loss_value, reg_param) #ratings 为用户对item评分的矩阵
imcf.train_model()
reco_items = imcf.recommend(user, reco_cnt) #user为要推荐的用户，reco_cnt为推荐item个数
```
