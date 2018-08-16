#python 
#similarity distance


    今天一个偶然的机会体会到了python数据运算的强大。求一个数组各样本之间的距离仅简单的几句代码就行。看来真的技术改变世界啊。废话不多说，记下来怕以后忘记。  

[python]  view plain  copy

    from scipy.spatial.distance import pdist, squareform  

下面结合API文档标注一下具体用法：

[python]  view plain  copy

    1.X = pdist(X, 'euclidean')  

计算数组X样本之间的欧式距离 返回值为 Y 为压缩距离元组或矩阵（以下等同）

[python]  view plain  copy

    2. X = pdist(X, 'minkowski', p)  

计算数组样本之间的明氏距离 
[python]  view plain  copy

    3. Y = pdist(X, 'cityblock')  

计算数组样本之间的曼哈顿距离
[python]  view plain  copy

    4. X = pdist(X, 'seuclidean', V=None)  

计算数组样本之间的标准化欧式距离 ，v是方差向量，表示 v[i]表示第i个分量的方差，如果缺失。默认自动计算。
[python]  view plain  copy

    5. X = pdist(X, 'sqeuclidean')  

计算数组样本之间欧式距离的平方

[python]  view plain  copy

    6. X = pdist(X, 'cosine')  

计算数组样本之间余弦距离 公式为：
[python]  view plain  copy

    7. X = pdist(X, 'correlation')  

计算数组样本之间的相关距离。

[python]  view plain  copy

    8.X = pdist(X, 'hamming')  

计算数据样本之间的汉明距离
[python]  view plain  copy

    9. X = pdist(X, 'jaccard')  

计算数据样本之间的杰卡德距离

[python]  view plain  copy

    10. X = pdist(X, 'chebyshev')  

计算数组样本之间的切比雪夫距离

[python]  view plain  copy

    11. X = pdist(X, 'canberra')  

[python]  view plain  copy

    计算数组样本之间的堪培拉距离  

[python]  view plain  copy

    12. X = pdist(X, 'mahalanobis', VI=None)  

[python]  view plain  copy

    计算数据样本之间的马氏距离  

还有好多不常用的距离就不一一写出了，如果想查阅可以点 点我，点我

[python]  view plain  copy

    除了对指定的距离计算该函数还可以穿lmbda表达式进行计算，如下  

[python]  view plain  copy

    dm = pdist(X, lambda u, v: np.sqrt(((u-v)**2).sum()))  

二、得到压缩矩阵后还需下一步即：

[python]  view plain  copy

    Y=scipy.spatial.distance.squareform(X, force='no', checks=True)  

[python]  view plain  copy

    其中，X就是上文提到的压缩矩阵Y，force 如同MATLAB一样，如果force等于‘tovector’ or ‘tomatrix’,输入就会被当做距离矩阵或距离向量。  

[python]  view plain  copy

    cheak当X-X.T比较小或diag(X)接近于零，是是有必要设成True的，<span style="font-family: Arial, Helvetica, sans-serif;">返回值Y为一个距离矩阵Y[i，j]表示样本i与样本j的距离。</span> 