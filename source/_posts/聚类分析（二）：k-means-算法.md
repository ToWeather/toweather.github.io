---
title: 聚类分析（二）：k-means 算法
layout: post
date: 2018-02-10 20:33:10
tags:
- 聚类
- 非监督学习
- k-means 算法
categories:
- 机器学习算法
keywords: 聚类,k-means,中心点,非监督学习,clustering,machine learning

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# k-means 聚类算法
k-means 算法是一种经典的针对数值型样本的聚类算法。如前面的博文 [*聚类分析（一）：层次聚类算法*][1] 中所述，k-means 算法是一种基于中心点模型的聚类算法，它所产生的每一个 `cluster` 都维持一个中心点（为属于该 `cluster` 的所有样本点的均值，k-means 算法由此得名），每一个样本点被分配给距离其最近的中心点所对应的 `cluster`。在该算法中，`cluster` 的数目 \\( K \\) 需要事先指定。我们可以很清楚地看到，在以上聚类场景中，一个最佳的目标是找到 \\( K \\) 个中心点以及每个样本点的分配方法，使得每个样本点距其被分配的 `cluster` 所对应的中心点的平方 Euclidean 距离之和最小。而 k-means 算法正是为实现这个目标所提出。

## 表示为求解特定的优化问题
假定数据集为 \\( \\lbrace {\\bf x}\_1, {\\bf x}\_2, …, {\\bf x}\_N \\rbrace \\)，其包含 \\( N \\) 个样本点，每个样本点的维度为 \\( D \\)。我们的目的是将该数据集划分为 \\( K \\) 个 `cluster`，其中 \\( K \\) 是一个预先给定的值。假设每个 `cluster` 的中心点为向量 \\( {\\bf \\mu}\_k  \\in \\Bbb{R}^{d} \\)，其中 \\( k = 1, …, K \\)。如前面所述，我们的目的是找到中心点 \\( \\lbrace {\\bf \\mu}\_k \\rbrace \\)，以及每个样本点所属的类别，以使每个样本点距其被分配的 `cluster` 所对应的中心点的平方 Euclidean 距离之和最小。
为方便用数学符号描述该优化问题，我们以变量 \\( r\_{nk} \\in \\lbrace 0, 1 \\rbrace \\) 来表示样本点 \\( {\\bf x}\_n \\) 是否被分配至第 \\( k \\) 个 `cluster`，若样本点 \\( {\\bf x}\_n \\) 被分配至第 \\( k \\) 个 `cluster`，则 \\( r\_{nk} = 1 \\) 且 \\( r\_{nj} = 0  \\) \\( (j \\neq k) \\)。由此我们可以写出目标函数
$$  J = \\sum\_{n = 1}^{N} \\sum\_{k = 1}^{K} r\_{nk} \\| {\\bf x}\_n -  {\\bf \\mu}\_k \\|^{2} $$
它表示的即是每个样本点与其被分配的 `cluster` 的中心点的平方 Euclidean 距离之和。整个优化问题用数学语言描述即是寻找优化变量 \\( \\lbrace r\_{nk} \\rbrace \\) 和 \\( \\lbrace {\\bf \\mu}\_k \\rbrace \\) 的值，以使得目标函数 \\( J \\) 最小。
我们可以看到，由于 \\( \\lbrace r\_{nk} \\rbrace \\) 的定义域是非凸的，因而整个优化问题也是非凸的，从而寻找全局最优解变得十分困难，因此，我们转而寻找能得到局部最优解的算法。
k-means 算法即是一种简单高效的可以解决上述问题的迭代算法。k-means 算法是一种交替优化（alternative optimization）算法，其每一步迭代包括两个步骤，这两个步骤分别对一组变量求优而将另一组变量视为定值。具体地，首先我们为中心点 \\( \\lbrace {\\bf \\mu}\_k \\rbrace \\) 选定初始值；然后在第一个迭代步骤中固定 \\( \\lbrace {\\bf \\mu}\_k \\rbrace \\) 的值，对目标函数 \\( J \\) 根据  \\( \\lbrace r\_{nk} \\rbrace \\) 求最小值；再在第二个迭代步骤中固定 \\( \\lbrace r\_{nk} \\rbrace \\) 的值，对  \\( J \\) 根据 \\( {\\bf \\mu}\_k \\) 求最小值；如此交替迭代，直至目标函数收敛。
考虑迭代过程中的两个优化问题。首先考虑固定 \\( {\\bf \\mu}\_k \\) 求解 \\( r\_{nk} \\) 的问题，可以看到 \\( J \\) 是关于 \\( r\_{nk} \\) 的线性函数，因此我们很容易给出一个闭式解：\\( J \\) 包含 \\( N \\) 个独立的求和项，因此我们可以对每一个项单独求其最小值，显然，\\( r\_{nk} \\) 的解为
$$ r\_{nk} = \\begin{cases} 1, & \\text {if \\( k = \\rm {arg}  \\rm {min}\_{j}  \\| {\\bf x}\_n -  {\\bf \\mu}\_j \\|^{2} \\) } \\\\ 0, & \\text {otherwise} \\end{cases} $$
从上式可以看出，此步迭代的含义即是将每个样本点分配给距离其最近的中心点所对应的 `cluster`。
再来考虑固定  \\( r\_{nk} \\) 求解 \\( {\\bf \\mu}\_k \\) 的问题，目标函数 \\( J \\) 是关于 \\( {\\bf \\mu}\_k \\) 的二次函数，因此可以通过将 \\( J \\) 关于 \\( {\\bf \\mu}\_k \\) 的导数置为 0 来求解 \\( J \\) 关于  \\( {\\bf \\mu}\_k \\) 的最小值：
$$ 2 \\sum\_{n = 1}^{N} r\_{nk}({\\bf x}\_n -  {\\bf \\mu}\_k) = 0 $$
容易求出 \\( {\\bf \\mu}\_k \\) 的值为
$$ {\\bf \\mu}\_k = \\frac {\\sum\_{n} r\_{nk} {\\bf x}\_n} {\\sum\_{n} r\_{nk} } $$
该式表明，这一步迭代是将中心点更新为所有被分配至该 `cluster` 的样本点的均值。
k-means 算法的核心即是以上两个迭代步骤，由于每一步迭代均会减小或保持（不会增长）目标函数 \\( J \\) 的值，因而该算法最终一定会收敛，但可能会收敛至某个局部最优解而不是全局最优解。
虽然上面已经讲的很清楚了，但在这里我们还是总结一下 k-means 算法的过程：

1.  初始化每个 `cluster` 的中心点。最终的聚类结果受初始化的影响很大，一般采用随机的方式生成中心点，对于比较有特点的数据集可采用一些启发式的方法选取中心点。由于 k-means 算法收敛于局部最优解的特性，在有些情况下我们会选取多组初始值，对其分别运行算法，最终选取目标函数值最小的一组方案作为聚类结果；
2. 将每个样本点分配给距离其最近的中心点所对应的 `cluster`；
3. 更新每个 `cluster` 的中心点为被分配给该 `cluster` 的所有样本点的均值；
4. 交替进行 2～3 步，直至迭代到了最大的步数或者前后两次目标函数的值的差值小于一个阈值为止。

[ PRML 教材 ][2]中给出的 k-means 算法的运行示例非常好，这里拿过来借鉴一下，如下图所示。数据集为经过标准化处理（减去均值、对标准差归一化）后的 Old Faithful 数据集，记录的是黄石国家公园的 Old Faithful 热喷泉喷发的时长与此次喷发距离上次喷发的等待时间。我们选取 \\( K = 2 \\)，小图（a）为对中心点初始化，小图（b）至小图（i）为交替迭代过程，可以看到，经过短短的数次迭代，k-means 算法就已达到了收敛。
<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/illustration_of_k-means_algorithm.png" width = "660" height = "550" alt = "k-means 算法运行图示" align = center />
</div>

## 算法复杂度及其优缺点
### 算法复杂度
k-means 算法每次迭代均需要计算每个样本点到每个中心点的距离，一共要计算 \\( O(NK) \\) 次，而计算某一个样本点到某一个中心点的距离所需时间复杂度为 \\(  O(D) \\)，其中 \\( N \\) 为样本点的个数，\\( K \\) 为指定的聚类个数，\\( D \\) 为样本点的维度；因此，一次迭代过程的时间复杂度为 \\( O(NKD) \\)，又由于迭代次数有限，所以 k-means 算法的时间复杂度为 \\( O(NKD) \\)。  
实际实现中，一般采用高效的数据结构（如 kd 树）来结构化地存储对象之间的距离信息，因而可减小 k-means 算法运行的时间开销。

### 缺点
k-means 算法虽简单易用，但其有一些很明显的缺点，总结如下：
- 由于其假设每个 `cluster` 的先验概率是一样的，这样就容易产生出大小（指包含的样本点的多少）相对均匀的 `cluster`；但事实上的数据的 `cluster` 有可能并不是如此。
- 由于其假设每一个 `cluster` 的分布形状都为球形（spherical），（“球形分布”表明一个 `cluster` 内的数据在每个维度上的方差都相同，且不同维度上的特征都不相关），这样其产生的聚类结果也趋向于球形的 `cluster` ，对于具有非凸的或者形状很特别的 `cluster` 的数据集，其聚类效果往往很差。
- 由于其假设不同的 `cluster` 具有相似的密度，因此对于具有密度差别较大的 `cluster` 的数据集，其聚类效果不好。
- 其对异常点（outliers）很敏感，这是由于其采用了平方 Euclidean 距离作为距离衡量准则。
- `cluster` 的数目 \\( K \\) 需要预先指定，但由于很多情况下我们对数据也是知之甚少的，因而怎么选择一个合适的 \\( K \\) 值也是一个问题。一般确定 \\( K \\) 的值的方法有以下几种：a）选定一个 \\( K \\) 的区间，例如 2～10，对每一个 \\( K \\) 值分别运行多次 k-means 算法，取目标函数 \\( J \\) 的值最小的 \\( K \\) 作为聚类数目；b）利用 [ Elbow 方法 ][3] 来确定 \\( K \\) 的值；c）利用 [ gap statistics ][4] 来确定 \\( K \\) 的值；d）根据问题的目的和对数据的粗略了解来确定 \\( K \\) 的值。
- 其对初始值敏感，不好的初始值往往会导致效果不好的聚类结果（收敛到不好的局部最优解）。一般采取选取多组初始值的方法或采用优化初始值选取的算法（如 [ k-means++ 算法 ][5]）来克服此问题。
- 其仅适用于数值类型的样本。但其扩展算法 [ k-modes 算法 ][6] 适用于离散类型的样本。
其中前面三个缺点都是基于 k-means 算法的假设，这些假设的来源是 k-means 算法仅仅用一个中心点来代表 `cluster`，而关于 `cluster` 的其他信息一概没有做限制，那么根据 [ Occam 剃刀原理 ][7]，k-means 算法中的 `cluster` 应是最简单的那一种，即对应这三个假设。后面我们讲到 k-means 算法是 “EM 算法求解高斯混合模型的最大似然参数估计问题” 的特例的时候会得出这些假设中的部分。

### 优点
尽管 k-means 算法有以上这些缺点，但一些好的地方还是让其应用广泛，其优点总结如下：
- 实现起来简单，总是可以收敛，算法复杂度低。
- 其产生的聚类结果容易阐释。
- 在实际应用中，数据集如果不满足以上部分假设条件，仍然有可能产生比较让人满意的聚类结果。

---- 
# 实现 k-means 聚类
在这一部分，我们首先手写一个简单的 k-means 算法，然后用该算法来展示一下不同的初始化值对聚类结果的影响；然后再使用 scikit-learn 中的 KMeans 类来展示 k-means 算法对不同类型的数据集的聚类效果。

## 利用 python 实现 k-means 聚类
首先我们手写一个 k-means 聚类算法，这里，我将该算法封装成了一个类，代码如下所示：
```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import time

from sklearn.datasets import make_blobs

class KMeansClust():
    def __init__(self, n_clust=2, max_iter=50, tol=1e-10):
        self.data_set = None
        self.centers_his = []
        self.pred_label = None
        self.pred_label_his = []
        self.n_clust = n_clust
        self.max_iter = max_iter
        self.tol = tol

    def predict(self, data_set):
        self.data_set = data_set
        n_samples, n_features = self.data_set.shape
        self.pred_label = np.zeros(n_samples, dtype=int)
	
	start_time = time.time()
	
        # 初始化中心点
        centers = np.random.rand(self.n_clust, n_features)
        for i in range(n_features):
            dim_min = np.min(self.data_set[:, i])
            dim_max = np.max(self.data_set[:, i])
            centers[:, i] = dim_min + (dim_max - dim_min) * centers[:, i]
        self.centers_his.append(copy.deepcopy(centers))
        self.pred_label_his.append(copy.deepcopy(self.pred_label))

        print("The initializing cluster centers are: %s" % centers)

        # 开始迭代
        pre_J = 1e10
        iter_cnt = 0
        while iter_cnt <= self.max_iter:
            iter_cnt += 1
            # E 步：将各个样本点分配给距其最近的中心点所对应的 cluster
            for i in range(n_samples):
                self.pred_label[i] = np.argmin(np.sum((self.data_set[i, :] - centers) ** 2, axis=1))
            # M 步：更新中心点
            for i in range(self.n_clust):
                centers[i] = np.mean(self.data_set[self.pred_label == i], axis=0)
            self.centers_his.append(copy.deepcopy(centers))
            self.pred_label_his.append(copy.deepcopy(self.pred_label))
            # 重新计算目标函数 J
            crt_J = np.sum((self.data_set - centers[self.pred_label]) ** 2) / n_samples
            print("iteration %s, current value of J: %.4f" % (iter_cnt, crt_J))
            # 若前后两次迭代产生的目标函数的值变化不大，则结束迭代
            if np.abs(pre_J - crt_J) < self.tol:
                break
            pre_J = crt_J

        print("total iteration num: %s, final value of J: %.4f, time used: %.4f seconds" 
                % (iter_cnt, crt_J, time.time() - start_time))

    # 可视化算法每次迭代产生的结果
    def plot_clustering(self, iter_cnt=-1, title=None):
        if iter_cnt >= len(self.centers_his) or iter_cnt < -1:
            raise Exception("iter_cnt is not valid!")
        plt.scatter(self.data_set[:, 0], self.data_set[:, 1],
                        c=self.pred_label_his[iter_cnt], alpha=0.8)
        plt.scatter(self.centers_his[iter_cnt][:, 0], self.centers_his[iter_cnt][:, 1],
                        c='r', marker='x')
        if title is not None:
            plt.title(title, size=14)
        plt.axis('on')
        plt.tight_layout()
```

创建一个 `KMeansClust` 类的实例即可进行 k-means 聚类，在创建实例的时候，会初始化一系列的参数，如聚类个数、最大迭代次数、终止迭代的条件等等；然后该实例调用自己的方法 `predict` 即可对给定的数据集进行 k-means 聚类；方法 `plot_clustering` 则可以可视化每一次迭代所产生的结果。
利用 `KMeansClust` 类进行 k-means 聚类的代码如下所示：
```python
if __name__ == '__main__':
    # 生成数据集
    n_samples = 1500
    centers = [[0, 0], [5, 6], [8, 3.5]]
    cluster_std = [2, 1.0, 0.5]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)

    # 运行 k-means 算法
    kmeans_cluster = KMeansClust(n_clust=3)
    kmeans_cluster.predict(X)

    # 可视化中心点的初始化以及算法的聚类结果
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    kmeans_cluster.plot_clustering(iter_cnt=0, title='initialization centers')
    plt.subplot(1, 2, 2)
    kmeans_cluster.plot_clustering(iter_cnt=-1, title='k-means clustering result')
    plt.show()
```

以上代码首先由三个不同的球形高斯分布产生了一个数据集，而后运行了 k-means 聚类方法，中心点的初始化是随机生成的。最终得到如下的输出和可视化结果：
```
The initializing cluster centers are: 
[[-6.12152378  2.14971475]
 [ 6.71575768 -5.41421872]
 [-1.30016464 -2.3824513 ]]
iteration 1, current value of J: 12.5459
iteration 2, current value of J: 7.3479
iteration 3, current value of J: 5.2928
iteration 4, current value of J: 5.1493
iteration 5, current value of J: 5.1152
iteration 6, current value of J: 5.1079
iteration 7, current value of J: 5.1065
iteration 8, current value of J: 5.1063
iteration 9, current value of J: 5.1052
iteration 10, current value of J: 5.0970
iteration 11, current value of J: 5.0592
iteration 12, current value of J: 4.9402
iteration 13, current value of J: 4.5036
iteration 14, current value of J: 3.6246
iteration 15, current value of J: 3.2003
iteration 16, current value of J: 3.1678
iteration 17, current value of J: 3.1658
iteration 18, current value of J: 3.1657
iteration 19, current value of J: 3.1657
total iteration num: 19, final value of J: 3.1657, time used: 0.3488 seconds
```

<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/kmeans_result(good_initialization).png" width = "1000" height = "500" alt = "k-means 算法运行结果（好的初始化）" align = center />
</div>

可以看到，这次算法产生的聚类结果比较好；但并不总是这样，例如某次运行该算法产生的聚类结果如下图所示，可以看出，这一次由于初始值的不同，该算法收敛到了一个不好的局部最优解。
<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/kmeans_result(bad_initialization).png" width = "1000" height = "500" alt = "k-means 算法运行结果（不好的初始化）" align = center />
</div>

## 利用 sklearn 实现 k-means 聚类
sklearn 中的 `KMeans` 类可以用来进行 k-means 聚类，sklearn 对该模块进行了计算的优化以及中心点初始化的优化，因而其效果和效率肯定要比上面手写的 k-means 算法要好。在这里，我们直接采用 sklearn 官网的 [demo][8] 来展示 `KMeans` 类的用法，顺便看一下 k-means 算法在破坏了其假设条件的数据集下的运行结果。
代码如下（直接照搬 sklearn 官网）：
```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# 设定一个不合理的 K 值
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# 产生一个非球形分布的数据集
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# 产生一个各 cluster 的密度不一致的数据集
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# 产生一个各 cluster 的样本数目不一致的数据集
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()
```

运行结果如下图所示：
<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/kmeans_different_datasets.png" width = "780" height = "650" alt = "k-means 算法在不同数据集下的表现" align = center />
</div>

上述代码分别产生了四个数据集，并分别对它们进行 k-means 聚类。第一个数据集符合所有 k-means 算法的假设条件，但是我们给定的 K 值与实际数据不符；第二个数据集破坏了球形分布的假设条件；第三个数据集破坏了各 `cluster` 的密度相近的假设条件；第四个数据集则破坏了各 `cluster` 内的样本数目相近的假设条件。可以看到，虽然有一些数据集破坏了 k-means 算法的某些假设条件（密度相近、数目相近），但算法的聚类结果仍然比较好；但如果数据集的分布偏离球形分布太远的话，最终的聚类结果会很差。










[1]:	http://heathcliff.me/%E8%81%9A%E7%B1%BB%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%EF%BC%9A%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/
[2]:	http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf
[3]:	https://www.wikipedia.com/en/Determining_the_number_of_clusters_in_a_data_set
[4]:	https://datasciencelab.wordpress.com/tag/gap-statistic/
[5]:	http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
[6]:	http://www.irma-international.org/viewtitle/10828/
[7]:	https://www.wikipedia.com/en/Occam's_razor
[8]:	http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py