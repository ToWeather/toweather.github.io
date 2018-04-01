---
title: 聚类分析（四）：DBSCAN 算法
date: 2018-03-22 20:18:35
tags:
- 聚类
- 非监督学习
- 密度聚类
categories:
- 机器学习算法
keywords: 聚类,非监督学习,密度聚类,clustering,machine learning

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

*本篇文章为讲述聚类算法的第四篇文章，其它相关文章请参见 [**聚类分析系列文章**][1]*。

# DBSCAN 算法
DBSCAN（Density-Based Spatial Clustering of Applications with Noise) 算法是一种被广泛使用的**密度聚类算法**。密度聚类算法认为各个 `cluster` 是样本点密度高的区域，而 `cluster` 之间是由样本点密度低的区域所隔开的；这种聚类的方式实际上是符合人们的直觉的，这也是以 DBSCAN 算法为代表的密度聚类算法在实际应用中通常表现的比较好的原因之一。

要想描述清楚 DBSCAN 算法，我们得先定义好一些概念，例如何为密度高的区域、何为密度低的区域等等。

## 几个定义
在密度聚类算法中，我们需要区分密度高的区域以及密度低的区域，并以此作为聚类的依据。DBSCAN 通过将样本点划分为三种类型来达到此目的：核心点（core points）、边缘点（border points）以及噪声点（noise），核心点和边缘点共同构成一个一个的 `cluster`，而噪声点则存在于各 `cluster` 之间的区域。为区分清楚这三类样本点，我们首先定义一些基本的概念。

>**定义1：** （样本点的 \\( \\text {Eps} \\)-邻域）假设数据集为 \\( \\bf X \\)，则样本点 \\( \\bf p \\) 的 \\( \\text {Eps} \\)-邻域定义为 \\( N\_{\\text {Eps}} ({\\bf p}) = \\lbrace {\\bf q} \\in {\\bf X} | d({\\bf p}, {\\bf q}) \\le {\\text {Eps}} \rbrace \\). 

我们再给定一个参数 \\( \\text {MinPts} \\)，并定义核心点须满足的条件为：其 \\( \\text {Eps} \\)-邻域内包含的样本点的数目不小于 \\( \\text {MinPts} \\) ， 因而核心点必然在密度高的区域，我们可能很自然地想到一种简单的聚类方法：聚类产生的每一个 `cluster` 中的所有点都是核心点，而所有非核心点都被归为噪声。但仔细地想一想，这种方式其实是不太有道理的，因为一个符合密度高的 `cluster` 的边界处的点并不一定需要是核心点，它可以是非核心点（我们称这类非核心点为边缘点，称其它的非核心点为噪声点），如下图中的图（a）所示。我们通过以下定义来继续刻画 DBSCAN 里面的 `cluster` 所具有的形式。

>**定义2： **（密度直达）我们称样本点 \\( \\bf p \\) 是由样本点 \\( \\bf q \\) 对于参数 \\( \\lbrace \\text {Eps}, \\text {MinPts} \\rbrace \\) 密度直达的，如果它们满足 \\( {\\bf p} \\in N\_{\\text {Eps}} ({\\bf q}) \\) 且 \\( |N\_{\\text {Eps}}({\\bf q})| \\ge \\text {MinPts} \\) （即样本点 \\( \\bf q \\) 是核心点）. 

很显然，密度直达并不是一个对称的性质，如下图中的图（b）所示，其中 \\( \\text {Eps} = 5 \\)，可以看到，样本点 \\( \\bf q \\) 为核心点，样本点 \\( \\bf p \\) 不是核心点，且 \\( \\bf p \\) 在 \\( \\bf q \\) 的 \\( \\text {Eps} \\)-邻域内，因而 \\( \\bf p \\) 可由 \\( \\bf q \\) 密度直达，而反过来则不成立。有了密度直达的定义后，我们再来给出密度可达的定义。

<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/illustration_of_definition_in_dbscan.png" width = "900" height = "600" alt = "DBSCAN 中的相关概念的图示" align = center />
</div>

>**定义3：**（密度可达）我们称样本点 \\( \\bf p \\) 是由样本点 \\( \\bf q \\) 对于参数 \\( \\lbrace \\text {Eps}, \\text {MinPts} \\rbrace \\) 密度可达的，如果存在一系列的样本点 \\( {\\bf p}\_{1}, ..., {\\bf p}\_n \\)（其中 \\( {\\bf p}\_1 = {\\bf q}, {\\bf p}\_n = {\\bf p} \\)）使得对于 \\( i = 1, ..., n-1 \\)，样本点 \\( {\\bf p}\_{i + 1} \\) 可由样本点 \\( {\\bf p}\_{i} \\) 密度可达.

我们可以看到，密度直达是密度可达的特例，同密度直达一样，密度可达同样不是一个对称的性质，如上图中的图（c）所示。为描述清楚我们希望得到的 `cluster` 中的任意两个样本点所需满足的条件，我们最后再给出一个密度相连的概念。

>**定义4：**（密度相连）我们称样本点 \\( \\bf p \\) 与样本点 \\( \\bf q \\) 对于参数 \\( \\lbrace \\text {Eps}, \\text {MinPts} \\rbrace \\) 是密度相连的，如果存在一个样本点 \\( {\\bf o} \\)，使得 \\( \\bf p \\) 和 \\( \\bf q \\) 均由样本点 \\( \\bf o \\) 密度可达。

密度相连是一个对称的性质，如上图中的图（d）所示。DBSCAN 算法期望找出一些 `cluster` ，使得每一个 `cluster` 中的任意两个样本点都是密度相连的，且每一个 `cluster` 在密度可达的意义上都是最大的。`cluster` 的定义如下：

>**定义5：**（`cluster`）假设数据集为 \\( \\bf X \\)，给定参数 \\( \\lbrace \\text {Eps}, \\text {MinPts} \\rbrace \\)，则某个 `cluster` \\( C \\) 是数据集  \\( \\bf X \\) 的一个非空子集，且满足如下条件：
		>1）对于任意的样本点 \\( \\bf p \\) 和 \\( \\bf q \\)，如果 \\( {\\bf p} \\in C \\) 且 \\( \\bf q \\) 可由 \\( \\bf p \\) 密度可达，则 \\( {\\bf q} \\in C \\) .（最大性）
		>2）对于 \\( C \\) 中的任意样本点 \\( \\bf p \\) 和 \\( \\bf q \\)， \\( \\bf p \\) 和 \\( \\bf q \\) 关于参数 \\( \\lbrace \\text {Eps}, \\text {MinPts} \\rbrace \\) 是密度相连的.（连接性）

这样我们就定义出了 DBSCAN 算法最终产生的 `cluster` 的形式，它们就是所谓的密度高的区域；那么，噪声点就是不属于任何 `cluster` 的样本点。根据以上定义，由于一个 `cluster` 中的任意两个样本点都是密度相连的，每一个 `cluster` 至少包含 \\( \\text {MinPts} \\) 个样本点。 

## 算法描述
DBSCAN 算法就是为了寻找以上定义 5 中定义的 `cluster`，其主要思路是先指定一个核心点作为种子，寻找所有由该点密度可达的样本点组成一个 `cluster`，再从未被聚类的核心点中指定一个作为种子，寻找所有由该点密度可达的样本点组成第二个 `cluster`，…，依此过程，直至没有未被聚类的核心点为止。依照 `cluster` 的“最大性”和“连接性”，在给定参数 \\( \\lbrace \\text {Eps}, \\text {MinPts} \\rbrace \\) 的情况下，最终产生的 `cluster` 的结果是一致的，与种子的选取顺序无关（某些样本点位于多个 `cluster` 的边缘的情况除外，这种情况下，这些临界位置的样本点的 `cluster` 归属与种子的选取顺序有关）。

## 合理地选取参数
DBSCAN 的聚类结果和效果取决于参数 \\( \\text {Eps} \\) 和 \\( \\text {MinPts} \\) 以及距离衡量方法的选取。

由于算法中涉及到寻找某个样本点的指定邻域内的样本点，因而需要衡量两个样本点之间的距离。这里选取合适的距离衡量函数也变得比较关键，一般而言，我们需要根据所要处理的数据的特点来选取距离函数，例如，当处理的数据是空间地理位置信息时，Euclidean 距离是一个比较合适的选择。各种不同的距离函数可参见 [*聚类分析（一）：层次聚类算法*][2]。

然后我们再来看参数的选取方法，我们一般选取数据集中最稀疏的 `cluster` 所对应的 \\( \\text {Eps} \\) 和 \\( \\text {MinPts} \\)。这里我们给出一种启发式的参数选取方法。假设 \\( d \\) 是某个样本点 \\( \\bf p \\) 距离它的第 \\( k \\) 近邻的距离，则一般情况下 \\( \\bf p \\) 的 \\( d \\)-邻域内正好包含 \\( k + 1 \\) 个样本点。我们可以推断，在一个合理的 `cluster` 内，改变 \\( k \\) 的值不应该导致 \\( d \\) 值有较大的变化，除非 \\( \\bf p \\) 的第 \\( k \\) 近邻们（\\( k = 1, 2, 3,… \\) ） 都近似在一条直线上，而这种情形的数据不可能构成一个合理的 `cluster`。

因而我们一般将 \\( k \\) 的值固定下来，一个合理的选择是令 \\( k = 3 \\) 或 \\( k = 4 \\)，那么 \\( \\text {MinPts} \\) 的值也确定了（为 \\( k + 1 \\)）；然后再来看每个样本点的 \\( \\text {k-dist} \\) 距离（即该样本点距离它的第 \\( k \\) 近邻的距离）的分布情况，我们把每个样本点的 \\( \\text {k-dist} \\) 距离从大到小排列，并将它绘制出来，如下图所示。我们再来根据这个图像来选择 \\( \\text {Eps} \\)，一种情况是我们对数据集有一定的了解，知道噪声点占总样本数的大致比例，则直接从图像中选取该比例处所对应的样本点的 \\( \\text {k-dist} \\) 距离作为 \\( \\text {Eps} \\)，如下图中的图（a）所示；另一种情况是，我们对数据集不太了解，此时就只能分析 \\( \\text {k-dist} \\) 图了，一般情况下，我们选取第一个斜率变化明显的“折点”所对应的 \\( \\text {k-dist} \\) 距离作为 \\( \\text {k-dist} \\) ，如下图中的图（b）所示。

<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/illustration_of_parameter_selection_for_dbscan.png" width = "900" height = "500" alt = "DBSCAN 中数据集的 k-dist 图" align = center />
</div>

## 算法的复杂度及其优缺点
### 算法复杂度
DBSCAN 算法的主要计算开销在于对数据集中的每一个样本点都判断一次其是否为核心点，而进行这一步骤的关键是求该样本点的 \\( \\text {Eps} \\)-邻域内的样本点，如果采用穷举的遍历的方法的话，该操作的时间复杂度为 \\( O(N) \\)，其中 \\( N \\) 为总样本数；但我们一般会对数据集建立索引，以此来加快查询某邻域内数据的速度，例如，当采用 [R\* tree][3] 建立索引时，查询邻域的平均时间复杂度为 \\( O(\\log N) \\)。因而，DBSCAN 算法的平均时间复杂度为 \\( O(N\\log N) \\)；由于只需要存储数据集以及索引信息，DBSCAN算法的空间复杂度与总样本数在一个量级，即 \\( O(N) \\)。

### 优缺点
DBSCAN 算法有很多优点，总结如下：
- DBSCAN 不需要事先指定最终需要生成的 `cluster` 的数目，这一点解决了其它聚类算法（如 k-means、GMM 聚类等）在实际应用中的一个悖论：我们由于对数据集的结构不了解才要进行聚类，如果我们事先知道了数据集中的 `cluster` 的数目，实际上我们对数据集是有相当多的了解的。
- DBSCAN 可以找到具有任意形状的 `cluster`，如非凸的 `cluster`，这基于其对 `cluster` 的定义（`cluster` 是由密度低的区域所隔开的密度高的区域）。
- DBSCAN 对异常值鲁棒，因为它定义了噪声点的概念。
- DBSCAN 算法在给定参数下对同一数据集的聚类结果是确定的（除非出现某些样本点位于多个 `cluster` 的边缘的情况，这种情况下，这些临界位置的样本点的 `cluster` 归属与种子的选取顺序有关，而它们对于聚类结果的影响也是微乎其微的）。
- DBSCAN 的运行速度快，当采用索引时，其复杂度仅为  \\( O(N\\log N) \\)。

当然，它也有一个主要缺点，即对于具有密度相差较大的 `cluster` 的数据集的聚类效果不好，这是因为在这种情况下，如果参数 \\( \\text {MinPts} \\) 和 \\( \\text {Eps} \\) 是参照数据集中最稀疏的 `cluster` 所选取的，那么很有可能最终所有的样本最终都被归为一个 `cluster`，因为可能数据集中的 `cluster` 之间的区域的密度和最稀疏的 `cluster` 的密度相当；如果选取的参数 \\( \\text {MinPts} \\) 和 \\( \\text {Eps} \\) 倾向于聚出密度比较大的 `cluster`，那么极有可能，比较稀疏的这些 `cluster` 都被归为噪声。[ OPTICS][4] 算法一般被用来解决这一问题。

---- 
# 实现 DBSCAN 聚类
现在我们来实现 DBSCAN 算法。首先我们定义一个用于进行 DBSCAN 聚类的类 `DBSCAN`，进行聚类时，我们得先构造一个该类的实例，初始化时，我们须指定 DBSCAN 算法的参数 `min_pts` 和 `eps` 以及距离衡量方法（默认为 `euclidean`），对数据集进行聚类时，我们对构造出来的实例调用方法 `predict`。`predict` 方法首先为“查询某样本点的邻域”这一经常被用到的操作做了一些准备工作，一般是计算各样本点间的距离并保存，但在当数据维度为 2、距离度量方法为 Euclidean 距离时，我们使用 `rtree` 模块为数据集建立了空间索引，以加速查询邻域的速度（一般来讲，为提高效率，应该对任意数据维度、任意距离度量方法下的数据集建立索引，但这里为实现方便，仅对此一种类型的数据集建立了索引，其它类型的数据集我们还是通过遍历距离矩阵的形式进行低效的邻域查找）。然后再对数据进行 DBSCAN 聚类，即遍历数据集中的样本点，若该样本点未被聚类且为核心点时，以该样本点为种子按照密度可达的方式扩展至最大为止。代码如下：
```python
import numpy as np
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rtree

UNCLASSIFIED = -2
NOISE = -1


class DBSCAN():
    def __init__(self, min_pts, eps, metric='euclidean', index_flag=True):
        self.min_pts = min_pts
        self.eps = eps
        self.metric = metric
        self.index_flag = index_flag
        self.data_set = None
        self.pred_label = None
        self.core_points = set()

    def predict(self, data_set):
        self.data_set = data_set
        self.n_samples, self.n_features = self.data_set.shape

        self.data_index = None
        self.dist_matrix = None

        start_time = time.time()
        if self.n_features == 2 and self.metric == 'euclidean' \
            and self.index_flag:
            # 此种情形下对数据集建立空间索引
            self.construct_index()
        else:
            # 其它情形下对数据集计算距离矩阵
            self.cal_dist_matrix()

        self.pred_label = np.array([UNCLASSIFIED] * self.n_samples)

        # 开始 DBSCAN 聚类
        crt_cluster_label = -1
        for i in range(self.n_samples):
            if self.pred_label[i] == UNCLASSIFIED:
                query_result = self.query_eps_region_data(i)
                if len(query_result) < self.min_pts:
                    self.pred_label[i] = NOISE
                else:
                    crt_cluster_label += 1
                    self.core_points.add(i)
                    for j in query_result:
                        self.pred_label[j] = crt_cluster_label
                    query_result.discard(i)
                    self.generate_cluster_by_seed(query_result, crt_cluster_label)
        print("time used: %.4f seconds" % (time.time() - start_time))

    def construct_index(self):
        self.data_index = rtree.index.Index()
        for i in range(self.n_samples):
            data = self.data_set[i]
            self.data_index.insert(i, (data[0], data[1], data[0], data[1]))

    @staticmethod
    def distance(data1, data2, metric='euclidean'):
        if metric == 'euclidean':
            dist = np.sqrt(np.dot(data1 - data2, data1 - data2))
        elif metric == 'manhattan':
            dist = np.sum(np.abs(data1 - data2))
        elif metric == 'chebyshev':
            dist = np.max(np.abs(data1 - data2))
        else:
            raise Exception("invalid or unsupported distance metric!")
        return dist

    def cal_dist_matrix(self):
        self.dist_matrix = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                dist = self.distance(self.data_set[i], self.data_set[j], self.metric)
                self.dist_matrix[i, j], self.dist_matrix[j, i] = dist, dist

    def query_eps_region_data(self, i):
        if self.data_index:
            data = self.data_set[i]
            query_result = set()
            buff_polygon = Point(data[0], data[1]).buffer(self.eps)
            xmin, ymin, xmax, ymax = buff_polygon.bounds
            for idx in self.data_index.intersection((xmin, ymin, xmax, ymax)):
                if Point(self.data_set[idx][0], self.data_set[idx][1]).intersects(buff_polygon):
                    query_result.add(idx)
        else:
            query_result = set(item[0] for item in np.argwhere(self.dist_matrix[i] <= self.eps))
        return query_result

    def generate_cluster_by_seed(self, seed_set, cluster_label):
        while seed_set:
            crt_data_index = seed_set.pop()
            crt_query_result = self.query_eps_region_data(crt_data_index)
            if len(crt_query_result) >= self.min_pts:
                self.core_points.add(crt_data_index)
                for i in crt_query_result:
                    if self.pred_label[i] == UNCLASSIFIED:
                        seed_set.add(i)
                    self.pred_label[i] = cluster_label
```

我们下面构造一个数据集，并对该数据集运行我们手写的算法进行 DBSCAN 聚类，并将 DBSCAN 中定义的核心点、边缘点以及噪声点可视化；同时，我们还想验证一下我们手写的算法的正确性，所以我们利用 `sklearn` 中实现的 `DBSCAN` 类对同一份数据集进行了 DBSCAN 聚类，代码如下：
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN as DBSCAN_SKLEARN

def plot_clustering(X, y, core_pts_idx=None, title=None):
    if core_pts_idx is not None:
        core_pts_idx = np.array(list(core_pts_idx), dtype=int)
        core_sample_mask = np.zeros_like(y, dtype=bool)
        core_sample_mask[core_pts_idx] = True

        unique_labels = set(y)
        colors = [plt.cm.Spectral(item) for item in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            class_member_mask = (y == k)
            xy = X[class_member_mask & core_sample_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=12, alpha=0.6)
            xy = X[class_member_mask & ~core_sample_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6, alpha=0.6)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6)
    if title is not None:
        plt.title(title, size=14)
    plt.axis('on')
    plt.tight_layout()

# 构造数据集
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# 利用我自己手写的 DBSCAN 算法对数据集进行聚类
dbscan_diy = DBSCAN(min_pts=20, eps=0.5)
dbscan_diy.predict(X)
n_clusters = len(set(dbscan_diy.pred_label)) - (1 if -1 in dbscan_diy.pred_label else 0)
print("count of clusters generated: %s" % n_clusters)
print("propotion of noise data for dbscan_diy: %.4f" % (np.sum(dbscan_diy.pred_label == -1) / n_samples))
plt.subplot(1, 2, 1)
plot_clustering(X, dbscan_diy.pred_label, dbscan_diy.core_points,
                title="DBSCAN(DIY) Results")

# 利用 sklearn 实现的 DBSCAN 算法对数据集进行聚类
dbscan_sklearn = DBSCAN_SKLEARN(min_samples=20, eps=0.5)
dbscan_sklearn.fit(X)
print("propotion of noise data for dbscan_sklearn: %.4f" % (np.sum(dbscan_sklearn.labels_ == -1) / n_samples))
plt.subplot(1, 2, 2)
plot_clustering(X, dbscan_sklearn.labels_, dbscan_sklearn.core_sample_indices_,
                title="DBSCAN(SKLEARN) Results")

plt.show()
```


运行得到的输出和可视化结果如下所示：
```
time used: 4.2602 seconds
count of clusters generated: 3
propotion of noise data for dbscan_diy: 0.1220
propotion of noise data for dbscan_sklearn: 0.1220
```

<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/dbscan_clustering_results.png" width = "1000" height = "500" alt = "DBSCAN 的运行结果" align = center />
</div>

上图中，大圆圈表示核心点、非黑色的小圆圈表示边缘点、黑色的小圆圈表示噪声点，它们的分布服从我们的直观感受。我们还可以看到，手写算法和 sklearn 实现的算法的产出时一模一样的（从可视化结果的对比以及噪声点所占的比例可以得出），这也验证了手写算法的正确性。

我们再来看一下 DBSCAN 算法对于非凸数据集的聚类效果，代码如下：
```python
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler

n_samples = 1500
noisy_circles, _ = make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_circles = StandardScaler().fit_transform(noisy_circles)
noisy_moons, _ = make_moons(n_samples=n_samples, noise=.05)
noisy_moons = StandardScaler().fit_transform(noisy_moons)
dbscan = DBSCAN(min_pts=5, eps=0.22)
dbscan.predict(noisy_circles)
plt.subplot(1, 2, 1)
plot_clustering(noisy_circles, dbscan.pred_label, title="Concentric Circles Dataset")

dbscan.predict(noisy_moons)
plt.subplot(1, 2, 2)
plot_clustering(noisy_moons, dbscan.pred_label, title="Interleaved Moons DataSet")

plt.show()
```


运行的结果如下图所示：

<div align = center>
<img src="https://raw.githubusercontent.com/ToWeather/MarkdownPhotos/master/dbscan_clustering_result_for_nonconvex_datasets.png" width = "1000" height = "500" alt = "DBSCAN 在非凸数据集下的运行结果" align = center />
</div>

可以看到 DBSCAN 算法对非凸数据集的聚类效果非常好。


 



[1]:	../clustering-analysis/index.html
[2]:	../%E8%81%9A%E7%B1%BB%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%EF%BC%9A%E5%B1%82%E6%AC%A1%E8%81%9A%E7%B1%BB%E7%AE%97%E6%B3%95/
[3]:	https://en.wikipedia.org/wiki/R*_tree?oldformat=true
[4]:	http://suraj.lums.edu.pk/~cs536a04/handouts/OPTICS.pdf