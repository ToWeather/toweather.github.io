---
title: 聚类分析（三）：高斯混合模型与 EM 算法
date: 2018-03-12 21:26:08
tags:
- 聚类
- 非监督学习
- 高斯混合模型
- GMM
- 生成模型
- EM 算法
categories:
- 机器学习算法
keywords: 聚类,非监督学习,高斯混合模型,GMM,生成模型,EM 算法,clustering,machine learning,最大似然估计

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

*本篇文章为讲述聚类算法的第三篇文章，其它相关文章请参见 [**聚类分析系列文章**][1]*。

# 高斯混合模型
## 高斯混合模型简介
高斯混合模型（Gaussian Mixture Model，GMM）即是几个高斯分布的线性叠加，其可提供更加复杂但同时又便于分析的概率密度函数来对现实世界中的数据进行建模。具体地，高斯混合分布的概率密度函数如下所示：
$$
p({\bf x}) = \sum_{k = 1}^{K} \pi_{k} {\cal N}({\bf x}|{\bf \mu}_k, {\bf \Sigma}_k) 
\tag {1}
$$
如果用该分布随机地产生样本数据，则每产生一个样本数据的操作如下：首先在 $ K $ 个类别中以 $ \pi_k $ 的概率随机选择一个类别 $ k $，然后再依照该类别所对应的高斯分布 $ {\cal N}({\bf x}|{\bf \mu}_k, {\bf \Sigma}_k) $ 随机产生一个数据 $ {\bf x} $。但最终生成数据集后，我们所观察到的仅仅只是 $ {\bf x} $，而观察不到用于产生 $ {\bf x} $ 的类别信息。我们称这些观察不到的变量为隐变量（latent variables）。为描述方便，我们以随机向量 $ {\bf z} \in {\lbrace 0, 1 \rbrace}^{K} $ 来表示高斯混合模型中的类别变量，$ {\bf z} $ 中仅有一个元素的值为 $ 1 $，而其它元素的值为 $ 0 $，例如当 $ z_{k} = 1$ 时，表示当前数据是由高斯混合分布中的第 $ k $ 个类别所产生。 对应于上面高斯混合分布的概率密度函数，隐变量 $ {\bf z} $ 的概率质量函数为
$$
p(z_{k} = 1) = \pi_k 
\tag {2}
$$
其中 $ \lbrace \pi_{k} \rbrace $ 须满足
$$
\sum_{k = 1}^{K} \pi_k = 1  \text{,} \ \ \ 0 \le \pi_k \le 1
$$
给定 $ {\bf z} $ 的值的情况下，$ {\bf x} $ 服从高斯分布
$$
p({\bf x} | z_{k} = 1) = {\cal N}({\bf x} | {\bf \mu}_{k}, {\bf \Sigma}_{k}) 
\tag {3}
$$
因而可以得到 $ {\bf x} $ 的边缘概率分布为
$$
p({\bf x}) = \sum_{\bf z} p({\bf z})p({\bf x} | {\bf z}) = \sum_{k =1}^{K} \pi_k {\cal N}({\bf x} | {\bf \mu}_{k}, {\bf \Sigma}_{k}) 
\tag {4}
$$
该分布就是我们前面所看到的高斯混合分布。
## 最大似然估计问题
假设有数据集 $ \lbrace {\bf x}_1, {\bf x}_2, …, {\bf x}_N \rbrace $，其中样本的维度为 $ D $，我们希望用高斯混合模型来对这个数据集来建模。为分析方便，我们可以以矩阵 $ {\bf X} = [ {\bf x}_1, {\bf x}_2, …, {\bf x}_N ]^{T} \in {\Bbb R}^{N \times D} $ 来表示数据集，以矩阵 $ {\bf Z} = [ {\bf z}_1, {\bf z}_2, …, {\bf z}_N ]^{T} \in {\Bbb R}^{N \times K} $ 来表示各个样本对应的隐变量。假设每个样本均是由某高斯混合分布独立同分布（i.i.d.）地产生的，则可以写出该数据集的对数似然函数为
$$
L({\bf \pi}, {\bf \mu}, {\bf \Sigma}) = \ln p({\bf X} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) = \sum_{n = 1}^{N} \ln \lbrace  \sum_{k = 1}^{K} \pi_k {\cal N}({\bf x}_{n} | {\bf \mu}_{k}, {\bf \Sigma}_{k}) \rbrace 
\tag {5}
$$
我们希望求解出使得以上对数似然函数最大的参数集 $ \lbrace {\bf \pi}, {\bf \mu}, {\bf \Sigma} \rbrace $，从而我们就可以依照求解出的高斯混合模型来对数据集进行分析和阐释，例如对数据集进行聚类分析。但该似然函数的对数符号里面出现了求和项，这就使得对数运算不能直接作用于单个高斯概率密度函数（高斯分布属于指数分布族，而对数运算作用于指数分布族的概率密度函数可以抵消掉指数符号，使得问题的求解变得非常简单），从而使得直接求解该最大似然估计问题变得十分复杂。事实上，该问题也没有闭式解。

 另一个值得注意的地方是，在求解高斯混合模型的最大似然估计问题时，可能会得到一个使 $ L({\bf \pi}, {\bf \mu}, {\bf \Sigma}) $ 发散的解。具体地，观察公式 $ (5) $，如果某一个高斯分量的期望向量正好取到了某一个样本点的值，则当该高斯分量的协方差矩阵为奇异矩阵（行列式为 $ 0 $）的时候，会导致求和项中的该项的值为无穷大，从而也使得 $ L({\bf \pi}, {\bf \mu}, {\bf \Sigma}) $ 的值为无穷大；这样确实使得式 $ (5) $ 最大了，但求解出的高斯混合模型中的某一个高斯分量却直接坍缩到了某一个样本点，这显然是与我们的初衷不相符合的。这也体现了最大似然估计方法容易导致过拟合现象的特点，如果我们采用最大后验概率估计的方法，则可避免出现该问题；此外，如果我们采用迭代算法来求解该最大似然估计问题（如梯度下降法或之后要讲到的 EM 算法）的话，可以在迭代的过程中用一些启发式的方法加以干预来避免出现高斯分量坍缩的问题（将趋于坍缩的高斯分量的期望设置为不与任何样本点相同的值，将其协方差矩阵设置为一个具有较大的行列式值的矩阵）。

----
# EM 算法求解高斯混合模型
EM （Expectation-Maximization）算法是一类非常优秀且强大的用于求解含隐变量的模型的最大似然参数估计问题的算法。我们将在这一部分启发式地推导出用于求解高斯混合模型的最大似然参数估计问题的 EM 算法。

我们对对数似然函数 $ (5) $ 关于各参数 $ {\bf \pi} $， $ {\bf \mu} $， $ {\bf \Sigma} $ 分别求偏导，并将其置为 $ 0 $ 可以得到一系列的方程，而使得式 $ (5) $ 最大的解也一定满足这些方程。

首先令式 $ (5) $ 关于 $ {\bf \mu}_k $ 的偏导为 $ 0 $ 可得以下方程：
$$
\sum_{n = 1}^{N} \frac{\pi_k {\cal N}({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k)} { \sum_{j} \pi_j {\cal N} ({\bf x}_n | {\bf \mu}_j, {\bf \Sigma}_j)} {\bf \Sigma}_k ({\bf x}_n - {\bf \mu}_k) = 0
\tag {6}
$$
注意到，上式中含有项
$$
\gamma (z_{nk}) =   \frac{\pi_k {\cal N}({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k)} { \sum_{j} \pi_j {\cal N} ({\bf x}_n | {\bf \mu}_j, {\bf \Sigma}_j)} = p(z_{nk} = 1 | {\bf x}_n) 
\tag {7}
$$
该项具有重要的物理意义，它为给定样本点 $ {\bf x}_n $ 后隐变量 $ {\bf z}_n $ 的后验概率，可直观理解某个高斯分量对产生该样本点所负有的“责任”（resposibility）；GMM 聚类就是利用 $ \gamma (z_{nk}) $ 的值来做软分配的。

因而，我们可以由式 $ (6) $ 和式 $ (7) $ 写出
$$
{\bf \mu}_k = \frac{1} {N_k} \sum_{n = 1}^{N} \gamma(z_{nk}) {\bf x}_n
\tag {8}
$$
其中 $ N_k = \sum_{n = 1}^{N} \gamma(z_{nk}) $，我们可以将 $ N_{k} $ 解释为被分配给类别 $ k $ 的有效样本数量，而 $ {\bf \mu}_{k} $ 即为所有样本点的加权算术平均值，每个样本点的权重等于第 $ k$ 个高斯分量对产生该样本点所负有的“责任”。

我们再将式 $ (5) $ 对 $ {\bf \Sigma}_k $ 的偏导数置为 $ 0 $ 可求得
$$
{\bf \Sigma}_k = \frac{1} {N_k} \sum_{n = 1}^{N} \gamma(z_{nk}) ({\bf x}_n - {\bf \mu}_k)({\bf x}_n - {\bf \mu}_k)^{\text T}
\tag {9}
$$
可以看到上式和单个高斯分布的最大似然参数估计问题求出来的协方差矩阵的解的形式是一样的，只是关于每个样本点做了加权，而加权值仍然是 $ \gamma(z_{nk}) $。

最后我们再来推导 $ \pi_k $ 的最大似然解须满足的条件。由于 $ \pi_k $ 有归一化的约束，我们可以利用 [ Lagrange 乘数法 ][2] 来求解（将 $ \ln p({\bf X} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) + \lambda (\sum_{k = 1}^{K} \pi_k - 1) $ 关于 $ \pi_k $ 的偏导数置 $ 0 $），最后可求得
$$
\pi_k = \frac{N_k} {N}
\tag {10}
$$
关于类别 $ k $ 的先验概率的估计值可以理解为所有样本点中被分配给第 $ k $ 个类别的有效样本点的个数占总样本数量的比例。

注意到，式 $ (8) $ 至 $ (10) $ 即是高斯混合模型的最大似然参数估计问题的解所需满足的条件，但它们并不是 $ \lbrace {\bf \pi}, {\bf \mu}, {\bf \Sigma} \rbrace $ 的闭式解，因为这些式子中给出的表达式中都含有 $ \gamma (z\_{nk}) $，而 $ \gamma (z\_{nk}) $ 反过来又以一种非常复杂的方式依赖于所要求解的参数（见式 $ (7) $）。

尽管如此，以上的这些公式仍然提供了一种用迭代的方式来解决最大似然估计问题的思路。 先将参数 $ \lbrace {\bf \pi}, {\bf \mu}, {\bf \Sigma} \rbrace $ 固定为一个初始值，再按式 $ (7) $ 计算出隐变量的后验概率 $ \gamma (z_{nk}) $ （E 步）；然后再固定 $ \gamma (z_{nk}) $，按式 $ (8) $ 至 $ (10) $ 分别更新参数  $ \lbrace {\bf \pi}, {\bf \mu}, {\bf \Sigma} \rbrace $ 的值（M 步），依次交替迭代，直至似然函数收敛或所求解的参数收敛为止。

实际上，上述描述即是用于求解高斯混合模型的最大似然参数估计问题的 EM 算法，该算法可以求得一个局部最优解，因为其每一步迭代都会增加似然函数的值（稍后讲述一般 EM 算法的时候会证明）。

我们仍然借用 PRML 教材中的例子来阐述 EM 算法在求解上述问题时的迭代过程。下图中的数据集依旧来源于经过标准化处理的 Old Faithful 数据集，我们选用含有两个高斯分量的高斯混合模型，下图中的圆圈或椭圆圈代表单个高斯分量（圆圈代表该高斯分量的概率密度函数的偏离期望一个标准差处的等高线），以蓝色和红色来区分不同的高斯分量；图（a）为各参数的初始化的可视化呈现，在这里我们为两个高斯分量选取了相同的协方差矩阵，且该协方差矩阵正比于单位矩阵；图（b）为 EM 算法的 E 步，即更新后验概率  $ \gamma (z_{nk}) $，在图中则体现为将每一个样本点染上有可能产生它的高斯分量的颜色（例如，某一个样本点的由蓝色的高斯分量产生的概率为 $ p_1 $ ，由红色的高斯分量产生的概率为 $ p_2 $，则我们将其染上 $ p_1 $ 比例的蓝色，染上  $ p_2 $ 比例的红色）；图（c）为 EM 算法的 M 步，即更新 GMM 模型的各参数，在图中表现为椭圆圈的变化；图 （d）～（f）分别是后续的迭代步骤，到第 20 次迭代的时候，算法已经基本上收敛。
<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/95119/36/19198/564054/5e9d2234E755f59ca/df0c82b898c6b3e1.png" width = "660" height = "550" alt = "EM 算法求解 GMM 运行过程" align = center />
</div>
----
# 一般 EM 算法

前面我们只是启发式地推导了用于求解 GMM 的最大似然估计问题的 EM 算法，但这仅仅只是 EM 算法的一个特例。实际上，EM 算法可用于求解各种含有隐变量的模型的最大似然参数估计问题或者最大后验概率参数估计问题。在这里，我们将以一种更正式的形式来推导这类适用范围广泛的 EM 算法。

假设观测到的变量集为 $ {\bf X} $，隐变量集为 $ {\bf Z} $，模型中所涉及到的参数集为 $ {\bf \theta} $，我们的目的是最大化关于 $ {\bf X} $ 的似然函数
$$
p({\bf X} | {\bf \theta}) = \sum_{\bf Z} p({\bf X}, {\bf Z} | {\bf \theta})
\tag {11}
$$
一般来讲，直接优化 $ p({\bf X} | {\bf \theta}) $ 是比较困难的，而优化完全数据集的似然函数 $ p({\bf X}, {\bf Z} | {\bf \theta}) $ 的难度则会大大减小，EM 算法就是基于这样的思路。

首先我们引入一个关于隐变量 $ {\bf Z} $ 的分布 $ q({\bf Z}) $，然后我们可以将对数似然函数 $ \ln p({\bf X} | {\bf \theta}) $ 分解为如下
$$
\ln p({\bf X} | {\bf \theta}) = {\cal L}(q, {\bf \theta}) + {\text KL}(q || p) 
\tag {12}
$$
其中
$$
{\cal L}(q, {\bf \theta}) = \sum_{\bf Z} q({\bf Z}) \ln \lbrace \frac{p({\bf X}, {\bf Z} | {\bf \theta})} {q({\bf Z})} \rbrace
\tag {13}
$$
$$
{\text KL}(q || p) = - \sum_{\bf Z} q({\bf Z}) \ln \lbrace \frac{ p({\bf Z} | {\bf X}, {\bf \theta})} {q({\bf Z})} \rbrace
\tag {14}
$$
其中 $ {\cal L}(q, {\bf \theta}) $ 为关于概率分布 $ q({\bf Z}) $ 的泛函，且为关于参数集 $ {\bf \theta} $ 的函数，另外，$ {\cal L}(q, {\bf \theta}) $ 的表达式中包含了关于完全数据集的似然函数 $ p({\bf X}, {\bf Z} | {\bf \theta}) $，这是我们需要好好加以利用的；$ {\text KL}(q || p) $ 为概率分布 $ q({\bf Z}) $ 与隐变量的后验概率分布 $ p({\bf Z} | {\bf X}, {\bf \theta}) $ 间的 [KL 散度][3]，它的值一般大于 $ 0 $，只有在两个概率分布完全相同的情况下才等于 $ 0 $，因而其一般被用来衡量两个概率分布之间的差异。

利用 $ {\text KL}(q || p) \ge 0 $ 的性质，我们可以得到 $ {\cal L}(q, {\bf \theta}) \le \ln p({\bf X} | {\bf \theta}) $，即 $ {\cal L}(q, {\bf \theta}) $ 是对数似然函数 $ \ln p({\bf X} | {\bf \theta}) $ 的一个下界。 $ \ln p({\bf X} | {\bf \theta}) $ 与 $ {\cal L}(q, {\bf \theta}) $  及 $ {\text KL}(q || p) $ 的关系可用下图中的图（a）来表示。 

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/95615/21/19360/177497/5e9d2261E6567325f/4847b152a367efac.png" width = "1000" height = "600" alt = "EM 算法的推导过程示意图" align = center />
</div>


有了以上的分解之后，下面我们来推导 EM 算法的迭代过程。

假设当前迭代步骤的参数的值为 $ {\bf \theta}^{\text {old}} $，我们先固定 $ {\bf \theta}^{\text {old}} $ 的值，来求 $ {\cal L}(q, {\bf \theta}^{\text {old}}) $ 关于概率分布 $ q({\bf Z}) $ 的最大值。可以看到，$ \ln p({\bf X} | {\bf \theta} ^{\text {old}}) $ 现在是一个定值，所以当 $ {\text KL}(q || p) $ 等于 $ 0 $ 时， $ {\cal L}(q, {\bf \theta}^{\text {old}}) $ 最大，如上图中的图（b）所示。此时由 $ {\text KL}(q || p) = 0 $ 可以推出，$ q({\bf Z}) = p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old}}) $。

现在再固定 $ q({\bf Z}) $，来求 $ {\cal L}(q, {\bf \theta}) $ 关于 $ {\bf \theta} $ 的最大值，假设求得的最佳 $ {\bf \theta} $ 的值为 $ {\bf \theta} ^{\text {new}} $，此时 $ {\cal L}(q, {\bf \theta} ^{\text {new}}) $ 相比 $ {\cal L}(q, {\bf \theta}^{\text {old}}) $ 的值增大了，而由于 $ {\bf \theta} $ 值的改变又使得当前 $ {\text KL}(q || p) $ 的值大于或等于 $ 0 $ （当算法收敛时保持 $ 0 $ 的值不变），所以根据式 $ (14) $，对数似然函数 $ \ln p({\bf X} | {\bf \theta}) $ 的值在本次迭代过程中肯定会有所增长（当算法收敛时保持不变），此步迭代过程如上图中的图（c）所示。

更具体地来说，以上的第一个迭代步骤被称之为 E 步（Expectation），即求解隐变量的后验概率函数 $ p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old}}) $，我们将 $ q({\bf Z}) = p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old}}) $ 带入  $ {\cal L}(q, {\bf \theta}) $  中，可得
$$
\begin{aligned}
{\cal L}(q, {\bf \theta}) &= \sum_{\bf Z} p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old}}) \ln p({\bf X}, {\bf Z} | {\bf \theta}) - \sum_{\bf Z} p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old} }) \ln p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old} }) \\
& = {\cal Q} ({\bf \theta}, {\bf \theta}^{\text {old}}) + {\text {const} }
\end{aligned}
\tag {15}
$$
我们只对上式中的第一项 $ {\cal Q} ({\bf \theta}, {\bf \theta}^{\text {old}}) $ 感兴趣（第二项为常数），可以看到它是完全数据集的对数似然函数的关于隐变量的后验概率分布的期望值，这也是 EM 算法中 “E” （Expectation）的来源。

第二个迭代步骤被称为 M 步（Maximization），是因为要对式 $ (15) $ 中的 $ {\cal Q} ({\bf \theta}, {\bf \theta}^{\text {old}}) $ 求解关于  $ {\bf \theta} $ 的最大值，由于 $ {\cal Q} ({\bf \theta}, {\bf \theta}^{\text {old}}) $ 的形式相较于对数似然函数  $ \ln p({\bf X} | {\bf \theta}) $ 来说简化了很多，因而对其求解最大值也是非常方便的，且一般都存在闭式解。

最后还是来总结一下 EM 算法的运行过程：
1. 选择一个初始参数集 $ {\bf \theta}^{\text {old}} $；
2. **E  步**，计算隐变量的后验概率函数 $ p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old}}) $；
3. **M 步**，按下式计算 $ {\bf \theta}^{\text {new}} $ 的值
$$
{\bf \theta}^{\text {new}} = \rm {arg}  \rm {max}_{\bf \theta} {\cal Q} ({\bf \theta}, {\bf \theta}^{\text {old} })
\tag {16}
$$
其中 
$$
{\cal Q} ({\bf \theta}, {\bf \theta}^{\text {old}}) = \sum_{\bf Z} p({\bf Z} | {\bf X}, {\bf \theta}^{\text {old}}) \ln p({\bf X}, {\bf Z} | {\bf \theta})
\tag {17}
$$
4. 检查是否满足收敛条件（如前后两次迭代后对数似然函数的差值小于一个阈值），若满足，则结束迭代；若不满足，则令 $ {\bf \theta}^{\text {old}} = {\bf \theta}^{\text {new}} $ ，回到第 2 步继续迭代。

----
# 再探高斯混合模型
在给出了适用于一般模型的 EM 算法之后，我们再来从一般 EM 算法的迭代步骤推导出适用于高斯混合模型的 EM 算法的迭代步骤（式 $ (7) $ 至 $ (10) $ ）。

## 推导过程
在高斯混合模型中，参数集 $ {\bf \theta} = \lbrace {\bf \pi}, {\bf \mu}, {\bf \Sigma} \rbrace $，完整数据集 $ \lbrace {\bf X}, {\bf Z} \rbrace $ 的似然函数为
$$
p({\bf X}, {\bf Z} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) = \prod_{n = 1}^{N} \prod_{k = 1}^{K} {\pi_k}^{z_{nk}} {\cal N} ({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k)^{z_{nk}}
\tag {18}
$$
对其取对数可得
$$
\ln p({\bf X}, {\bf Z} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) = \sum_{n = 1}^{N} \sum_{k = 1}^{K} z_{nk} \lbrace \ln \pi_k + \ln {\cal N} ({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k) \rbrace
\tag {19}
$$
按照 EM 算法的迭代步骤，我们先求解隐变量 $ {\bf Z} $ 的后验概率函数，其具有如下形式
$$
p({\bf Z} | {\bf X}, {\bf \pi}, {\bf \mu}, {\bf \Sigma}) \propto p({\bf X}, {\bf Z} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) = \prod_{n = 1}^{N} \prod_{k = 1}^{K} ({\pi_k} {\cal N} ({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k))^{z_{nk}}
\tag {20}
$$
再来推导完全数据集的对数似然函数在 $ p({\bf Z} | {\bf X}, {\bf \pi}, {\bf \mu}, {\bf \Sigma}) $ 下的期望
$$
{\Bbb E}_{\bf Z}[ \ln p({\bf X}, {\bf Z} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) ] = \sum_{n = 1}^{N} \sum_{k = 1}^{K} {\Bbb E}[z_{nk}] \lbrace \ln \pi_k + \ln {\cal N} ({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k) \rbrace
\tag {21}
$$
而 $ {\Bbb E}[z_{nk}] $ 的值可以根据式 $ (20) $ 求出，由于 $ z_{nk} $ 只可能取 $ 1 $ 或 $ 0 $，而取 $ 0 $ 时对期望没有贡献，故有
$$
{\Bbb E}[z_{nk}] = p(z_{nk} = 1 | {\bf x}_n, {\bf \pi}, {\bf \mu}_k, {\bf \Sigma}_k) = \frac{\pi_k {\cal N}({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k)} { \sum_{j} \pi_j {\cal N} ({\bf x}_n | {\bf \mu}_j, {\bf \Sigma}_j)} = \gamma (z_{nk})
\tag {22}
$$
将上式代入公式 $ (21) $ 中，可得
$$
{\Bbb E}_{\bf Z}[ \ln p({\bf X}, {\bf Z} | {\bf \pi}, {\bf \mu}, {\bf \Sigma}) ] = \sum_{n = 1}^{N} \sum_{k = 1}^{K} \gamma (z_{nk}) \lbrace \ln \pi_k + \ln {\cal N} ({\bf x}_n | {\bf \mu}_k, {\bf \Sigma}_k) \rbrace
\tag {23}
$$
接下来我们就可以对该式关于参数 $ \lbrace {\bf \pi}, {\bf \mu}, {\bf \Sigma} \rbrace $ 求解最大值了，可以验证，各参数的更新方程就是式 $ (8) $ 至 $ (10) $。

## 与 k-means 算法的关系
敏感的人很容易就发现求解 GMM 的最大似然参数估计问题的 EM 算法和 k-means 算法非常相似，比如二者的每一步迭代都分为两个步骤、二者的每一步迭代都会使目标函数减小或是似然函数增大、二者都对初始值敏感等等。实际上，k-means 算法是“用于求解 GMM 的 EM 算法”的特例。

首先，k-means 算法对数据集的建模是一个简化版的高斯混合模型，该模型仍然含有 $ K $ 个高斯分量，但 k-means 算法做了如下假设：
1. 假设每个高斯分量的先验概率相等，即 $ \pi\_k = 1 / K $;
2. 假设每个高斯分量的协方差矩阵均为 $ \epsilon {\bf I} $。

所以某一个高斯分量的概率密度函数为
$$
p({\bf x} | {\bf \mu}_k, {\bf \Sigma}_k) = \frac {1} {(2\pi\epsilon)^{D/2}} \exp \lbrace -\frac {\| {\bf x} - {\bf \mu}_k \|^{2}} {2\epsilon} \rbrace
\tag {24}
$$
故根据 EM 算法，可求得隐变量的后验概率函数为
$$
\gamma(z_{nk}) = \frac{\pi_k \exp \lbrace -\| {\bf x} - {\bf \mu}_k \|^{2} /2\epsilon \rbrace } {\sum_j \pi_j \exp \lbrace -\| {\bf x} - {\bf \mu}_j \|^{2} /2\epsilon \rbrace } = \frac{\exp \lbrace -\| {\bf x} - {\bf \mu}_k \|^{2} /2\epsilon \rbrace } {\sum_j \exp \lbrace -\| {\bf x} - {\bf \mu}_j \|^{2} /2\epsilon \rbrace }
\tag {25}
$$
在这里，k-means 算法做了第三个重要的改变，它使用硬分配策略来将每一个样本分配给该样本点对应的 $ \gamma(z\_{nk}) $ 的值最大的那个高斯分量，即有
$$
r_{nk} = \begin{cases} 1, & \text {if \( k = \rm {arg}  \rm {min}_{j}  \| {\bf x}_n -  {\bf \mu}_j \|^{2} \) } \\ 0, & \text {otherwise} \end{cases} 
\tag {26}
$$
由于在 k-means 算法里面只有每个高斯分量的期望对其有意义，因而后续也只对 $ {\bf \mu}_k $ 求优，将式 $ (8) $ 中的 $ \gamma(z_{nk}) $ 替换为 $ r_{nk} $，即可得到 $ {\bf \mu}_k $ 的更新方法，与 k-means 算法中对中心点的更新方法一致。

现在我们再来理解 k-means 算法对数据集的假设就非常容易了：由于假设每个高斯分量的先验概率相等以及每个高斯分量的协方差矩阵都一致且正比于单位阵，所以 k-means 算法期待数据集具有球状的 `cluster`、期待每个 `cluster` 中的样本数量相近、期待每个 `cluster` 的密度相近。

----
# 实现 GMM 聚类
前面我们看到，在将数据集建模为高斯混合模型，并利用 EM 算法求解出了该模型的参数后，我们可以顺势利用 $ \gamma(z_{nk}) $ 的值来对数据集进行聚类。$ \gamma(z_{nk}) $ 给出了样本 $ {\bf x}_n $ 是由 `cluster` $ k $ 产生的置信程度，最简单的 GMM 聚类即是将样本 $ {\bf x}_n $ 分配给 $ \gamma(z_{nk}) $ 值最大的 `cluster`。在这一部分，我们先手写一个简单的 GMM 聚类算法；然后再使用 scikit-learn 中的 `GaussianMixture` 类来展示 GMM 聚类算法对不同类型的数据集的聚类效果。

## 利用 python 实现 GMM 聚类

首先我们手写了一个 GMM 聚类算法，并将其封装成了一个类，代码如下所示：
```python
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs

class GMMClust():
    def __init__(self, n_components=2, max_iter=100, tol=1e-10):
        self.data_set = None
        self.n_components = n_components
        self.pred_label = None
        self.gamma = None
        self.component_prob = None
        self.means = None
        self.covars = None
        self.max_iter = max_iter
        self.tol = tol

    # 计算高斯分布的概率密度函数 
    @staticmethod
    def cal_gaussian_prob(x, mean, covar, delta=1e-10):
        n_dim = x.shape[0]
        covar = covar + delta * np.eye(n_dim)
        prob = np.exp(-0.5 * np.dot((x - mean).reshape(1, n_dim),
                                    np.dot(np.linalg.inv(covar),
                                           (x - mean).reshape(n_dim, 1))))
        prob /= np.sqrt(np.linalg.det(covar) * ((2 * np.pi) ** n_dim))
        return prob

    # 计算每一个样本点的似然函数 
    def cal_sample_likelihood(self, i):
        sample_likelihood = sum(self.component_prob[k] *
                                self.cal_gaussian_prob(self.data_set[i],
                                                       self.means[k], self.covars[k])
                                for k in range(self.n_components))
        return sample_likelihood

    def predict(self, data_set):
        self.data_set = data_set
        self.n_samples, self.n_features = self.data_set.shape
        self.pred_label = np.zeros(self.n_samples, dtype=int)
        self.gamma = np.zeros((self.n_samples, self.n_components))

        start_time = time.time()

        # 初始化各参数
        self.component_prob = [1.0 / self.n_components] * self.n_components
        self.means = np.random.rand(self.n_components, self.n_features)
        for i in range(self.n_features):
            dim_min = np.min(self.data_set[:, i])
            dim_max = np.max(self.data_set[:, i])
            self.means[:, i] = dim_min + (dim_max - dim_min) * self.means[:, i]
        self.covars = np.zeros((self.n_components, self.n_features, self.n_features))
        for i in range(self.n_components):
            self.covars[i] = np.eye(self.n_features)

        # 开始迭代
        pre_L = 0
        iter_cnt = 0
        while iter_cnt < self.max_iter:
            iter_cnt += 1
            crt_L = 0
            # E 步
            for i in range(self.n_samples):
                sample_likelihood = self.cal_sample_likelihood(i)
                crt_L += np.log(sample_likelihood)
                for k in range(self.n_components):
                    self.gamma[i, k] = self.component_prob[k] * \
                                       self.cal_gaussian_prob(self.data_set[i],
                                                              self.means[k],
                                                              self.covars[k]) / sample_likelihood
            # M 步
            effective_num = np.sum(self.gamma, axis=0)
            for k in range(self.n_components):
                self.means[k] = sum(self.gamma[i, k] * self.data_set[i] for i in range(self.n_samples))
                self.means[k] /= effective_num[k]
                self.covars[k] = sum(self.gamma[i, k] *
                                     np.outer(self.data_set[i] - self.means[k],
                                              self.data_set[i] - self.means[k])
                                     for i in range(self.n_samples))
                self.covars[k] /= effective_num[k]
                self.component_prob[k] = effective_num[k] / self.n_samples

            print("iteration %s, current value of the log likelihood: %.4f" % (iter_cnt, crt_L))

            if abs(crt_L - pre_L) < self.tol:
                break
            pre_L = crt_L

        self.pred_label = np.argmax(self.gamma, axis=1)
        print("total iteration num: %s, final value of the log likelihood: %.4f, "
              "time used: %.4f seconds" % (iter_cnt, crt_L, time.time() - start_time))

    # 可视化算法的聚类结果
    def plot_clustering(self, kind, y=None, title=None):
        if kind == 1:
            y = self.pred_label
        plt.scatter(self.data_set[:, 0], self.data_set[:, 1],
                    c=y, alpha=0.8)
        if kind == 1:
            plt.scatter(self.means[:, 0], self.means[:, 1],
                        c='r', marker='x')
        if title is not None:
            plt.title(title, size=14)
        plt.axis('on')
        plt.tight_layout()
```

创建一个 `GMMClust` 类的实例即可对某数据集进行 GMM 聚类，在创建实例的时候，会初始化一系列的参数，如聚类个数、最大迭代次数、终止迭代的条件等等；然后该实例调用自己的方法 `predict` 即可对给定的数据集进行 GMM 聚类；方法 `plot_clustering` 则可以可视化聚类的结果。利用 `GMMClust` 类进行 GMM 聚类的代码如下所示：
```python
    # 生成数据集
    n_samples = 1500
    centers = [[0, 0], [5, 6], [8, 3.5]]
    cluster_std = [2, 1.0, 0.5]
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)

    # 运行 GMM 聚类算法
    gmm_cluster = GMMClust(n_components=3)
    gmm_cluster.predict(X)
    for i in range(3):
        print("cluster %s" % i)
        print("    mean: %s, covariance: %s" %(gmm_cluster.means[i], gmm_cluster.covars[i]))

    # 可视化数据集的原始类别情况以及算法的聚类结果
    plt.subplot(1, 2, 1)
    gmm_cluster.plot_clustering(kind=0, y=y, title='The Original Dataset')
    plt.subplot(1, 2, 2)
    gmm_cluster.plot_clustering(kind=1, title='GMM Clustering Result')
    plt.show()
```

以上代码首先由三个不同的球形高斯分布产生了一个数据集，之后我们对其进行 GMM 聚类，可得到如下的输出和可视化结果：
```
iteration 1, current value of the log likelihood: -15761.9757
iteration 2, current value of the log likelihood: -6435.3937
iteration 3, current value of the log likelihood: -6410.5633
iteration 4, current value of the log likelihood: -6399.4306
iteration 5, current value of the log likelihood: -6389.0317
iteration 6, current value of the log likelihood: -6377.9131
iteration 7, current value of the log likelihood: -6367.5704
iteration 8, current value of the log likelihood: -6359.2076
iteration 9, current value of the log likelihood: -6350.8678
iteration 10, current value of the log likelihood: -6338.6458
... ...
iteration 35, current value of the log likelihood: -5859.0324
iteration 36, current value of the log likelihood: -5859.0324
iteration 37, current value of the log likelihood: -5859.0324
iteration 38, current value of the log likelihood: -5859.0324
iteration 39, current value of the log likelihood: -5859.0324
iteration 40, current value of the log likelihood: -5859.0324
total iteration num: 40, final value of the log likelihood: -5859.0324, time used: 18.3565 seconds
cluster 0
    mean: [ 0.10120126 -0.04519941], covariance: [[3.49173063 0.08460269]
 [0.08460269 3.95599185]]
cluster 1
    mean: [5.03791461 5.9759609 ], covariance: [[ 1.0864461  -0.00345936]
 [-0.00345936  0.9630804 ]]
cluster 2
    mean: [7.99780506 3.51066619], covariance: [[0.23815215 0.01120954]
 [0.01120954 0.27281129]]
```

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/102854/22/19383/939345/5e9d22e0E299b5a2e/67904aca69ecfeb9.png" width = "1000" height = "500" alt = "GMM 聚类结果" align = center />
</div>


可以看到，对于给定的数据集， GMM 聚类的效果是非常好的，和数据集原本的 `cluster` 非常接近。其中一部分原因是由于我们产生数据集的模型就是一个高斯混合模型，但另一个更重要的原因可以归结为高斯混合模型是一个比较复杂、可以学习到数据中更为有用的信息的模型；因而一般情况下，GMM 聚类对于其它的数据集的聚类效果也比较好。但由于模型的复杂性，GMM 聚类要比 k-means 聚类迭代的步数要多一些，每一步迭代的计算复杂度也更大一些，因此我们一般不采用运行多次 GMM 聚类算法来应对初始化参数的问题，而是先对数据集运行一次 k-means 聚类算法，找出各个 `cluster` 的中心点，然后再以这些中心点来对 GMM 聚类算法进行初始化。

## 利用 sklearn 实现 GMM 聚类
sklearn 中的  `GaussianMixture` 类可以用来进行 GMM 聚类，其中的 `fit` 函数接收一个数据集，并从该数据集中学习到高斯混合模型的参数；`predict` 函数则利用前面学习到的模型对给定的数据集进行 GMM 聚类。在这里，我们和前面利用 sklearn 实现 k-means 聚类的时候一样，来考察 GMM 距离在不同类型的数据集下的聚类结果。代码如下：
```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170

# 产生一个理想的数据集
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
gmm = GaussianMixture(n_components=3, random_state=random_state)
gmm.fit(X)
y_pred = gmm.predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Normal Blobs")

# 产生一个非球形分布的数据集
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
gmm = GaussianMixture(n_components=3, random_state=random_state)
gmm.fit(X_aniso)
y_pred = gmm.predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# 产生一个各 cluster 的密度不一致的数据集
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
gmm = GaussianMixture(n_components=3, random_state=random_state)
gmm.fit(X_varied)
y_pred = gmm.predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Density Blobs")

# 产生一个各 cluster 的样本数目不一致的数据集
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:50]))
gmm = GaussianMixture(n_components=3, random_state=random_state)
gmm.fit(X_filtered)
y_pred = gmm.predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()
```

运行结果如下图所示：
<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/99160/5/19427/295974/5e9d2328Ea6fca295/44ab6241e831cab7.png" width = "780" height = "650" alt = "GMM在不同数据集下的表现" align = center />
</div>


可以看到，GMM 聚类算法对不同的数据集的聚类效果都很不错，这主要归因于高斯混合模型强大的拟合能力。但对于非凸的或者形状很奇怪的 `cluster`，GMM 聚类算法的聚类效果会很差，这还是因为它假设数据是由高斯分布所产生，而高斯分布产生的数据组成的 `cluster` 都是超椭球形的。









[1]:	../clustering-analysis/index.html
[2]:	https://en.wikipedia.org/wiki/Lagrange_multiplier?oldformat=true
[3]:	https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence?oldformat=true