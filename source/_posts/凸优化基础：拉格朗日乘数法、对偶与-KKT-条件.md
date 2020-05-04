---
title: 凸优化基础：拉格朗日乘数法、对偶与 KKT 条件
date: 2019-05-01 12:26:57
tags:
- 凸优化
- 数学优化
- 对偶
- 拉格朗日乘数法
- KKT 条件
categories:
- 数学基础
keywords: 凸优化,拉格朗日乘数法,Lagrange Multipliers,对偶,KKT条件
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# 数学优化概述

所谓数学优化，就是在给定约束的情况下对某个目标函数求最小值或最大值。信号处理领域和通信领域的很多问题，最终的求解形式都可以归结为数学优化问题。

凸优化问题是数学优化问题的一个子类，该问题的目标函数是凸函数，且可行域是一个凸集。对于很多数学优化问题，科学界并没有一套通用且可行的解法，但对于解决凸优化问题，其理论和方法都已经很成熟了。因而现在对于非凸问题的解法通常都是先将原问题松弛至一个凸问题，然后再求解该凸问题。

本文先讲述用于求解局部最优化问题的拉格朗日乘数法（Lagrange Multipliers）以及局部最优解需满足的 KKT 条件；然后再讲述对偶（duality）—— 一种将原优化问题简化的方法；最后再对凸优化问题、对偶、KKT 条件三者之间的关系做一个简单的讲解。

在讲述前先对数学优化问题给出一个正式且通用的定义：

> **定义 1：** 函数 $f,g_1,\cdots,g_m$ 以及   $h_1,\cdots,h_l$ 均被定义在自变量空间 $\Omega = {\Bbb R}^n $ 上，则一般的优化问题具有如下形式
>
> $$
> \min_{ {\bf x} \in \Omega} f({\bf x}) \quad {\text {s.t.} }  \quad g_j({\bf x}) \leq 0 \ \  \forall j  \ \   {\text {and}  }  \ \   h_i({\bf x}) = 0 \  \  \forall i
> \tag{1}
> $$

后面都会遵循这个定义及对应的符号。比如后面我们都会假定优化问题为求目标函数在可行域内的最小值（或极小值），并且假定不等式约束的方向是“约束函数小于等于0”。这样的假设是合理的，因为任何一个其他优化问题都可以转换为上述形式。

---
# 局部最优化问题及其解法

局部最优即是函数的极小值（或极大值）。对于无约束条件的目标函数（假设其连续可导），极值处的点  ${\bf x}^{\ast}$  必然满足导数或梯度为 ${\bf 0}$  的性质：
$$
\nabla_{\bf x} f({\bf x}^{\ast}) = {\bf 0}
\tag{2}
$$

但满足式（2）的点并不一定是极值点，还有可能是鞍点（saddle point）。为此，我们进一步考虑目标函数的二阶特性，若在满足式（2）的地方同时满足如下二阶正定的特性（式（3）），则该点必然是目标函数的一个极小值点。
$$
{\bf v}^t \nabla^2 f({\bf x}^{\ast}) {\bf v} > {\bf 0},\ \forall {\bf v} \in {\Bbb R}^{n} \backslash \lbrace {\bf 0} \rbrace
\tag{3}
$$

其中  $ \nabla^2 f({\bf x}) $  为 Hessian 矩阵，体现的是目标函数  $f({\bf x})$  的二阶特性。在某一点处的 Hessian 矩阵是正定矩阵说明在该点的邻域内目标函数是一个凸函数，这也符合直觉。

这样，式（2）和式（3）构成了无约束局部最小化问题的解的充要条件。类似地，式（2）和 “Hessian 矩阵为负定矩阵”则是无约束局部最大化问题的解的充要条件，后面除非特殊说明，我们不再讨论求最大值或极大值的问题。

## 等式约束优化问题

我们通过一个例子来说明等式约束优化问题的解法。
> ** 问题 1:  ** 求解等式约束优化问题 $\min_{ {\bf x} \in {\Bbb R}^2 } f({\bf x}) \ \  {\text {s.t.} }  \ \   h({\bf x}) = 0$ ，其中  $ f({\bf x}) = x_1 + x_2 ,\ h({\bf x}) = x_{1}^2 + x_{2}^2 - 2 $。

我们可以看到该例中的自变量 $ {\bf x} $ 为一个二维向量，$ x_1 $ 和 $x_2$ 分别是 ${\bf x} $ 的两个分量，这样的设定也非常利于我们作可视化的展示。

我们可以在一个二维坐标系中画出目标函数  $ f({\bf x}) = x_1 + x_2$ 在给定函数值下的图像，变化函数值我们可以得出一组类似等高线的图像（如下图中蓝色曲线）；同时我们还可以画出由等式约束  $h({\bf x}) =  0 $ 所构成的可行域的图像（如下图中红色曲线）。
<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/114022/29/4113/192146/5eab0404E70052ef1/ce72208d06341c07.png" width = "900" height = "600" alt = "f(x) 图像及 h(x) = 0 图像" align = center />
</div>

给定可行域上的某个点  ${\bf x}_{\text F}$，若我们想要向问题的最优解逼近（假设将自变量变化 $\delta {\bf x}$ ），须满足以下两个条件：

$$
h({\bf x}_{\text F} + \alpha \delta {\bf x}) = 0 
\tag{4}
$$

$$
f({\bf x}_{\text F} + \alpha \delta {\bf x}) \lt f({\bf x}_{\text F})
\tag{5}
$$

要想满足条件（4），$\delta {\bf x}$  的方向必须与等式约束所构成的曲面  $h({\bf x}) =  0$  的切向量平行，亦即与约束函数的梯度方向 $\nabla_{\bf x} h({\bf x}_{\text F}) $ 正交，如下图所示。
<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/119212/24/4045/167464/5eab0405Ea52dc99a/6c7ab40a66d1a952.png" width = "900" height = "600" alt = "可行域曲面的切向量方向" align = center />
</div>


要想满足条件（5），则  $\delta {\bf x}$  须有与目标函数的梯度反方向  $-\nabla_{\bf x} f({\bf x}_{\text F})$ 方向一致的分量，即须满足
$$
\delta {\bf x} \cdot (-\nabla_{\bf x} f({\bf x}_{\text F})) \gt 0
\tag{6}
$$

如下图所示。
<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/114174/14/4016/165911/5eab0418E97cc904d/1111ed8fde2717d1.png" width = "900" height = "600" alt = "目标函数的负梯度方向" align = center />
</div>

将以上两点结合起来看，我们很快就可以找到求解等式约束下的优化问题的最优解须满足的条件
$$
\nabla_{\bf x} f({\bf x}_{\text F}) = \mu \nabla_{\bf x} h({\bf x}_{\text F})
\tag{7}
$$

其中  $\mu$  为任意标量。在该场景下，由于  $\delta {\bf x}$  需要正交于  $h({\bf x}_{\text F})$ ，即有
$$
\delta {\bf x} \cdot (-\nabla_{\bf x} f({\bf x}_{\text F})) = -\delta {\bf x} \cdot \mu \nabla_{\bf x} h({\bf x}_{\text F}) = 0
$$

即在满足式（7）的  ${\bf x}_{\text F}$ 处，再也找不到一个变化自变量的方向，使得变化后的 $ {\bf x} $ 仍然在可行域内的同时使得目标函数减小，因而式（7）构成了求解等式约束问题的一个必要条件。

下图展示了问题 1 中满足式（7）的两个点（critical points）。
<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/115424/33/3987/147120/5eab0418Eff2d667c/038d019c3fcab7ea.png" width = "900" height = "600" alt = "critical points" align = center />
</div>

但仅仅满足式（7）并不能保证  ${\bf x}_{\text F}$ 是带等式约束的局部最小值问题的解，它同样有可能是局部最大值问题的解或者鞍点，因而我们仍然需要再加上二阶正定的条件。综上所述，式（7）和二阶正定特性一起构成了带等式约束的局部最小问题的解，我们通过如下定理对上面的结论做一个正式的阐述。
> ** 定理 1: **  对于等式约束优化问题
> $$
> \min_{ {\bf x} \in {\Bbb R}^n} f({\bf x}) \ \ \  \text {s.t.} \ \ \ h({\bf x}) = 0\tag{8}
> $$
> 定义拉格朗日乘数
> $$
> {\cal L}({\bf x}, \mu) = f({\bf x}) + \mu h({\bf x})\tag{9}
> $$
> 则若 ${\bf x}^{\ast}$ 为该问题的解，则必存在 $\mu^{\ast}$，满足如下条件
> $$
> \nabla_{\bf x} {\cal L}({\bf x}^{\ast}, \mu^{\ast}) = {\bf 0}\tag{10}
> $$
>
> $$
> \nabla_{\mu} {\cal L}({\bf x}^{\ast}, \mu^{\ast}) = 0\tag{11}
> $$
>
> $$
> {\bf y}^t(\nabla_{\bf x\bf x} {\cal L}({\bf x}^{\ast}, \mu^{\ast})){\bf y} \geq 0 \ \ \ \forall {\bf y} \ \ \text {s.t.} \ \ \nabla_{\bf x}h({\bf x}^{\ast})^t {\bf y} = 0\tag{12}
> $$
>
> 反过来，若式（10）～（12）成立，则 ${\bf x}^{\ast}$ 必然为原优化问题的一个局部最小解。

即式（10）～（12）是该等式约束局部优化问题有解的充要条件。注意到 ${\cal L}({\bf x}^{\ast}, \mu^{\ast}) = f({\bf x}^{\ast})$ ，式（10）等价于式（7）；式（11）强调了最优解一定满足等式约束条件；式（12）即为二阶正定性条件。

## 不等式约束优化问题

我们仍然通过一个例子来说明不等式约束优化问题的解法。

> **问题 2：**求解不等式约束优化问题 $\min_{ {\bf x} \in {\Bbb R}^2 } f({\bf x}) \ \  {\text {s.t.} }  \ \   g({\bf x}) \leq 0$ ，其中  $f({\bf x}) = x_1^2 + x_2^2, \  \ g({\bf x}) = x_1^2 + x_2^2 - 1$.

函数 $f({\bf x}) = x_1^2 + x_2^2$ 的等高线如下图所示。

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/114653/18/4098/234669/5eabdf05E04347c35/dab2dcebde1ce8e9.png" width = "900" height = "600" alt = "iso-contours of the objective function" align = center />
</div>

不等式约束 $g({\bf x}) \leq 0$ 所产生的可行域如下图所示。

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/119328/17/4115/225961/5eabf6d0E1fd0530e/08af8353e35bc7f6.png" width = "900" height = "600" alt = "feasible region" align = center />
</div>

我们可以看到在这个例子中，无约束条件下优化问题的解 ${\bf x}^{\ast}$ 正好落在可行域内，即 $g({\bf x}^{\ast}) < 0$，如下图所示。此时不等式约束优化问题的解即等价于无约束优化问题的解，即式（2）和式（3）是该问题的解的充要条件。 

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/108799/16/15149/275001/5eabf923E758a94f0/826e7a7ae4c9f78f.png" width = "900" height = "600" alt = "unconstraint minimum lies in the feasible region" align = center />
</div>

再考虑无约束条件下优化问题的解落在可行域之外的情形，考虑如下例子。

> **问题 3：**求解不等式约束优化问题 $\min_{ {\bf x} \in {\Bbb R}^2 } f({\bf x}) \ \  {\text {s.t.} }  \ \   g({\bf x}) \leq 0$ ，其中  $f({\bf x}) = (x_1 - 1.1)^2 + (x_2 - 1.1)^2$, $g({\bf x}) = x_1^2 + x_2^2 - 1$.

此时目标函数的等高线和可行域如下图所示，此时无约束优化问题的解落在可行域之外。我们可以推断，该情形下不等式约束优化问题的解必然在可行域的边界上，即此时问题退化为等式约束优化问题的求解。

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/113252/28/4109/227223/5eabf922E32286a84/1a8bb891e14e22ad.png" width = "900" height = "600" alt = "unconstraint minimum lies outside the feasible region" align = center />
</div>

回忆之前等式约束优化问题须满足的条件：
$$
-\nabla_{\bf x} f({\bf x}) = \lambda \nabla_{\bf x} g({\bf x})
\tag{13}
$$
即需要目标函数的梯度的反方向（目标函数下降最快方向）与约束函数的梯度方向平行。但考虑到此时可行域是由曲面所包裹的空间而不仅仅是曲面，目标函数的梯度的反方向需要与约束函数的梯度方向一致，即  $\lambda > 0$，直观来理解就是在可行域的表面上，目标函数的下降方向与可行域的扩张方向一致，导致可行域无法再扩张（注意当前在可行域的表面），因而目标函数的值将降无可降。如下图所示。

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/111099/7/4127/272579/5eabf92fEfeaaa0b2/d44115a9495597c6.png" width = "900" height = "600" alt = "gradient descent direction of the objective function should be the same as the gradient direction of the constraint function" align = center />
</div>

因此，不等式约束优化问题的解法可以由以上两种情形所概括，用如下定理作一个总结，后面也会对该定理作进一步阐释。

> **定理 2：** 对于不等式约束优化问题
> $$
> \min_{ {\bf x} \in {\Bbb R}^n} f({\bf x}) \ \ \  \text {s.t.} \ \ \ g({\bf x}) \leq 0\tag{14}
> $$
> 定义拉格朗日乘数
> $$
> {\cal L}({\bf x}, \lambda) = f({\bf x}) + \lambda g({\bf x})\tag{15}
> $$
> 则若 ${\bf x}^{\ast}$ 为该问题的解，则必存在 $\lambda^{\ast}$，满足如下条件
> $$
> \nabla_{\bf x} {\cal L}({\bf x}^{\ast}, \lambda^{\ast}) = {\bf 0}
> \tag{16}
> $$
>
> $$
> \lambda^{\ast} \geq 0
> \tag{17}
> $$
>
> $$
> \lambda^{\ast}g({\bf x}^{\ast}) = 0
> \tag{18}
> $$
>
> $$
> g({\bf x}^{\ast}) \leq 0
> \tag{19}
> $$
>
> $$
> {\bf y}^t \nabla_{\bf x \bf x} {\cal L}({\bf x}^{\ast}, \lambda^{\ast}){\bf y} \geq 0  \ \ \ \text {for certain} \ \ {\bf y}
> \tag{20}
> $$
>
> 反过来，若式（16）～（20）成立，则 ${\bf x}^{\ast}$ 必然是原优化问题的一个局部最优解。

上面定理中式（20）为二阶正定性条件，当不等式约束没有起到实质作用时（即无约束优化问题的解在可行域内），${\bf y} \in {\Bbb R}^n \backslash \lbrace {\bf 0} \rbrace$；当不等式约束起到实质作用时，${\bf y}$ 为与 $\nabla_{\bf x} g({\bf x}^{\ast})$ 正交的任意向量。后面不再对拉格朗日乘数的二阶正定性约束条件进行说明。

我们可以看到，在不等式约束没有起到实质性作用时，由于 $g({\bf x}^{\ast}) < 0$，由式（18）可知， $\lambda^{\ast}$ 取值为 0，式（16）退化为 $\nabla_{\bf x} f({\bf x}^{\ast}) = {\bf 0}$；在不等式约束起到实质作用时，$g({\bf x}^{\ast}) = 0$，则 $\lambda^{\ast} > 0$，故 ${\cal L}({\bf x}^{\ast}, \lambda^{\ast}) = f({\bf x}^{\ast}) + \lambda^{\ast}g({\bf x}^{\ast})$，式（16）即为式（13）。

## KKT 条件
分别考虑了等式约束优化问题和不等式约束优化问题之后，我们可以将一般优化问题（1）的解的充要条件总结成如下定理。
>**定理 3：** 对于一般优化问题（1），定义拉格朗日乘数为
>$$
>{\cal L}({\bf x}, {\bf \mu}, {\bf \lambda}) = f({\bf x}) + {\bf \mu}^t {\bf h}({\bf x}) + {\bf \lambda}^t {\bf g} ({\bf x})
>\tag{21}
>$$
>其中 ${\bf \mu} = (\mu_1, \cdots,\mu_l)$，${\bf h}({\bf x}) = (h_1({\bf x}),\cdots,h_l({\bf x}))$，${\bf \lambda} = (\lambda_1,\cdots,\lambda_m)$，${\bf g}({\bf x}) = (g_1({\bf x}), \cdots, g_m({\bf x}))$ 均为向量。
>
>则优化问题（1）的局部最优解的充要条件为
>$$
>\nabla_{\bf x} {\cal L}({\bf x}^{\ast}, {\bf \mu}^{\ast}, {\bf \lambda}^{\ast}) = {\bf 0}
>\tag{22}
>$$
>
>$$
>\lambda_j^{\ast} \geq 0 \ \ \ \text {for} \ \ j = 1,\cdots,m
>\tag{23}
>$$
>
>$$
>\lambda_j^{\ast} g_j({\bf x}^{\ast}) = 0 \ \ \ \text {for} \ \ j = 1, \cdots, m
>\tag{24}
>$$
>
>$$
>g_j({\bf x}^{\ast}) \leq 0 \ \ \ \text {for} \ \ j = 1, \cdots, m
>\tag{25}
>$$
>
>$$
>h_i({\bf x}^{\ast}) = 0 \ \ \ \text {for} \ \ i = 1, \cdots, l
>\tag{26}
>$$
>
>$$
>{\bf y}^t \nabla_{\bf x \bf x} {\cal L}({\bf x}^{\ast}, {\bf \mu}^{\ast}, {\bf \lambda}^{\ast}){\bf y} \geq 0  \ \ \ \text {for certain} \ \ {\bf y}
>\tag{27}
>$$

我们称式（22）～（27）为一般优化问题（1）的 [Karush-Kuhn-Tucker（KKT） 条件](https://en.wikipedia.org/wiki/Karush–Kuhn–Tucker_conditions)。

---

# 拉格朗日对偶

## 对偶问题形式

仍然考虑一般优化问题（1），拉格朗日函数为式（21），定义
$$
p({\bf x}) = \max_{ {\bf \mu}, {\bf \lambda}: \ \lambda_j \geq 0} {\cal L}({\bf x}, {\bf \mu}, {\bf \lambda})
\tag{28}
$$
很容易发现
$$
p({\bf x}) = 
\begin{cases}
f({\bf x}),   & \text {当} \ {\bf x } \ \text {在可行域内} \\
+\infty,      & 其他
\end{cases}
\tag{29}
$$
因为当 ${\bf x}$ 在可行域内时，有 $g_j({\bf x}) \leq 0$，$h_i({\bf x}) = 0$，又要求 $\lambda_j \geq 0$，因此要对 ${\cal L}({\bf x}, {\bf \mu}, {\bf \lambda})$ 关于 $\bf \mu$ 和 $\bf \lambda$ 取最大值，只能将 $\lambda_j$ 都置为 0，因此此时拉格朗日函数的最大取值为 $f({\bf x})$；当 $\bf x$ 在可行域之外时，我们可以将项 $\sum_j \lambda_j g_j(\bf x)$ 和 $\sum_i \mu_ih_i({\bf x})$ 设置的任意大，因此此时拉格朗日函数的最大取值为 $+\infty$.

则一般优化问题（1）等价于以下问题：
$$
\min_{\bf x} p({\bf x}) = \min_{\bf x} \max_{ {\bf \mu}, {\bf \lambda}: \ \lambda_j \geq 0} {\cal L}({\bf x}, {\bf \mu}, {\bf \lambda})
\tag{30}
$$
我们称该问题或一般优化问题（1）为主问题（primal problem），记 $p^{\ast} = \min_{\bf x} p({\bf x})$  为主问题的值。

定义对偶函数为
$$
q({\bf \mu, \bf \lambda}) = \min_{\bf x} {\cal L}({\bf x}, {\bf \mu}, {\bf \lambda})
\tag{31}
$$
则对偶问题（dual problem）为
$$
\max_{\mu, \lambda: \ \lambda_j \geq 0} q({\bf \mu}, {\bf \lambda}) = \max_{\mu, \lambda: \ \lambda_j \geq 0} \min_{\bf x} {\cal L}({\bf x}, {\bf \mu}, {\bf \lambda})
\tag{32}
$$
对偶问题（32）与主问题（30）相比交换了对不同变量的优化顺序。记 $q^{\ast} = \max_{\mu, \lambda: \lambda_j \geq 0} {\cal L}({\bf x}, \mu, \lambda)$ 为对偶问题的值。

对偶问题有一些比较好的性质，使其相对来说比较容易求解。不管原始优化问题是否为凸问题，对偶函数 $q(\mu, \lambda)$ 都是一个凹函数，因而对偶问题是一个凸问题，可以简单证明如下。

为表述方便，$\beta = (\mu_1, \cdots, \mu_l, \lambda_1, \cdots, \lambda_m)$，对偶函数的定义域为 $D_q = \lbrace \beta \ | \ q(\beta) > -\infty \rbrace$，则对任意 ${\bf x} \in {\Bbb R}^n$， $\beta_a,\beta_b \in D_q$ 以及 $\alpha \in (0, 1)$，有
$$
\begin{align}
{\cal L}({\bf x}, \alpha\beta_a + (1 - \alpha)\beta_b) &= f({\bf x}) + (\alpha\mu_a + (1-\alpha)\mu_b)^t h({\bf x}) + (\alpha\lambda_a + (1 - \alpha)\lambda_b)^t g({\bf x}) \\
&= \alpha [f({\bf x}) + \mu_a^th({\bf x}) + \lambda_a^tg({\bf x})] + (1 - \alpha)[f({\bf x}) + \mu_b^th({\bf x}) + \lambda_b^tg({\bf x})] \\
&= \alpha {\cal L}({\bf x}, \beta_a) + (1 - \alpha) {\cal L} ({\bf x}, \beta_b)
\end{align}
\tag{33}
$$
对上式两边同时对 $\bf x$ 取最小值，有
$$
\begin{align}
\min_{\bf x}{\cal L}({\bf x}, \alpha\beta_a + (1 - \alpha)\beta_b) &= \min_{\bf x}[\alpha {\cal L}({\bf x}, \beta_a) + (1 - \alpha) {\cal L} ({\bf x}, \beta_b)] \\
&\geq \alpha \min_{\bf x} {\cal L}({\bf x}, \beta_a) + (1 - \alpha)\min_{\bf x}{\cal L}({\bf x}, \beta_b)
\end{align}
\tag{34}
$$
上式说明了对偶函数 $q(\mu, \lambda)$ 为凹函数。

引入对偶问题的目的就是想通过它来求解主问题，但二者并不一定等价，后面我们会讨论对偶问题与主问题之间的关系（弱对偶定理），以及满足什么条件的情况下二者等价（强对偶定理）。在此之前，我们先来看一下对偶问题的几何阐释，以便我们对问题本质认识得更加深刻。

## 对偶问题的几何阐释

我们还是通过例子来阐释对偶问题。

> **问题 4：**求解以下不等式约束优化问题 $\min_{ {\bf x} \in {\Bbb R}^2} f({\bf x}) \ \ \text {s.t.} \ \ g({\bf x}) \leq 0$，其中 $ f({\bf x}) = 0.4(x_1^2 + x_2^2)$，$g({\bf  x}) = 2 - x_1 - x_2$.

函数 $f({\bf x})$ 的等高线和由不等式约束 $g({\bf x}) \leq 0$ 所决定的可行域如下图所示。

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/119752/32/3443/285202/5eafe387Ec77dcb81/c9868648311c7f9c.png" width = "800" height = "550" alt = "the iso contour and the feasible region" align = center />
</div>

我们将该坐标系下的每个点 ${\bf x} \in {\Bbb R}^2$ 映射至空间 $(g({\bf x}), f({\bf x})) \in {\Bbb R}^2$，定义该映射产生的新的点集为
$$
G = \lbrace (y, z) \ | \ y = g({\bf x}), z = f({\bf x}) \ \ \text {for} \ \ {\bf x} \in {\Bbb R}^2 \rbrace
\tag{35}
$$
该映射产生的点集在坐标系 $y-z$ 下如下图所示

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/116803/39/4778/169787/5eafe968Eedb17e1b/0d18f8737eb352a3.png" width = "900" height = "600" alt = "the map of the original space" align = center />
</div>

上图中蓝色部分为所有 ${\bf x} \in {\Bbb R}^2$ 按照 (35) 映射之后所构成的点集，可行域为 $y \leq 0$，如上图中红色部分所示。可以很容易地看到，问题 4 要求解的“在 $y \leq 0$ 的条件下 $z$ 最小”这个问题的解即为上图中的红色点 $(y^{\ast}, z^{\ast})$。 

我们再来分析问题 4 的拉格朗日对偶问题，其对应的拉格朗日函数为
$$
{\cal L}({\bf x}, \lambda) = f({\bf x}) + \lambda g({\bf x}) = z + \lambda y, \ \ \ \lambda  \geq 0
\tag{36}
$$
我们考虑直线 $z + \lambda y = \alpha$ 的特点，该直线的斜率为 $-\lambda$，与 $z$ 轴的截距为 $\alpha$，如下图所示

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/117192/4/4753/124232/5eb0025dEf022018c/22586fe127c6d6b8.png" width = "900" height = "600" alt = "the line of lagrangian in z-y space" align = center />
</div>

固定 $\lambda$ 时（即固定直线的斜率），我们在 $G$ 内平移直线（即需要直线经过 $G$ 内至少一个点），此时我们根据 $G$ 内所有斜率为 $-\lambda$ 的直线与 $z$ 轴的最小截距来求得拉格朗日对偶函数 $ q(\lambda) = \min_{\bf x} {\cal L}({\bf x}, \lambda) = \min_{(y, z) \in G} (z + \lambda y) $ 的值，如下图所示

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/112451/10/4731/133581/5eb00999E2f36c388/59a2972bb84762c4.png" width = "900" height = "600" alt = "the lagrangian duality function in z-y space" align = center />
</div>

我们再来变化 $\lambda$，来求解对偶问题 $\max_{\lambda: \lambda \geq0} q(\lambda)$，在几何意义上，就是变化上述直线的斜率，找到使得以斜率来划分的可行直线族中的最小 $z$ 截距 $q(\lambda)$ 最大的 $\lambda^{\ast}$.  我们可以看到，在这个问题中，对偶问题的解与主问题的解一致（我们称该现象为强对偶），如下图所示。

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/118669/6/4814/157717/5eb00be2Ee921610b/725e0af5bade5b15.png" width = "900" height = "600" alt = "the optimum of the lagrangian duality problem in z-y space" align = center />
</div>

## 弱对偶定理

弱对偶定理给出了一般情形下，对偶问题与主问题的最优值之间的关系，如下所述。

> **定理 4：**令 ${\bf x} \in {\Bbb R}^n$ 为一般优化问题（1）的主问题（30）的一个可行解； 令 $({\bf \mu}, {\bf \lambda})$ 为对偶问题（32）的一个可行解，即满足 $\lambda_j \geq {\bf 0}$，则有
> $$
> p({\bf x}) \geq q({\bf \mu, \bf \lambda})
> \tag{37}
> $$
> 更进一步，令主问题的最优值为 $p^{\ast} = \min_{\bf x} p({\bf x})$，令对偶问题的最优值为 $q^{\ast} = \max_{\mu, \lambda: \lambda_j \geq 0} q(\mu, \lambda)$，有
> $$
> p^{\ast} \geq q^{\ast}
> \tag{38}
> $$

证明过程非常简单，有
$$
p({\bf x}) = \max_{\mu, \lambda: \lambda_j \geq 0} {\cal L}({\bf x}, \mu, \lambda) \geq  {\cal L}({\bf x}, \mu, \lambda) \geq \min_{\bf x} {\cal L}({\bf x, \mu, \lambda}) = q({\mu, \lambda})
\tag{39}
$$
上式即证明了（37），再对（39）左边关于 $\bf x$ 求极小值，右边关于 $(\mu, \lambda)$ 求极大值，可证明（38）：
$$
p^{\ast} = \min_{\bf x} p({\bf x}) \geq \max_{\mu, \lambda: \lambda_j \geq 0} q(\mu, \lambda) = q^{\ast}
\tag{40}
$$
弱对偶定理说明了对一般优化问题 （1），对偶问题的最优值 $p^{\ast}$ 是主问题的最优值 $q^{\ast}$ 的一个下限，我们称 $p^{\ast}$ 与 $q^{\ast}$ 之间的差异为对偶差（duality gap）。

举个对偶差的简单例子，考虑下图所示的一维优化问题

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/112274/1/4871/182239/5eb01949E591f4f7a/677dfd5d01bd6291.png" width = "900" height = "600" alt = "the duality gap example-1" align = center />
</div>

可以看到目标函数 $f(x)$ 是非凸的，不等式约束函数 $g(x) = -x - m$，按照式（35）的做法，将 $x \in {\Bbb R}$ 映射至 $z-y$ 平面中，如下图所示

<div align = center>
<img src="http://free-cn-01.cdn.bilnn.com/ddimg/jfs/t1/117656/32/4845/201074/5eb01948E4e37434b/19bb13d944ee0198.png" width = "900" height = "600" alt = "the duality gap example-2" align = center />
</div>

可以看到这个问题中，主问题的最优值和对偶问题的最优值并不一致，两者之间存在对偶差。

那么在什么情形下，主问题的解与对偶问题的解一致呢？以下的强对偶定理给出回答。

## 强对偶定理

> **定理 5：** 对于一般优化问题（1），若 $f({\bf x})$ 和 $g_j({\bf x}), \ j = 1,\cdots,m$ 均为凸函数，且 $h_i({\bf x}), \ i = 1,\cdots,l$ 均为仿射函数，且约束 $g_j({\bf x}) \leq 0$ 是严格可行的，则存在 ${\bf x}^{\ast}$，${\mu}^{\ast}$，${\lambda}^{\ast}$ ，使得 ${\bf x}^{\ast}$ 是主问题（30）的解，$(\mu^{\ast}, \lambda^{\ast})$ 是对偶问题（32）的解，有
> $$
> p^{\ast} = q^{\ast} = {\cal L}({\bf x}^{\ast}, \mu^{\ast}, \lambda^{\ast})
> \tag{41}
> $$
> 

定理 5 说明了在一定的条件下（该条件为 [Slater 条件]([https://en.wikipedia.org/wiki/Slater%27s_condition](https://en.wikipedia.org/wiki/Slater's_condition)) ），强对偶成立，我们可以通过解决对偶问题来解决主问题，对强对偶定理的证明详见 [Slater 强对偶条件证明](https://www.ece.nus.edu.sg/stfpage/vtan/ee5138/slater.pdf)。另外，当 Slater 条件满足时，优化问题的最优解满足式（22）～（27）所列出的 KKT 条件。

