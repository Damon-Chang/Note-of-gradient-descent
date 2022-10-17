# An overview of gradient descent optimization algorithms
> - 梯度下降算法概述，是一个综述性文章。
> - 梯度下降优化算法虽然越来越流行，但通常被用作黑盒优化器，因为很难找到对其优缺点的实际解释。本文旨在为读者提供有关不同算法行为的直觉，使她能够使用它们。在本概述的过程中，我们研究了梯度下降的不同变体，总结了挑战，介绍了最常见的优化算法，审查了并行和分布式设置中的架构，并研究了优化梯度下降的其他策略
- Section 2：介绍不同的梯度算法；
- Section 3：简要总结训练中遇到的挑战；
- Section 4：从如何解决遇到的挑战角度介绍最常见的优化算法，然后得到这些算法的更新规则；
- Section 5：介绍在并行和分布式设置中优化梯度下降的算法和架构；
- Section 6：考虑有助于优化梯度下降的其他策略；

## Gradient Descent

> 梯度下降用于大多数受欢迎的算法，也是最普遍的优化神经网络的算法。同时每个深度学习库也包含许多梯度下降算法。这些算法通常被用作黑盒优化，因为很难找到对其优缺点的实际解释。

**更新准则：**
目标函数 $J(\theta)$ ，其中 $\theta\in\mathcal R^d$ ：
$$\theta=\theta-\eta\nabla_{\theta}J(\theta)$$
### Gradient descent的变体
#### 批量梯度下降（Batch gradient descent，BGD）
计算 $\nabla J(\theta)$ 时不使用全部数据，分次（**epoch**）选择一定批量的样本计算。批量梯度下降保证了 $J(\theta)$ 在凸函数时收敛到全局极小值点（**global minimum**），在非凸问题上收敛到局部极小值点（**local minimum**）。
- 优点：每次的计算量减小，易于实现，方便计算；
- 缺点：重复计算产生赘余，速度慢，不能在线计算（新数据不友好）。
```
for i in range(nb_epochs ):
	params_grad = evaluate_gradient(loss_function , data , params)
	params = params - learning_rate * params_grad
```
#### 随机梯度下降（Stochastic gradient descent，SGD）
相比于BGD下降，SGD对每一个数据 $(x^{(i)},y^{(i)})$ 都更新参数：

$$\theta=\theta-\eta\nabla_{\theta}J(\theta;x^{(i)};y^{(i)})$$

- 优点：每个样本点更新一次参数，减少了冗余计算，提高了算法运行速度；
- 缺点：SGD以高方差频繁更新参数导致目标函数大幅波动。
> **SGD和批量梯度下降的比较**   
> SGD可能会在收敛到极小值点后继续改变参数值，也可能跳出当前极小值点收敛到更好的局部极小值点。  
> 实践表明，当慢慢降低学习率时，SGD和批量梯度下降巨有相同的收敛性。

	for i in range(nb_epochs ):
		np.random.shuffle(data)
		for example in data:
			params_grad = evaluate_gradient(loss_function , example , params)
			params = params - learning_rate * params_grad

#### 小批量梯度下降（Mini-batch gradient descent)
小批量梯度下降结合了SGD和批量梯度下降优点，对样本中每个小批量中 $n$ 个样本进行参数更新:

$$ \theta=\theta-\eta\nabla_{\theta}J(\theta;x^{(i:i+n)};y^{(i:i+n)}) $$

这样一来:   

- a) 降低了参数更新的方差，是的目标函数收敛更稳定；   
- b) 可以利用最先进的深度学习库所共有的高度优化的矩阵优化，这些库可以计算小批量数据的梯度非常有效。

一般小批量的数量在50~256，可根据处理问题的不同选择。小批量SGD通常是训练神经网络的方法，一般认为SDG就是小批量SGD。

	for i in range(nb_epochs ):
		np.random.shuffle(data)
		for batch in get_batches(data , batch_size =50):
			params_grad = evaluate_gradient(loss_function , batch , params)
			params = params - learning_rate * params_grad

## 挑战
小批量梯度下降算法并不能保证算法的首先性，面对着如下挑战：
- 学习率选择困难。学习率过小会导致收敛极其缓慢；学习率过大会破坏收敛而且使损失方程在最小值点震荡甚至发散。
- 学习率时间表尝试在训练过程中调整学习率，例如使用**退火算法（annealing）**，即通过一个预定义好的时间表和阈值降低学习率，或是目标函数一次迭代变化量低于阈值的时候。这些时间表和阈值是提前定义好的，因此不能够适应数据集的特征；
- 相同的学习率用于所有参数更新的挑战。如果数据是稀疏的而特征是频率相差很大，则不应该为所有参数更新执行相同的学习率，而是应该对很少出现的特征执行较大的学习率；
- 最小化神经网络常见的高度非凸误差函数的另一个关键挑战是避免陷入其众多次优局部最小值中。这种困难实际上不是来自局部最小值，而是来自鞍点，即一个维度向上倾斜而另一个维度向下倾斜的点。这些鞍点通常被具有相同误差的平台包围，这使得 SGD 难以逃脱，因为梯度在所有维度上都接近于零。
## 梯度下降优化算法
### 动量（Momentum）
**产生背景：** SGD算法在沟壑处（即曲面一个维度的陡峭程度远大于另一个维度）陷入困境。这在局部最小值附近很常见。该情况下，SGD在沟壑的斜坡上震荡，只是沿着底部向最优方向上缓慢前进。
![](OverviewNote_md_files/cad493d0-3c76-11ed-b19b-bd4b5976af6e.jpeg?v=1&type=image)   
Momentum能够帮助SGD在相关方向上加速收敛并减缓震荡。其在更新方向上加入了 $\lambda$ 倍的前一步方向。

$$ v_t=\lambda v_{t-1}+\eta \nabla_\theta J(\theta) $$

$$ \theta=\theta-v_t $$

其中动量项 $\gamma$ 通常取为 $0.9$或者是一个相似值。   
本质上，使用momentum时，小球在下山过程中积累动量，变得越来越快。直到其达到终止速度，如果有空气阻力即 $\gamma<1$ 。更新梯度时同样的事也会发生： 对于梯度指向方向相同的维度，动量项会增加；对于梯度改变的方向，动量项的更新会减少。结果，我们获得了更快的收敛速度并减少了震荡。
### 涅斯捷罗夫加速梯度（Nesterov accelerate gradient，NAG）
小球若从斜坡上盲目地滚下来是很不令人满意的。我们想找到一个聪明的小球，有意识自己将去往何方，因此他也可以在斜坡升高前减速。   
NAG建立在Momentum的基础上，给小球这样的先见之明。已知我们将使用动量 $\gamma v_{t-1}$ 来移动参数 $\theta$ ，通过计算 $\theta-\gamma v_{t-1}$ 给定下辖一个参数位置的估计（全部更新时梯度缺失），这是关于参数将来位置的粗略概念。我们现在可以通过计算**近似的未来参数位置**而不是当前梯度位置来计算参数的梯度来有效地获得先见之明。

$$v_t=\gamma v_{t-1}+\eta \nabla_\theta  J(\theta-\gamma v_{t-1})$$ $$\theta=\theta-v_t$$

![](OverviewNote_md_files/886cd830-3cc2-11ed-b19b-bd4b5976af6e.jpeg?v=1&type=image)
现在积累梯度方向进行大跳跃，再计算在下一步梯度方向。二者矢量和作为向量更新方向。
这种预期更新可以防止我们走得太快并导致响应速度提高，这显著提高了RNNs在许多任务上的性能。   
既然我们能够使我们的更新适应我们的误差函数的斜率并反过来加速 SGD，我们还希望我们的更新适应每个单独的参数，以根据它们的重要性执行更大或更小的更新。   
### Adagrad
- **Adagrad使得学习率适应参数**。对不常出现的参数执行较大更新；对频繁出现的参数执行较小更新；
- Adagrd适合处理稀疏数据。

鉴于Adagrad在每次时间步长 $t$ 对每个参数 $\theta_i$ 执行更新，方便起见，用 $g_{t,i}$ 表示目标方程在时间 $t$ 时对参数 $\theta_i$ 的参数：

$$g_{t,i}=\nabla_{\theta_t} J(\theta_{t,i})$$

SDG在时间 $t$ 处对每个参数 $\theta_i$ 执行更新变为：

$$\theta_{t+1,i}=\theta_{t,i}-\eta\cdot g_{t,i}$$

在其更新规则中，Adagrad根据每个 $\theta_i$ 的梯度为每个 $\theta_i$ 调整一般的学习率 $\eta$ ：

$$\theta_{t+1,i}=\theta_{t,i}-\frac{\eta}{\sqrt{G_{t,i}+\epsilon}}\cdot g_{t,i}$$

其中 $G_t \in R^{d\times d}$ 是一个对角矩阵，对角线上元素是 $0-t$ 时刻关于 $\theta_i$ 的梯度的平方和； $\epsilon$ 是一个平滑项，防止分子为0（通常取为 $1e-8$ 左右。
> 注：平方根是必不可少的，否则算法的表现会差很多。
 
矩阵形式：

$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t$$

其中 $\odot$ 表示对应元素相乘。
- 优点：Adagrad实现了自动调节学习率，一般初始值设置为 $0.01$ .
- 缺点：Adagrad 的主要弱点是它在分母中累积平方梯度：由于每个添加的项都是正数，因此累积和在训练期间不断增长。这反过来会导致学习率缩小并最终变得无限小，此时算法不再能够获得额外的知识。
### Adadelta
Adadelta是Adagrad的改进。它旨在降低激进的、单调递减的学习率。Adadelta不是累计过去所有的梯度平方和，而是将累计过去梯度的窗口限制在一个固定大小 $w$ .
**计算方式**：不是低效地存储 $w$ 个先前的平方梯度，梯度和递归地定义为过去所有梯度平方的衰减平均值。则 $t$ 时刻的学习率 $\eta$ 仅取决于梯度均值和当前的向量:

$$E\[ g^2 \]_t=\gamma E\[ g^2 \]_{t-1}+(1-\gamma)g_t^2$$

其中 $\gamma$ 取值在  $0.9$ 左右，则更新规则可写为：

$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_t$$

$$\theta_{t+1}=\theta_t-\frac{\eta}{RMS \ g \_t}\odot g_t$$

注意，其中：

$$\triangle\theta_t=-\frac{\eta}{\sqrt{G_t+\epsilon}}\odot g_ t=-\frac{\eta}{RMS[g]_t}\odot g_ t$$     

作者指出，此更新（以及 SGD、Momentum 或 Adagrad）中的单位不匹配，即更新应该具有与参数相同的假设单位。为了实现这一点，他们首先定义了另一个指数衰减平均值，这一次不是平方梯度，而是平方参数更新：

$$ E[\triangle \theta^2]_ t=\gamma {E[\triangle \theta^2]}_ {t-1} +(1-\gamma)\triangle\theta_ t^2 $$

因此，参数更新的均方根误差为：

$$ RMS[\triangle \theta]_ t=\sqrt{{E[\triangle\theta^2]}_ t+\epsilon} $$    

由于 ${RMS[\triangle\theta]}_ t$ 未知，使用 $RMS[\triangle\theta]_ {t-1}$ 近似代替。用 $RMS[\triangle\theta]_ {t-1}$ 代替 $\eta$ ，则Adadelta更新规则为：

$$\triangle\theta_t=-\frac{RMS[\triangle\theta]_ {t-1}}{RMS[g]_ t} g_t$$

$$\theta_{t+1}=\theta_t+\triangle\theta_t$$

使用Adadelta我们甚至无需设置一个初始学习率，因为他已经在参数更新过程中被删除。
### RMSprop
RMSprop是一个未发表的自适应学习率方法。是Geoff Hinton在他的[课程](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)中提出来的。
RMSprop 和 Adadelta 都是在同一时间独立开发的，因为需要解决 Adagrad 急剧下降的学习率问题。RMSprop 实际上与我们上面推导出的 Adadelta 的第一个更新向量相同，更新规则：

$$E[g^2]_ t=0.9E[g^2]_ {t-1}+0.1g_ t^2$$

$$\theta_ {t+1}=\theta_ t-\frac{\eta}{\sqrt{E[g^2]_ t+\epsilon}} g_ t$$
 
RMSprop页将学习率除以平方梯度指数衰减的平均值。Hinton建议： $\gamma=0.9$ , $\eta=0.001$.
### Adam
Adam（Adaptive Moment Estimation）自适应矩估计对每一个参数计算自适应学习率。除了像Adadelta和RMSprop一样存储过去梯度指数递减的平均平方值 $v_t$ ，Adam还存储过去梯度指数递减的平均值 $m_t$ ，和动量类似.

$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$$

$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

 -  $m_t$ 和 $v_t$ 分别是梯度的一阶矩（均值）和二阶矩（非中心方差）的估计值，因此得名。由于 $m_t$ 和 $v_t$ 被初始化为 0 的向量，Adam 的作者观察到它们偏向于零，尤其是在初始时间步长期间，尤其是当衰减率较小时（即 $β_1$ 和 $β_2$ 接近 1）。
- 通过计算偏差校正的一阶和二阶矩估计来抵消这些偏差，校正方法如下：

$$\hat{m}_t=\frac{m_t}{1-\beta_1^t}$$

$$\hat{v_t}=\frac{v_t}{1-\beta_2^t}$$

得到Adam的参数更新方程：

$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}$$

默认值： $\beta_1=0,9$ , $\beta_2=0.999$ , $\epsilon=10^{-8}$ 。Adam算法在自适应学习中表现良好且和其他算法相比具有优势。
### AdaMax
在Adam中，因子 $v_t$ 使得梯度和过去梯度的 $l_2$ 范数以及当前梯度 $|g_t|^2$ 成反比：

$$v_t=\beta_2v_{t-1}+(1-\beta_2)|g_t|^2$$

推广到 $l_p$ 范数（Kingma和Ba也将 $\beta_2$ 参数化为 $\beta_2^p$ ）：

$$v_t=\beta_2^pv_{t-1}+(1-\beta_2^p)|g_t|^p$$

**范数选择：**
-  $p$ 较大时算法逐渐变得数值不稳定，因此 $l_ 1$ 和 $l_ 2$ 范数最为常用；
- 但 $l_ \infty$ 有不错的稳定性。

 使用AdaMax算法 $l_ \infty$ 收敛到更稳定的值。为避免和Adam算法的混淆，我们使用 $u_t$ 表示无穷范数约束 $v_t$  :

$$u_t=\beta_ 2^\infty v_ {t-1}+(1-\beta_ 2^\infty)|g_t|^\infty=max(\beta_ 2\cdot v_ {t-1},|g_ t|)$$

在 $\sqrt{\hat{v_ t}}+\epsilon$ 用 $u_ t$ 代替 $v_ t$ ，得到AdaMax的迭代规则：

$$\theta_ {t+1}=\theta_ t-\frac{\eta}{u_ t}\hat{m_ t}$$

**注意** ：由于 $u_ t$ 依赖于最大算子，因此这里不像Adam中的 $m_ t$ 和 $v_ t$ 那样倾向于零。因此不需要为 $u_ t$ 计算偏差校正。
**良好的默认值**：是 $\eta = 0.002$ ，$\beta_1 = 0.9$ ，$\beta_2 = 0.999$.

### Nadam
> 背景：Adam可以看成RMSprop和monmentum的结合：RMSprop提供了过去梯度平方 $v_ t$ 的指数级递减；momentum贡献过去梯度平均 $m_ t$ 的指数级衰减。
> **注意**：Nesterov加速算法（NAG）优于一般的动量算法。

Nadam（Nesterov-accelerated Adaptive Moment Estimation，加速自适应矩估计）结合了Adam和NAG。**为结合NAG我们需要修改其动量项 $m_ t$.**
**回顾Momentum更新规则** ：

$$g_ t=\nabla_ {\theta_ t} J(\theta_ t)$$

$$m_ t=\gamma m_ {t-1}+\eta g_ t$$

$$\theta_ {t+1}=\theta_ t-m_ t$$

其中 $J$ 是目标方程， $\eta$ 是动量衰减项， $\eta$ 是步长。上面的第三项扩展为：

$$\theta_ {t+1}=\theta_ t-(\gamma m_ {t-1}+\eta g_ t)$$

这再次证明了动量算法更新方向包含了过去的动量方向以及当前的梯度方向。
**回顾NAG更新原则：**
NAG通过利用当前动量近似计算下一步梯度，使我们能够找到更准确地梯度方向：

$$g_ t=\nabla_ {\theta_ t} J(\theta_ t-\gamma m_ {t-1})$$

$$m_ t=\gamma m_ {t-1}+\eta g_ t$$

$$\theta_ {t+1}=\theta_ t-m_ t$$

Dozat建议：与其使用两次动量（1.近似计算更新下一步梯度；2.更新参数），不如应用前瞻动量直接更新参数（即把前瞻梯度换成前瞻动量）：

$$g_ t=\nabla_ {\theta_ t} J(\theta_ t)$$

$$m_ t= \gamma m_ {t-1}+\eta g_ t$$

$$\theta_ {t+1}=\theta_ t-(\gamma m_ t+\eta g_ t)$$

回顾Adam：

$$m_ t=\beta_ 1 m_ {t-1}+(1-\beta_ 1)g_ t$$

$$\hat{m_ t}=\frac{m_ t}{1-\beta_ 1^t}$$

$$\theta_ {t+1}=\theta_ t-\frac{\eta}{\sqrt{\hat{v_ t}}+\epsilon} \hat{m_ t}$$

展开得：

$$\theta_ {t+1}=\theta_ t-\frac{\eta}{\sqrt{\hat{v_ t}}+\epsilon}(\frac{\beta_ 1 m_ {t-1}}{1-\beta_1^t} +\frac{(1-\beta_ 1)g_ t}{1-\beta^t_ 1})$$

注意到 $\frac{\beta_ 1 m_ {t-1}}{1-\beta_ 1^t}$ 是上一步的动量的偏差矫正估计，使用 $\hat{m}_ {t-1}$ 替代可得：

$$\theta_ {t+1}=\theta_ t-\frac{\eta}{\sqrt{\hat{v_ t}}+\epsilon}(\beta_ 1 \hat{m}_ {t-1} +\frac{(1-\beta_ 1)g_ t}{1-\beta^t_ 1})$$

添加Nesterov动量项，将上一步动量的偏差校正估计 $\hat{m_ {t-1}}$ 换成当前的动量偏差矫正 $\hat{m_ t}$ ，得到Nadam更新规则：

$$\theta_ {t+1}=\theta_ t-\frac{\eta}{\sqrt{\hat{v_ t}}+\epsilon}(\beta_ 1 \hat{m}_ {t} +\frac{(1-\beta_ 1)g_ t}{1-\beta^t_ 1})$$

### 算法可视化
