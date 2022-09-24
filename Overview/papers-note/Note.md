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
计算$\nabla J(\theta)$时不使用全部数据，分次（**epoch**）选择一定批量的样本计算。批量梯度下降保证了$J(\theta)$在凸函数时收敛到全局极小值点（**global minimum**），在非凸问题上收敛到局部极小值点（**local minimum**）。
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
- 学习率时间表尝试在训练过程中调整学习率，
- 
