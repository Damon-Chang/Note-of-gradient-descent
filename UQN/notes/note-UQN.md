# 文献阅读笔记
# Unbiased Quasi-hyperbolic Nesterov-gradient Momentum-based Optimizers for Accelerating Convergence
> 作者：Weiwei Cheng1, Xiaochun Yang1*, Bin Wang1 and Wei Wang2,
> 单位：Northeastern University
## Abstract
**动量优化器（Momentum-Based Optimizers，MBOs）：**通过使用梯度的指数衰减平均值**微调节**方向加速收敛：

$$ v_ t=\gamma v_ {t-1}+\eta \nabla_ \theta J(\theta)$$

$$\theta=\theta-v_ t$$

**存在问题**：大部分MBOs是有偏的，即梯度下降方向和最快下降方向不一致。   
**导致问题**：迭代次数变多，成本变高。   
**改进策略**：采用无偏策略调整诸多MBOs的下降方向。    
**具体方法**：   
- 提出**无偏准双曲Nesterov梯度策略（Unbiased Quasi-hyperbolic Nesterov-gradient strategy，UQN）**：将我们的无偏策略和现有的准双曲和Nesterov梯度结合。
- 效果：
	- 每一步迭代都向着局部最快下降方向；
	- 预测未来梯度以避免越过极小值点；
	- 降低梯度方差。

**实验**：将无偏策略应用到许多MBOs上。   

## 缩写说明
- MBOs：Momentum-Based Optimizers，基于动量的优化器；
- GD：Gradient Descent，梯度下降；
- QHM：Quasi-Hyperbolic Momentum，拟双曲动量；
- NAG：Nesterov Accelerated Gradient，Nesterov加速梯度；
- U-QHM：Unbiased Quasi-Hyperbolic Momentum；
- U-NAG：Unbiased Nesterov Accelerated Gradient；
- UQN：Unbiased Quasi-hyperbolic Nesterov-gradient Momentum-based Optimizers，基于向量的拟双曲Nesterov梯度优化器；
- 
- 
- 


## 简介
**两种经典的MBOs：**
- **QHM（Quasi-Hyperbolic Momentum）**：拟双曲动量，增加当前梯度的权重可以有效减少梯度方差，并基于方差减少的非正式和推测的动机加速收敛；
- **NAG（Nesterov Accelerated Gradient）：**Nesterov加速梯度，通过添加Nesterov梯度（即未来梯度）可以加速收敛。直觉上，是小球足够聪明，在曲面提高之前就进行减速。
> 注：未来梯度（feature gradient）能够预知小球未来的方向，使得这个小球足够聪明以避免越过极小值点。

**加入无偏策略：**   
将无偏策略（unbiased strategy）加入QHM和NAG中得到U-QHM和U-NAG。再将U-QHM和U-NAG结合得到UQN，UQN策略统一了速度方向和斜率方向，减少了梯度方差并防止球越过最小点。
**本文的主要贡献如下:**   
- 对MBOs：分析小球的速度方向，并发现其方向和局部最快下降方向不一致，因此我们提出了一种无偏策略使得他们一致，并加速收敛；
- 对于收敛速度：为加速收敛，将U-QHM和U-NAG结合得到UQN——结合了无偏策略、QHM、NAG的优点，并且可以推广到更多的MBOs之中；
- 对比试验。

**文章内容：**
- Section 2：列举广泛使用的MBOs的更新规则，描述MBOs更新规则的一般形式；
- Section 3：主要介绍调整速度方向的**无偏策略**，并给出三个Momentum方法的例子；
- Secion 4：介绍**UQN策略**，并将其应用到一些广泛使用的MBOs中；
- Section 5：证明无偏策略和UQN的收敛性；
- Section 6：进行无偏策略和UQN的超参数扫描实验和普遍性核实实验；
- Section 7：总结和展望。

## 准备工作——MBOs 
广泛使用的MBOs主要有：Momentum，QHM，NAG，Adam，QHAdam，Nadam，AdaMax。QHM提高当前梯度的权重，NAG引入未来梯度。这两个方法导致梯度方差下降，避免了越过最值点。Adam使用自适应学习率自动调节学习率。Nadam=Adam+NAG，QHAdam=Adam+QHM。AdaMax在Adam基础上考虑无穷范数以改进策略。
### Momentum：梯度改变相同：动量增加；梯度改变相反：动量减少。
- 改进：改善了当 $\nabla L(\theta_ t)$ 接近0时小球速度会很小的的情况。
- 方法：使用学习率 $\eta$ 和梯度 $g_ t$ 的指数衰减平均得到新参数 $\theta_ {t+1}$ .这里 $g_ t$ 仅取决于过去的梯度向量 $g_ t$ 和当前梯度 $\nabla L(\theta_ t)$.
- 更新规则：

$$g_ t=\gamma\cdot g_ {t-1}+\nabla L(\theta_ t)$$

$$\theta_ {t+1}=\theta_ t-\eta \cdot g_ t$$

- 初始化梯度 $g_ 0=\nabla L(\theta_ 0)$ ， $\gamma\in[0,1]$;
- 一般认为$\gamma=0.9$ ，说明 $g_ {t-1}$ 对 $g_ t$ 起主导作用。 
### QHM
> 提出QHM是未来降低减低梯度方差，而且已被证实对于降低迭代次数是一个高效的策略。

本质上讲，QHM取 $\nabla L(\theta_ t)$ 和 $g_ t'$ 的加权和，这相当于增加了 $\nabla L(\theta_ t)$ 的权重，更新规则如下：

$$g_ t'=\gamma \cdot g_ {t-1}'+\nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t)$$

$$\theta_ {t+1}=\theta_ t-\eta \cdot g_ t$$

QHM首先将利用 $\nabla L(\theta_ t)$ 和 $g_ {t-1}'$ 参数 $\theta_ t$ 更新到中间参数 $\theta_ t^M$ ，再利用 $\nabla L(\theta_ t)$ 将参数更新到 $\theta_ {t+1}$ .

### NAG
> 用于解决动量算法中小球会越过极小值点的问题。NAG方法通过增加Nesterov梯度增强小球的预测能力——当小球预测到将越过极小值点时小球会慢下来并收敛到极小值点。

**更新规则：**   

$$g_ t=\gamma \cdot g_{t-1}+\nabla L(\theta_t')$$

$$\theta_ {t+1}=\theta_ t-\eta\cdot g_ t$$

其中将来近似位置利用动量得到： $\theta_ t'=\theta_ t-\gamma\cdot\eta\cdot g_ {t-1}$ .NAG首先通过上一步方向 $g_ t$ 和动量项 $\gamma$ 和学习率 $\eta$ 得到未来近似位置 $\theta_ t'$ ，再利用梯度 $\nabla L(\theta_t')$ 更新到 $\theta_ {t+1}$.

### Adam
> Adam(Adaptive Moment Estimation，自适应矩估计)，是自适应学习率的MBO，基于动量。Adam使得梯度呈指数衰减，因为Momentum中的学习率是固定的，因此对于不同的任务需要很多时间和精力去重新设定超参数使其适应问题。    
> Adam通过梯度 $v_ t$ 平方的指数平均收敛来自动调整学习率 $\eta_ t$ 。    
> 这样一来，对于不常出现的特征参数执行较大更新，对于出现较频繁的特征参数执行较小更新。因为不常出现的特征其参数多为0，因此梯度平方和更小，倒数更大，更新就更大。反之，频繁出现的特征更新就更小。

**更新规则：**

$$v_ t=\lambda \cdot v_ {t-1}+\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t=\gamma \cdot g_ {t-1}+\nabla L(\theta_ t)$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

其中 $\lambda=0.999$ .
### QHAdam
> QHAdam结合了QHM和Adam。降低梯度方差并加速收敛。QHAdam通过提高 $\nabla L(\theta_ t)$ 的权重调整 $g_ t$ ，通过提高 $\nabla L^2(\theta_ t)$ 的权重调整 $\eta_ t$ 。

**更新规则：**

$$v_ t'=\lambda \cdot v_ {t-1}'+\nabla L^2(\theta_ t)$$

$$v_ t=v_ t'+\beta^2\cdot \nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma \cdot g_ {t-1}'+\nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta \cdot \nabla L(\theta_ t)$$

$$\theta_ {t+1}=\theta_ t-\eta_ t \cdot g_ t$$

### Nadam
> Nadam结合了NAG和Adam。Nadam在NAG中加入了自适应学习率 $\eta_ t$ 以适应不同任务。    
> Nadam在Adam中加入了未来梯度 $\nabla L(\theta_ t')$ 以提高小球的预测能力，以避免越过极小值点。

**更新规则：**

$$v_ t=\lambda \cdot v_ {t-1}+\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma \cdot g_ {t-1}'+\nabla L(\theta_ t')$$

$$g_ t=g_ t'+\beta \cdot \nabla L(\theta_ t)$$

$$\theta_ {t+1}=\theta_ t-\eta_ t \cdot g_ t$$

### AdaMax
> 由于Adam中两个范数在高维空间中不稳定。因此样本数据为高维时，Adam算法表现很不稳定，因此使用无穷范数 $|| \cdot||^ \infty$ 代替其中的范数。

**更新规则：**

$$v_ t=\lambda\cdot v_ {t-1}+|| \nabla L(\theta_ t) ||^\infty$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t=\gamma\cdot g_ {t-1}+\nabla L(\theta_ t)$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_t$$

### MBOs的一般更新格式：

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

-  $g_ t$ ：梯度指数衰减平均；
-  $\eta_ t$ ：学习率指数衰减平均。

####  $g_ t$ 调整规则
