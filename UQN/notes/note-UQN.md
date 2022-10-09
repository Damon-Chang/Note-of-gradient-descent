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

$$g_ t'=\gamma \cdot g_{t-1}'+M_ t''$$

$$g_ t=g_ t'+M_ t'$$

其中 $g_ 0=\nabla L(\theta_ 0)$ ， $\theta_ 0$ 是初始定义的参数。图示如下所示：

![梯度的更新图解](https://github.com/Damon-Chang/Note-of-gradient-descent/blob/main/UQN/notes/pics/37c89350-421d-11ed-bff7-65292cfeafc0.jpeg)

将上面的公式展开有： 

$$g_ t'=\gamma^t\cdot\nabla L(\theta_0)+\sum_ {i=0}^{t-1} \gamma^i \cdot M_{t-i}''$$

其中为了更一般的表示，令 

$$M_ t'=\omega'\cdot\nabla L(f'(t))$$

$$M_ t''=\omega''\cdot\nabla L(f''(t))$$

其中 $\omega'$ , $\omega''\in\mathcal{Q}^+$ ，$f'(t)$ 和 $f''(t)$ 是 $t$ 的函数。**不同的优化器具有不同的 $\omega',\omega'',f'(t),f''(t)$** ，例如：   
令 $\omega'=\beta,\omega''=1,f'(t)=\theta_t,f''(t)=\theta_t$ 就得到了QHM：

$$g_ t'=\gamma \cdot g_ {t-1}'+\nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t)$$

令 $\omega'=0,\omega''=1,f'(t)=0,f''(t)=\theta_t-\gamma\cdot\eta g_ {t-1}'$ 就得到NAG：

$$g_ t=\gamma \cdot g_{t-1}+\nabla L(\theta_t')$$

$$\theta_ {t+1}=\theta_ t-\eta\cdot g_ t$$

#### $\eta_ t$ 调整规则
在 MBOs中引入学习率 $\eta_ t$ 调整规则，归纳为下面的形式：

$$v_ t'=\lambda\cdot v_ {t-1}'+(N_ t'')^2$$

$$v_ t=v_ t'+(N_ t')^2$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

其中： $\epsilon$ 是一个极小的正数，保证了分母不等于0， $v_ t$ 是梯度平方的指数衰减平均值。 $\lambda\in [0,1]$ 且 $\lambda$ 接近1。不同的优化器有不同的 $N_ t'$ 和 $N_ t''$ 。例如：    
取 $N_ t''=\nabla L(\theta_ t)$ ， $N_ t''=\beta \cdot \nabla L(\theta_ t)$ ，就得到了QHAdam的学习率更新规则：

$$v_ t'=\lambda \cdot v_ {t-1}'+\nabla L^2(\theta_ t)$$

$$v_ t=v_ t'+\beta^2\cdot \nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

为了更清楚地显示优化器之间的差异，我们对面优化器的更新规则进行了转换。将其转换为上述的一般形式公式。其变量值如下表所示。   

![不同的优化器一般形式下的区别](https://github.com/Damon-Chang/Note-of-gradient-descent/blob/main/UQN/notes/pics/f82f3460-4230-11ed-bff7-65292cfeafc0.jpeg)

## 基于MBOs的无偏策略
1. 说明广泛使用的MBOs是有偏差的，即小球下降方向和最速下降方向（ $\nabla L(\theta_ t)$ 的方向）不一致；
2. 理想的优化器需要与最速下降方向一致，这样一来优化器可以快速下降并收敛到极小值；
3. 一种无偏策略（Unbiased  Strategy）使得 $g_ t$ 的期望和 $\nabla L(\theta_ t)$ 相等；
4. 最终将该无偏策略应用到Momentum，QHM，NAG。
### Consistency between $g_ t$ 和 $\nabla L(\theta_ t)$ 之间的一致性
 $g_ t$ 和 $\nabla L(\theta_ t)$ 方向是否一致的充要条件（necessary and sufficient condition）
##### **Thm 1** MBOs的方向 $g_ t$ 和梯度方向 $\nabla L(\theta_ t)$ 一致，当且仅当 $\omega''\cdot \sum_{i=0}^{t} \gamma^i +\omega'=1$ . 其中 $\omega'$ 和 $\omega''$ 在梯度更新公式中为 $g_ t'=\gamma \cdot g_{t-1}'+M_ t''$ ， $g_ t=g_ t'+M_ t'$ .
*proof.* 将 $g_ t'$ 和 $g_ t$ 展开:

$$g_ 0'=M_ 0'', g_ 0=M_ 0''+M_ 0';$

$$g_ 1'=\gamma\cdot M_ 0''+M_ 1'', g_ 1=\gamma\cdot M_ 0''+M_ 1''+M_ 1';$$

$$......$$

$$g_ t'=\sum_ {i=0}^{t} \gamma^i\cdot M_ {t-i}'', g_ t=\sum_ {t=0}^{t}\gamma^i\cdot M_ {t-1}''+M_ t'=\omega''\cdot \sum_ {i=0}^{t}\gamma^i\cdot \nabla L(f''(t))+\omega'\cdot\nabla L(f'(t)).$$

当 $\omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'=1$ 时，期望 $E[g_ t]=\nabla L(\theta_ t)$ ，因此此时 $g_ t$ 是梯度 $\nabla L(\theta_ t)$ 的无偏估计。

##### **引理1** MBOs方法中的梯度方向 $g_ t$ 和最快下降方向 $\nabla L(\theta_ t)$ 不一致。
*proof：*  **Momentum：** $\omega'=0,\omega''=1$ ，则 $\omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'=\lim_ {t\to\infty}\sum_{i=0}^{t} \gamma^i \neq1$   
**QHM:** $\omega'=\beta,\omega''=1$ ，则 $\omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'=\lim_ {t\to\infty}\sum_{i=0}^{t} \gamma^i+\beta \neq1$      
 **NAG:** $\omega'=0,\omega''=1$ ，则 $\omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'=\lim_ {t\to\infty}\sum_{i=0}^{t} \gamma^i \neq1$      
 而Adam是基于Momentum的算法，只改变了学习率，QGAdam，Nadam，AdaMax都是基于Adam的算法。因此这些方法也继承了同样的缺点。
### 无偏策略
 - 策略：在 $\omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'$ 中调整 $\omega''$ 使得 $\omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'=1$ ，即 $g_ t=\nabla L(\theta_ t)$ .
 - 具体方法：
将 $\omega''$ 调整为 $\Omega''$ ，使得 $\Omega''\cdot \sum_{i=0}^{t} \gamma^i+\omega'=1$ 即 $\Omega''\cdot \sum_{i=0}^{t} \gamma^i=1-\omega'$ ，两边同乘 $(1-\gamma)$ :

$$(1-\gamma)\cdot\Omega''\cdot \sum_{i=0}^{t} \gamma^i=(1-\gamma)(1-\omega')$$

因为当 $t$ 足够大时 $(1-\gamma)\cdot \sum_{i=0}^{t} \gamma^i=1(1-\gamma^{t+1}\to1,t\to\infty)$ ，因此

$$\Omega''=(1-\gamma)\cdot (1-\omega')$$

因此，基于梯度更新的式子：

$$g_ t'=\gamma \cdot g_{t-1}'+M_ t''$$

$$g_ t=g_ t'+M_ t'$$

做调整 $M_ t''=\omega''\cdot\nabla L(f''(t))$ 变为 $M_ t''=\Omega''\cdot\nabla L(f''(t))=\frac{1-\omega'}{\omega''}\cdot (1-\gamma)\cdot \omega''\cdot \nabla L(f''(t))=\frac{1-\omega'}{\omega''}\cdot (1-\gamma)\cdot M_ t''$ . 因此 $g_ t$ 无偏策略的调整规则为：

$$g_ t'=\gamma\cdot g_ {t-1}'+\frac{1-\omega'}{\omega''}\cdot (1-\gamma)\cdot M_ t''$$

$$g_ t=g_ t'+M_ t'$$

其中 $0<\gamma<1$ .   

下面将无偏策略应用到无自适应学习率的 MBOs，即QNM，NAG。    
**我们不将无偏策略应用于采用自适应学习率的Adam，QHAdam，Nadam和AdaMax。**自适应学习率MBOs在不改变 $g_ t$ 的情况下调整 $\eta_t$ 。同时，避免自适应学习率对无偏策略的影响，更好地研究无偏策略的改进效果。

#### 无偏动量（U-Momentum）
 $g_ t$ 的更新策略更改为：

$$g_ t=g_ t'=\gamma\cdot g_ {t-1}'+(1-\gamma)\cdot \nabla L(\theta_ t)$$

原来 $\omega'=0, \omega''=1$其中根据无偏策略做出的改动为： $\omega'=0, \omega''=(1-\gamma)$ 
#### 无偏QHM（U-QHM）
 $g_ t$ 更新策略改为

$$g_ t'=\gamma\cdot g_ {t-1}'+(1-\beta)(1-\gamma)\nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta \cdot \nabla L(\theta_ t)$$

原来 $\omega''=1, \omega''=\beta$ 改为 $\omega''=(1-\beta)(1-\gamma), \omega'=\beta$ .
#### 无偏NAG
  $g_ t$ 更新策略改为

$$g_ t=\gamma\cdot g_ {t-1}'+(1-\gamma)\cdot \nabla L(\theta_ t')$$

原来 $\omega''=1, \omega'=0$ 变为 $\omega''=(1-\gamma), \omega'=0$ .
## UQN:结合U-QHM和U-NAG
UQN:加速收敛+统一 $g_ t$ 和 $\nabla L(\theta_ t)$ 方向+降低梯度方差。
### 非自适应学习率情况下使用UQN下MBOs的梯度方差分析
UQN-Momentum的更新规则：

$$\eta_ t=\eta$$

$$g_ t'=\gamma\cdot g_ t'+(1-\gamma)\cdot(1-\beta)\cdot\nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta-\gamma\cdot\eta\cdot g_ {t-1}')$$

$$\theta_ {t+1}=\theta-\eta_ t\cdot g_ t$$

##### Thm 2：$\lim_ {t\to\infty} Variance(UQN-Momentum)=\alpha\cdot\Sigma$ 其中 $\Sigma$ 是Momentum的梯度方差， $\alpha=\frac{2}{1+\gamma}\cdot \beta_ 2-2\frac{1-\gamma}{1+\gamma}\cdot\beta+\frac{1-\gamma}{1+\gamma}$ .
*proof* 将上面式子中的 $g_ t$ 展开可得

$$g_ t=(1-\beta)\cdot(1-\gamma)\cdot\gamma_ t\cdot\nabla L(\theta_ t)+……$$

$$+(1-\beta)\cdot(1-\gamma)\cdot\gamma_ 0\cdot\nabla L(\theta_ t)+\beta\nabla L(\theta_ {t+1})$$
 
假设 $\nabla L(\theta_ {t+1-i})$ 是独立同分布的随机向量。 $\nabla L(\theta_ {t+1-i})$ 的系数 $\delta_ i$ 是

$$ \delta_ i=\left\{ \begin{aligned} 
\beta & & {i=0} \\
(1-\beta)\cdot(1-\gamma)\cdot\gamma_ {i-1} & & {i=1,……,t+1}\\
\end{aligned} \right. $$

$${\lim_ {t\to\infty}}^2 Variance(UQN-Momentum)=\lim_ {t\to\infty} \sum_ {i=0}^{t+1} \delta_ i^2\cdot \Sigma$$

其中

$$\lim_ {t\to\infty} \sum_ {i=0}^{t+1} \delta_ i^2=\beta^2+(1-\beta)^2\cdot\frac{1-\gamma}{1+\gamma}$$

$$=\frac{2}{1+\gamma}\cdot \beta^2-2\frac{1-\gamma}{1+\gamma}\cdot\beta+\frac{1-\gamma}{1+\gamma}$$

因此 $\alpha=\lim_ {t\to\infty}\delta_ i^2=\frac{2}{1+\gamma}\cdot \beta_ 2-2\frac{1-\gamma}{1+\gamma}\cdot\beta+\frac{1-\gamma}{1+\gamma}$ .

由该定理可见，当 $\gamma,\beta\in(0,1)$ 时 $\alpha<1$ ，即UQN-Momentum相比于Monmentum降低了梯度方差。同理，利用上面的证明过程也可以说明，对于其他MBOs，UQN方法同昂可以降低梯度方差。

### 使用UQN改进具有自适应学习率的MBOs
将带有自适应学习率的MBOs（即Adam，QHAdam，Nadam，AdaMax）和UQN结合。一般形式的更新规则如下

$$v_ t'=\lambda\cdot v_ {t-1}'+(N_ t'')^2$$

$$v_ t=v_ t'+\beta^2\cdot\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma\cdot g_ {t-1}'+\frac{1-\beta}{\omega''}\cdot(1-\gamma)\cdot M_ t''$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t')$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

其中 $\theta_ t'=\theta_ t-\gamma\cdot\eta\cdot g_ {t-1}'$ .
#### UQN-Adam和UQN-QHAdam
 
$$v_ t'=\lambda\cdot v_ {t-1}'+\nabla^2L(\theta_ t)$$

$$v_ t=v_ t'+\beta^2\cdot\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma\cdot g_ {t-1}'+(1-\beta)\cdot(1-\gamma)\cdot \nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t')$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

#### UQN-Nadam
 
$$v_ t'=\lambda\cdot v_ {t-1}'+\nabla^2L(\theta_ t)$$

$$v_ t=v_ t'+\beta^2\cdot\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma\cdot g_ {t-1}'+(1-\beta)\cdot(1-\gamma)\cdot \nabla L(\theta_ t')$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t')$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

### UQN-Nadam
 
$$v_ t'=\lambda\cdot v_ {t-1}'+\nabla^2L(\theta_ t)$$

$$v_ t=v_ t'+\beta^2\cdot\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma\cdot g_ {t-1}'+(1-\beta)\cdot(1-\gamma)\cdot \nabla L(\theta_ t')$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t')$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

### UQN-AdaMax
 
$$v_ t'=\lambda\cdot v_ {t-1}'+\nabla^2L(\theta_ t)^\infty$$

$$v_ t=v_ t'+\beta^2\cdot\nabla L^2(\theta_ t)$$

$$\eta_ t=\frac{\eta}{\sqrt{v_ t}+\epsilon}$$

$$g_ t'=\gamma\cdot g_ {t-1}'+(1-\beta)\cdot(1-\gamma)\cdot \nabla L(\theta_ t)$$

$$g_ t=g_ t'+\beta\cdot\nabla L(\theta_ t')$$

$$\theta_ {t+1}=\theta_ t-\eta_ t\cdot g_ t$$

## 收敛性分析
##### Thm 3 用无偏策略改进的MBOs加速收敛策略收敛，当且仅当

$$k_ i<\tau\cdot\frac{\mu^2}{2m}$$

其中$\tau=1-\gamma\cdot(1-\omega')$ ，m是粒子的质量，µ 是牛顿力场的摩擦系数。
*proof* 因为Momentum就是基于物理中的动量的概念进行的梯度下降改进，这个定理也是关于物理中动力学的内容。 
##### 引理 2 使用UQN改善的MBOs策略加速收敛，当且仅当

$$k_ i<\tau\cdot\frac{\mu^2}{2m}$$

其中 $\tau=1-\gamma\cdot(1-\beta)$ .






