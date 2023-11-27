# Variational Autoencoder

Original paper:

[Auto-encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)

Reference:

[Understanding Variational Autoencoders (VAEs) | by Joseph Rocca | Towards Data Science](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)



## Purpose

* A generation (decoder) model, parameterized by $\theta$, uses $Z$ to generate $X$:  $Z \to X$. The probability denoted as $p_\theta(z)p_\theta(x|z)$
*  We wanted to compute $p_\theta(z|x)$: $X\to Z$. This is good since $X$ is available for us.

## Assumption

$p_\theta(x|z) = \mathcal N(x;g(z),\sigma_g^2)$, i.e., following the Gaussian distribution.

$p(z) = \mathcal N(z;0, 1)$, i.e., the unit Gaussian.

## Approximation

* Let $q_\phi(z|x)$ to approximate $p_\theta(z|x)$, which also follows Gaussian distribution $\mathcal N(e_\mu(x), e_\sigma(x)^2)$, where $e_\mu, e_\sigma$ are neural networks and is differentiable.

* Observe the  max-likelihood formula:

  $\mathbb E_{X\sim D} \log p(X) \approx \sum_{i=1}^n \log p(X_i)$

  Here, we first notice that $q_\phi(Z|X)$ is an tractable approximation (i.e., can be estimated using a neural network), we rewrite $\log p(X)$ as:

  $\mathbb E_{q(Z|X)} \log p(X) = \mathbb E_{q(Z|X)} [\log p(X|Z) + \log p(Z) - \log p(Z|X)]$

  Which can further be written as

  $\mathbb E_{q(Z|X)} [\log p(X|Z) -\log q(Z|X) + \log q(Z|X) - \log p(Z) - \log p(Z|X)]$

  $=\text{KL}[q(Z|X)\Vert p(Z|X)] - \text{KL}[q(Z|X)\Vert p(Z)] + \mathbb E_{q(Z|X)} \log p(X|Z)$

* One important observation is that, the first term $\text{DL}[q(Z|X)\Vert p(Z|X)]$ is always non-negative hence the above equation is a **lower bound**.
*  Thus, we just have to optimize $\mathbb E_{q(Z|X)} \text{KL}[q(Z|X)\Vert p(Z)] + \mathbb E_{q(Z|X)} \log p(X|Z)$

Notice that $q(Z|X)$ is Gaussian parameterized by a neural network, and $p(Z) = \mathcal N(Z;0, 1)$, the first term is differentiable. Also, $p(X|Z)$ is also NN-parameterized Gaussian, hence is also differentiable.

As for the $\mathbb E_{q(Z|X)}$, it can be sampled given a specific $X = X_i$

Overall, the training objective becomes:

$\sum_{i=1}^n \mathbb E_{\epsilon\sim \mathcal N(0, 1)}  \dfrac{1}{2\sigma_g^2} \Vert g(e_\mu(x_i) + \epsilon e_\sigma(x_i)) - x_i\Vert^2 + \dfrac12  \sum [1 + \log (e_\sigma(x_i)^2) - e_\mu(x_i)^2 - e_\sigma(x_i)^2]$

Notice: Use $e_\mu(x) + \epsilon e_\sigma(x)$ to denote $Z' \sim \mathcal N(e_\mu(x), e_\sigma(x)^2)$ is called **reparameterization trick**.

It seems just like the vanilla autoencoder loss plus the regularization term, and with a sampling step.

## Understanding

* Use $q(z|x)$ to approximate $p(z|x)$
* The non-negative term $\text{DL}[q(z|x)\Vert p(z|x)]$ is emitted in optimization, as we only consider the lower bound.

# Deep Variational Information Bottleneck

Original paper:

[Deep Variational Information Bottleneck, ICLR 2017](https://arxiv.org/pdf/1612.00410.pdf)



## Purpose

* In a neural network: $X \to Z \to Y$, minimize the mutual information $I(Z, X)$, and maximize the mutual information $I(Z, Y)$. In this way, $Z$ is a good representation.
* However, the Markov chain here is considered to be $Y \leftrightarrow X\leftrightarrow Z$
* The target: $R_{IB}(\theta) = I(Z, Y|\theta) - \beta I(Z,X;\theta)$

## Assumption

* Use a neural network to encode $p_\theta(Z|X)$ ($\theta$ could be omitted), i.e., $e_\mu(x), e_\sigma(x)$ for the mean and variance 

  (Notice that here $e_\sigma(x)$ is a $d \times d$ matrix is the square root covariance matrix, unlike the element-wise case in VAE).

* The label probability $p(Y|X)$ is independent with $\theta$

## Approximation

* $I(Z, Y|\theta) = \int p(z, y) [\log p(z, y) - \log p(z) p(y)]dzdy$

  $=\int p(z,y) [\log p(y|z) - \log p(y)]dzdy$

  Considering that $p(y|z)$ is intractable (just like the VAE case!), using $q(z|y)$ to approximate it, then we can get a **lower bound**“

  $ = \int p(z, y) [\log q(z|y) - \log p(y)]dzdy$

  *********

  *Why $p(y|z)$ is intractable?* 

  This is because $p(z|x)$ is parameterized by $\theta$, and $p(y|x)$ is intractable (we don't know the actual probability), hence, $p(y|z) = p(y,z)/p(z) = \int_x p(z|x)p(y|x)p(x)dx / \int_x p(z|x) p(x)dx$ cannot be computed.

  *************
  
  Further we have
  
  $=\int p(z,y)\log q(y|z)dydz - \int_y\int_z p(z,y)dz p(y)dy = \int p(z,y)\log q(y|z) dydz - \int_y p(y)\log y dy$
  
  $= \int p(y,z)\log q(y|z) dydz + H(Y)$
  
  Here the $H(Y)$ is irrelevant to the optimization since it only depends on $Y$.
  
  Considering we have to sample $x$, rewrite the first term as:
  
  $\int_{x,y,z} p(x)p(y|x)p(z|x)\log q(y|z)dxdydz = \int_{x,y,z} p(x,y)p(z|x)\log q(y|z)dxdydz$
  
  This can be computed.
  
* $I(Z,X|\theta) = \int p(x,z)\log p(z|x)dxdz - \int p(z)\log p(z)dz$. However, since $\log p(z)$ is intractable, we again use a fixed normal distribution $\mathcal N(0, I)$ to approximate it. Now the formula can be written as

  $\int_{x,z} p(x,z)\log p(z|x) - \log r(z) dxdz = \text{KL}[p(Z|x) \Vert r(Z)]$

  This is the same as the VAE case, and can be easily computed.

# Denoising Diffusion Probabilistic Models

[Denoising Diffusion Probabilistic Models   NeurIPS 2020](https://arxiv.org/pdf/2006.11239.pdf)

## Purpose

* The denoising process: $x_T \to \cdots \to x_1 \to x_0$, where $p_\theta(x_t|x_{t + 1}) = \mathcal N(x_{t-1};  \mu_\theta(x_{t}, t), \Sigma_\theta(x_t, t))$

  Since we want to generate image from noise, it is natural to think the denoising process first.

* The reverse (noise-adding) process: $x_0 \to x_1 \to \cdots \to x_T$, where $q(x_{t+1}|x_t)$ is approximated by a linear network $\mathcal N(x_t|\sqrt{1-\beta_t} x_{t-1}, \beta_tI)$

**Notations**:

* $\beta_t$: scale of noise added in step $t$

* $\alpha_t = 1 - \beta_t$: the degree of shrinkage of $x_{t-1}$ in step $t$ (since we have to make sure $\mathbb \Vert x_t\Vert$ is constant)
* $\bar \alpha_t = \prod \alpha_t$: the portion of $x_0$ in time step $t$. In every step we only keep $\alpha_t$ portion of the previous $x$, hence the portion of $x_0$ is decreasing exponentially.

## Approximation

**Probability modelling**

$p_\theta(x_0, \cdots, x_T) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$  the denoising process

$q(x_1, \cdots, x_T|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$ the reverse (noise-adding) process

Combine those, we have:

$p(x_0) = \dfrac{p(x_0,\cdots, x_T)}{p(x_1,...,x_T|x_0)}$, then $\log p(x_0) = \log p(x_{0:T}) - \log p(x_{1:T}|x_0)$

Consider $\mathbb E_{q} \log q(x_{1:T}|x_0) - \log p(x_{1:T}|x_0)$ (where $\mathbb E_q$ means $x_1, x_2, ..., x_T |x_0 \sim q(x_1|x_0)q(x_2|x_1)\cdots q(x_T|x_{T-1})$)

Is the KL-divergence and is always positive, we cam have a upper bound of $p(x_0)$:

$\log p(x_0) \le \mathbb E_q (\log p(x_{0:T}) - \log q(x_{1:T}|x_0))$

Thus, it is trainable since $p(x_{0:T})$ can be computed by **neural networks** and $q$ is **closed form**.

**Choice of $\beta$**

The parameter for the denoising process $\beta_t$ can be either fixed and also trainable

**Noise-adding process fast sampling**

As $q(x_t|x_{t-1})$ is simply mixing Gaussian noise into $x_{t-1}$, we have

$q(x_t|x_0) = \mathcal N\left(x_t; \prod_{s=1}^t \sqrt{1-\beta_t} x_0, [1-\prod_{s=1}^t (1-\beta_t)] I)\right)$

$=\mathcal N\left(x_t; \sqrt{\bar\alpha_t}x_0, (1-\bar \alpha_t)I\right)$

* Notations: $\alpha = 1 - \beta$, $\bar \alpha_t = \prod \alpha_t$

**Further optimize the loss function**

Using $\mathbb E_q (\log p(x_{0:T}) - \log q(x_{1:T}|x_0))$ requires high-variance Monte-Carlo sampling, hence rewrite it as:

$\mathbb E_q\left( -[\log q(x_T|x_0) + \log q(x_{T-1} |x_T, x_0)  + \log q(x_{T-2}|x_{T-1}, x_T, x_0) + \cdots] + \log p(x_{0:T})\right)$

Notice that $q(x_{T-2}|x_{T-1},x_T,x_0) = q(x_{T-2}|x_{T-1}, x_0)$ since $x_T$ solely depends on $x_T$, the loss can further be converted to

$\mathbb E_q \left[ D_\text{KL} (q(x_T|x_0) \Vert p(x_T)) + \sum_{t=2}^T D_\text{KL}(q(x_{t-1}|x_t, x_0)\Vert p(x_{t-1}|x_t)) + \log p_\theta(x_0|x_1) \right]$

Here

* $L_T \equiv D_\text{KL} (q(x_T|x_0) \Vert p(x_T)) $
* $L_{t-1} \equiv D_\text{KL}(q(x_{t-1}|x_t, x_0)\Vert p(x_{t-1}|x_t))$
* $L_0 \equiv \log p_\theta(x_0|x_1)$

**Notice** that $q(x_{t-1}|x_t, x_0)$ is also a Gaussian distribution and has a closed form! We omit the detailed representation here.

See the original paper Sec. 2 end.

## Training

* $L_T$ contains no learnable parameters

* $L_{t-1}$: First, consider $\Sigma_\theta(x_t, t) = \sigma_t^2I$ as constant. ($\sigma_t^2=\beta_t$ or $\sigma_t^2 = \dfrac{1-\bar \alpha_{t-1}}{1-\bar \alpha_t}\beta$ are both OK)

  * $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar \alpha_t}\epsilon$ where $\epsilon  \sim \mathcal N(0, I)$ 
  * $L_{t-1} = \mathbb E_q \left[ \dfrac{1}{2\sigma_t^2}\Vert \mathbb Ex_{t-1} - \mu_\theta(x_t, t) \Vert^2\right]$ Remember that $q(x_{t-1}|x_t, x_0)$ is also Gaussian, and use the closed form formula, we can further have $\mathbb Ex_{t-1} =\dfrac{1}{\sqrt{\bar \alpha_t}}\tilde \mu(x_t, x_0)=  \dfrac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \dfrac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon\right)$ (*Note here we also use $x_t$ and a random Gaussian $\epsilon$* to represent $x_0$)
  * Thus, the problem becomes use $\mu_\theta(x_t, t)$ to predict $\dfrac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \dfrac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon\right)$
  * Then to learn $\mu_\theta(x_t, t)$ is to learn $\epsilon_\theta(x_t, t)$ to predict $\epsilon \sim \mathcal N(0, 1)$

* Simplified Loss:

  $L(\theta) = \mathbb E_{t,x_0,\epsilon}\left[ \Vert \epsilon - \epsilon_\theta(\sqrt{\bar \alpha }x_0 + \sqrt{1-\bar \alpha_t}\epsilon, t) \Vert^2 \right]$
  
  

### Backbone Model to approximate nosie

[U-Net: Convolutional Networks for Biomedical Image Segmentation  MICCAO2015](https://arxiv.org/abs/1505.04597)

* Downsample layers + Upsample layers made of Convolutional and Residual layers, and residual connection between the downsampling and upsampling feature maps.
* Some modifications, including attentions are added.

> In practice, model architecture is important to the performance of DDPM. I found that, even for the simplest MNIST dataset, if just use DNN or some simple Conv networks leads to very bad performance, i.e., only generate noise picture. However, use U-net-like networks, including residual layers, will have a better performance.

## Inference

1. Sample $X_T \sim \mathcal N(0, 1)$

2. Iteratively: $x_{t-1} = \dfrac{1}{\sqrt{\alpha_t}}\left(x_t - \dfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t) \right) + \sigma_t \mathcal N(0, 1)$

   The first term is to compute $x_{t-1}$ using $q(x_{t-1}|x_0, x_t)$, and the second term is a random noise.

   $\sigma_t$ is determined by the scale:

   To maintain $\mathbb E \Vert x_{t-1} \Vert^2 = 1$, notice that:

   $x_t = \sqrt{\bar \alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, then if we assume $\epsilon_\theta(x_t, t) = \epsilon$, $\Vert\epsilon\Vert^2$, and $\Vert x_0 \Vert = 1$, we have:

   $\mathbb \Vert x_{t-1}\Vert^ 2= \dfrac{1}{\alpha_t}\Vert x_0 \Vert^2 + \dfrac{1}{\alpha_t}(1-\bar\alpha_t + \dfrac{1-\alpha_t}{1-\bar\alpha_t})\Vert \epsilon \Vert^2 + \sigma^2$,

   $1 = \dfrac1\alpha_t (2 - \bar\alpha_t + \dfrac{1-\alpha_t}{1-\bar\alpha_t}) + \sigma^2$

   Solve this, we can get $\sigma^2 = \beta_t\dfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$ (notice the fact that $\alpha_t \bar \alpha_{t-1} = \bar\alpha_t$)

**Langevin Dynamics**: In the inference, when computing $x_{t-1}$, a noise term is added, which resembles Langevin Dynamics.

