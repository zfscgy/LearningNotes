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

  $=\text{DL}[q(Z|X)\Vert p(Z|X)] - \text{DL}[q(Z|X)\Vert p(Z)] + \mathbb E_{q(Z|X)} \log p(X|Z)$

* One important observation is that, the first term $\text{DL}[q(Z|X)\Vert p(Z|X)]$ is always non-negative hence the above equation is a **lower bound**.
*  Thus, we just have to optimize $\mathbb E_{q(Z|X)} \text{DL}[q(Z|X)\Vert p(Z)] + \mathbb E_{q(Z|X)} \log p(X|Z)$

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

## Purpose

* The denoising process: $x_T \to \cdots \to x_2 \to x_1$, where $p(x_t|x_{t + 1}) = \mathcal N(x_{t}|\sqrt{1-\beta_t}x_{t+1}, \beta_t I)$, i.e., adding Gaussian noise during each step (and scaling to maintain the variance) 

  Since we want to generate image from noise, it is natural to think the denoising process first.

* The reverse (noise-adding) process: $x_1 \to x_2 \to \cdots \to x_T)$, where $p(x_{t+1}|x_t)$ is approximated by a linear network $\mathcal N(x_t|\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

### Approximation
