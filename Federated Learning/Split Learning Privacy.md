# Split Learning Procedure

$X \stackrel{\text{bottom model}}\rightarrow H \stackrel{\text{top model}}\rightarrow Y$

* $X$: input features

* $H$: hidden/intermediate/forward feature/representation/embedding
* $Y$: label/prediction

The intermediate $H$ and $\partial \text{Loss} / \partial H$ is exchanged during the computation.

## Fundamental

### [Vepakomma: arxiv 2018] Split learning for health

[Split learning for health: Distributed deep learning without sharing raw patient data (arxiv.org)](https://arxiv.org/abs/1812.00564)



# Privacy Problem

## Feature Leak (from top model party)

* Passive: Infer input $x$ from forward embedding $h$
* Active: guide the bottom model to produce invertible embeddings

## Label Leak (from bottom model paty)

* Passive: Infer output $y$ from forward embedding $h$ or embedding gradient $\partial L/\partial H$

## Papers

### Feature leak

#### [Vepakomma: ICDMW 2020] Nopeek

*[NoPeek: Information leakage reduction to share activations in distributed deep learning](https://ieeexplore.ieee.org/abstract/document/9346367/)*

**Settings**

*Passive attack*: the adversary do not modify the learning process

The adversary can get access to leaked input/embedding ($x/h$) pairs. The bottom model is unkonwn.

**Method**

* Adversary: gets a certain number of $X', H'$ pairs, then train a **reconstruction** network to output $x$ from $h$

* Defense: adding $\text{DCOR}(X,H)$ to the loss function, i.e., minimizing the **distance correlation** loss.

#### [Xinjian Luo: ICDE 2021] Feature Inference Attack for VFL

[Feature Inference Attack on Model Predictions in Vertical Federated Learning](https://ieeexplore.ieee.org/abstract/document/9458672)

**Settings**

*Passive attack*: the adversary do not modify the learning process

* *Active party* has label and some features
* *Passive party* has other features
* During training, the forward embedding of passive party ($h_0$) is sent to the active party. Active party concatenate with its own embedding to get $h = \text{concat}(h_0, h_1)$ and then feed it to the top model.

**Security assumption**: The complete model is known to the active party.

**Attack methods**

**Logistic regression** (Solve equations)

If there are $n$ classes, and the passive party has $k < n$ features. Then each iteration, active party receives:

$\begin{bmatrix}w_{1,1} x_1 + \cdots w_{1,k}x_k \\ \cdots \\w_{n,1} x_1 + \cdots w_{n,k}x_k \end{bmatrix} = \begin{bmatrix}h_1 \\\cdots \\h_n\end{bmatrix}$

where each row represents passive party's output on one class.

In this case, the number of variables = $k$ and the number of equations = $n$, which makes the variables solvable.

> Comment:
>
>  When the parameter of the passive party is unknown, there are $Bk + nk$ variables unknown in a batch size of $B$ (for inputs and weights), and there are $Bn$ equations (outputs sent to the active party). $Bk+ nk \le Bn \Leftrightarrow B(n-k) \ge nk \Leftrightarrow B \ge nk/(n - k)$. 
>
> I.e., with enough batch size, the number of equations exceeds the number of unknowns, which makes the equation solvable.

**Decision tree**: Restricting the path to infer passive party's features. (Omitted)

**Deep network** (Train a generator to revert forward embedding)

Bottom model: $f(x_a, x_p)$, where $x_a$ is active party input, $x_p$ is passive party input $x_p$.

generator: $g$

The loss is $\min_g \Vert f(x_a, g(x_a, r)) - h\Vert^2$.

The idea is to train a generator for $x_p$ to make the forward embedding same.

#### [Pasquini: CCS 2021] Unleashing the tigher

[*Unleashing the tiger: Inference attacks on split learning*](https://dl.acm.org/doi/abs/10.1145/3460120.3485259)

**Setting**

*Active attack*: the adversary will modify the learning process

The attack is conducted by the top model party **during split training**.

The top model party have some data $X'$ which is from same distribution of the actual input feature $X$

**Attack methods**

**Feature-space hijacking**

The top model party controls the learning process, then he forces the bottom model to be a invertible model, i.e., can use the forward embedding to generate the raw input.

* **Train the encoder-decoder:** 

  Suppose the forward embedding has dimension $k$, then the server use $X'$ to train a auto-encoder like network:  $\hat f, \hat f^{-1}$, where $\hat f$ is an encoder maps input features to a $d$-dimensional embedding, and $\hat f^{-1}$ is the corresponding decoder.

* **Make bottom model behave like encoder:**  

  use a discriminator $D$ to distinguish the output of bottom model $f(x)$ and the output of encoder $\hat f(x')$. Where $x$ is a input feature from the bottom model's party. Training is performed in a GAN-like manner:

  * First train $D$ to distinguish the forward embedding and encoder embedding
  * Then train the bottom model $f$ to make the forward embedding more like the encoder embedding. (Maximize $D$'s error)

* **Use decoder to reconstruct feature from the forward embedding:**

  In the process, the forward embedding $f(x)$ is more like the encoder embedding $\hat f(x)$, then we can use $\hat f^{-1}$ to reverse it.

**Other point**

* DCOR defense results poorly

#### [Zecheng He: ACSAC 2019] MIA Against Collaborative Inference

[Model inversion attacks against collaborative inference](https://dl.acm.org/doi/abs/10.1145/3359789.3359824)

**Setting**: 

Collaborative learning, which is almost the same as split learning, while this paper is earlier than split learning.

**Attack methods**

* **White-box attack:** the bottom model $f$ is known, minimizing $\Vert f(x') - f(x) \Vert + \text{Reg}(x')$, where $x$ is the real features, $x'$ is the features to recover. Regularization is $TV$, total variance for better image.
* **Reconstruction attack:** the adversary can have $X'$ and $H' = f(X')$, then he learns a reconstruction network.
* **Shadow model reconstruction:** the adversary can have $X', Y'$ but cannot use them to generate $H'$. But he can use the top model $g$ to learn a shadow bottom model $f'$, such that $L[g(f'(x')), y']$  is low. 

#### [Shouling Ji: Arxiv 2022] Hashing Defending

[All You Need Is Hashing: Defending Against Data Reconstruction Attack in Vertical Federated Learning](https://arxiv.org/pdf/2212.00325.pdf)

* Generate a hash code (binary vector) for each class
* When training, minimize $\text{CE}(\hat y, y)$ and $\text{Dist}(h, h_b)$, where $\hat y, y$ are predicted label and the true label, and $h, h_b$ are the embedding and the corresponding pre-determined hash code for the class.

> ​	Comment: this means the label information is totally compromised!

### Label Leak

#### [Fu Cong: USENIX 2022] Label Inference Attack

[Label Inference Attacks Against Vertical Federated Learning | USENIX](https://www.usenix.org/conference/usenixsecurity22/presentation/fu-chong)

From hidden embeddings:

**Passive attack**

* Model completion: using leaked $h', y'$ to train (fine-tune) the top model with bottom model fixed
* Using semi-supervised learning techniques like MixMatch (used to create new mixed images) to generate more training samples.

**Active attack**

* Bottom model owner can increase the learning rate, so that the top model is more rely on the bottom model, making the attack more effective.

From gradients:

**Direct attack**

From the gradients of the logits

> ​	Comment: this is completely meaningless

#### [Oscar Li: ICLR 2022] Marvell

[Label Leakage and Protection in Two-party Split Learning](https://arxiv.org/pdf/2102.08504.pdf)

**Setting**

Binary classification

**Attack methods**

* Norm-based scoring: $\nabla_h L = (\hat p_1 - y)\underbrace{\nabla_h g(h)}_\text{gradient on logits}$, where $\hat p_1$ is the predicted logit. This results in the distribution of $\Vert \nabla_h L \Vert$ is different between positive and negative samples.
* Direction-based scoring: observed that $\nabla_hg(h)$ are within a small angle whatever $h$ is, the direction of $\nabla_h L$ is largely dependent on $\nabla_h g(h)$. Given one positive example's gradient $\nabla_hg(h_+)$, use **cosine similarity** to predict, i.e., if the similarity is positive, then predict positive, otherwise predict negative.

**Defense methods**

* Element-wise Gaussian noise added on the embedding gradient.

* Alignment: adding noise to make the norm and direction similar for all samples.

* Optimized perturbation: Marvell

  * Observe that $\text{KL}(\tilde P_0 \Vert \tilde P_1) \le \epsilon \Rightarrow \max_r\text{AUC}(r) \le 1/2 + \sqrt \epsilon/2 - \epsilon / 8$

  * Minimizing $\min_{D_0, D_1} \text{KL}(\tilde P_0 \Vert \tilde P_1) + \text{KL}(\tilde P_1\Vert \tilde P_0)$ with constraint $p \cdot \text{tr}(\text{Cov}[D_0]) + (1-p)]\cdot \text{tr}(\text{Cov}[D_1])$

    Here $P_0, P_1$ is the perturbed gradient distributions (of negative/positive samples), and $D_0, D_1$ are noise distributions.

    $\text{tr}(\text{Cov}[D])$ is the expected euclidean norm square of $D$, i.e., $\mathbb E\Vert D\Vert_2^2 = \mathbb E [d_1^2 + d_2^2 + \cdots]$

#### [Shangyu Xie: Arxiv 2023] Regression Setting

[Label Inference Attack against Split Learning under Regression Setting](https://arxiv.org/pdf/2301.07284.pdf)

**Setting**

Split learning under regression setting. The feature party holds all the features and controls the bottom model, while the label part holds all the labels and controls the top model.

**Label Inference Attack**

The feature party constructs the surrogate model to replace the top model, and constructs the dummy labels to replace the original labels. The feature party fixes the trained bottom model, uses the following loss to train the surrogate model and the dummy labels, thus implementing label inference attacks.

**Gradient Distance Loss**

The gradient $g_s$ returned by the surrogate model must be close to the original gradient $g$ returned by the top model.

$L_g = L(g_s, g)$

**Training Accuracy Loss**

An accuracy loss is added as regularization to bind the surrogate model to behave like a normal top model.

$L_t = L(M_s(M_b(X)), Y)$

**Knowledge Learning Loss**

We also assume that the attacker can hold a small set of leaked labels.

$L_k = L_g(g_\text{leaked}, g) + L_t(X_\text{leaked}, Y_\text{leaked})$

#### [Junlin Liu: Arxiv 2022] Clustering Attack

[Clustering Label Inference Attack against Practical Split Learning](https://arxiv.org/pdf/2203.05222.pdf)

It is discovered that the output of the attacker's trained bottom model in Split Learning appears to be clustered with same label.
In clustering based label inference attack, the attacker performs dimensionality reduction and clustering on the output of the trained bottom model, and establishes a correspondence between the obtained clusters and their labels, thus implementing label inference attacks.

### Feature + Label leak

#### [Erdogan: WPES 2022] Unsplit

[UnSplit: Data-Oblivious Model Inversion, Model Stealing, and Label Inference Attacks against Split Learning](https://dl.acm.org/doi/abs/10.1145/3559613.3563201)

**Setting**

Image dataset; Passive attack

**Attack methods**

Here bottom model is $f$, top model is $g$

* Feature inference: the top model adversary initiates $x'$ and $f'$, corresponding to the *surrogate input feature and bottom model.* Then given the actual forward embedding $h$, the adversary minimize the distance between embeddings $\Vert f'(x') - h\Vert^2$:

  * trains $x'$ with $f'$ fixed (also with a regularization term called Total Variance for better image quality)
  * trains $f'$ with x' fixed

* Label inference: the bottom model adversary initiates $g'$, then minimize the distance of the gradients:

  $\Vert \partial \underbrace{L(g'(f(x), y')) / \partial f(x)}_\text{surrogate gradient} - \underbrace{\partial L(g(f(x)), y)/\partial f(x)}_\text{real gradient}\Vert$

**Other point**

* DCOR defense results poorly

> ​	Comment: seems focus on the image dataset, so that it is easy to invert the image from early convolutional layer outputs.

## Related surveys

