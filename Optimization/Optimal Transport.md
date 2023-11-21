# Optimal Transport

## Basics

### Measures

* Probability simplex: $\sum_n := \{\mathbf a\mid \sum a_i = 1\}$
* Discrete measure: $\alpha = \sum a_i\delta_{x_i}$
* General measure: $d\alpha(x) = \rho_\alpha(x)dx$, $\int_{\mathbb R^d} h(x)d\alpha(x) = \int_{\mathbb R^d} h(x)\rho_\alpha(x)dx$. $\alpha$ is measure, and $\rho_\alpha$ is the density

Here the measures can be considered cumulative function, whose derivative is density.

### Optimal assignment

Cost matrix $C\in \mathbb R^{n\times n}$, $C_{i, j}$ can be considered moving something from $i$ (old place) to $j$ (new place). (Does not mean $C_{i,i} = 0$)

Problem:

$\displaystyle \min_{\sigma\in \text{Perm}(n)} \dfrac1n\sum_{i=1}^n C_{i,\sigma(i)}$

* Solution not unique

### Monge problem

$\mathcal X \times \mathcal Y$: the source space and the target space

$c(x, y)$: the cost of moving unit mass from source space position $x$ to target space position $y$

$\mathbf a, \mathbf b$: density, $a_i$ means the density at $x_i$

The problem is

$\displaystyle \min_T \sum_i c(x_i, T(x_i)) \quad \text{s.t. } T_\#\alpha = b$

Here $T_\#$ is the **push-forward operator**, which here can be view as: ${T_\#}_{i, j} = 1$ means that $T(x_i, y_j) = 1$

> This problem does not consider the case that we may move mass from one source position to multiple target positions

* Push-forward operator

  For $T: \mathcal X\to \mathcal Y$, $T_\#\alpha := \sum_i a_i\delta_{T(x_i)}$, which generates the distribution after moving.

### Kantorovich relaxation

$P_{i, j} \in \mathbb R^{n\times m}$ is the mass from source position $i$ to target position $j$.

Source measure (density) $\mathbf a$, target measure (density) $\mathbf b$

$P$ satisfies: $P\mathbf 1_m = \mathbf a, P^T\mathbf 1_n = \mathbf b$

$C_{i, j}$: the cost of moving unit mass from $i$ to $j$

The problem is:

$\displaystyle \min_P \sum_{i,j} C_{i,j}P_{i,j}$