From the Boyd Book *Convex Optimization*

# Basic Sets

## Affine set

$\theta x_1 + (1-\theta)x_2$ still in the set ($\theta \in \mathbb R$)

### Affine hull

$\text{aff }C = \{\theta_1x_1 + \cdots + \theta_n x_n\}, x_1\cdots \in C$

### Relative interior

 $\text{rel }C$, is the interior of the set $C$ in $\text{aff } C$

## Convex set

$\theta x_1 + (1-\theta)x_2$ still in the set ($\theta \in [0, 1]$)

### Convex hull

$\text{conv }C = \{\theta_1x_1 + \cdots + \theta_n x_n\}, x_i \in C, \theta_i \in [0, 1]$

### Cone

$\theta x_1 + (1-\theta)x_2$ still in the set ($\theta > 0$)

### Conic hull

$\{\theta_1x_1 + \cdots + \theta_n x_n\}, x_i \in C, \theta_i \ge 0$

### Important examples

* Hyperplane $\{x \mid a^Tx = b\} = \{x \mid a^T(x-x_0) = 0\} = \{x_0 + a^\perp \mid (a^\perp)^Ta = 0\}$

* Halfspace $\{x \mid a^Tx = b\} = \{x \mid a^T(x - x_0) \le 0\}$

* Euclidean balls $B_r(x_c) = \{x \mid \Vert x - x_c\Vert_2 \le r\} = \{x_c + ru \mid \Vert u\Vert_2 \le 1 \}$

* Norm balls&cones

* Polyhedra 

  * Polyhedron $\mathcal P = \{x \mid a_i^Tx \le b_i, c_j^Tx = d_j\}$, intersection of hyperplanes, halfspaces; Convex hulls are polyhedron

  * Simplexes $C =  \text{conv}\{v_0, \cdots, v_k\} = \{\theta_0v_0 + \cdots + \theta_kv_k = \theta^Tv| \theta > 0, \mathbf 1^T\theta = 1\}$, e.g., unit simplex, probability simplex.

* Positive semi-definite cone $S_{+}^n = \{X\in S^n| X \ge 0\}$, i.e., real-valued symmetric matrices with positive entries

  If $A, B \in S_+^n$, then $x^T(\theta_1 A + \theta_2 B)x = \theta_1x^TAx + \theta_2x^TBx \ge 0$

## Convexity-preserving operations

### Intersections

Sets $S_1, S_2$ are convex, then $S_1 \cap S_2$ is convex

* $S_+^n = \cap_{z\ne 0} \{X\in S^n | z^TXz \ge 0\}$, since each term is linear function of $X$

* A closed convex set is the intersection of all halfspaces containing it.

### Affine functions

$f: \mathbb R^n \to \mathbb R^m$ is affine if it is linear plus constant.

Both affine function and its inverse preserves convexity

* Polyhedron $\{x|Ax < b, Cx = d\}$ is the inverse image of $y_0 < 0, y_1 = 0$ under the function $(Ax - b, Cx - d)$
* Hyperbolic cone $\{x|x^TPx < (c^Tx)^2, c^Tx \ge 0, P \in S_+^n\}$ is the inverse of $\{z|z^Tz < t^2, t \ge 0\}$ under the function $x = P^{1/2}z, t = c^Tx$

### Perspective function

$P(z, t) = z/t, z \in \mathbb R^n. t \in \mathbb R$

* Linear fractional $\dfrac{Ax + b}{c^Tx + d}$

## Generalized inequalities

### Proper cone

$K \sube \mathbb R^n$ is proper if it is convex, closed, solid (nonempty interior), pointed

### Generalized inequality

$y \le_K y \Leftrightarrow y - x \in K$

* Preserve under addition: $x \le_K y, u \le_K v \Leftrightarrow x + u \le_K y + v$
* Transitive: $x \le_K y, y\le_K z \Rightarrow x \le_K z$
* Preserved under non-negative scaling
* Reflexive: $x \le_K x$
* Antisymmetric: $x \le_K y, y\le_K x \Leftrightarrow x = y$
* Preserved under limits

### Minimum/Minimal elements

If $x \in S$:

* Minimum element of $S$: $x \le_K y$ for all $y \in S$ (If exists, then it is unique)
* Minimal element of $S$: $y \le_K x$ if and only if $y = x$. (One set can have multiple minimal elements)

## Separating/supporting Hyperplanes

### Separating hyperplane theorem

For joint convex sets $C \cap D = \empty$, there is a hyperplane $a^Tx = 0$ separates them, i.e., $a^Tx \le 0$ for $x \in C$ and $a^Tx \ge 0$ for $x \in D$.

* Strict separation: $a^Tx < 0$ for $x \in C$ and $a^Tx > 0$ for $x \in D$.

### Supporting hyperplane

* Boundary: $\textbf{bd } C = \textbf{cl } C \ \backslash \ \textbf{int } C$, closure - interior

For $x \in \textbf{bd }C$, if $a^Tx \le a^Tx_0, a\ne0$ for all $x\in C$, $\{x|a^T(x-x_0)=0\}$ is a supporting hyperplane at $x_0$.

* A convex set has supporting hyperplane at any boundary point
* Converse theorem: If a set is closed, has non-empty interior, at has a supporting hyperplane at any boundary point, then it is convex.

## Dual cones

$K^* = \{y \mid x^Ty \ge 0 \ \forall x\in K\}$

### Examples

* Orthogonal complement of subspace
* Self-dual: the dual of $\mathbb R_+^n$ is its self since $\{x^Ty\ge 0\}$ for all $x \ge 0$ $\Leftrightarrow y \ge 0$
* Self-dual Positive semidefinite cone.
* Dual of norm cone: $K = \{(x, t) \mid \Vert x \Vert \le t \}$, then $K^* = \{(y, s) \mid \Vert y\Vert_* \le v\}$

  $\Vert y \Vert_* = \sup \{u^Tx \mid \Vert x \Vert \le 1\}$ is the **dual norm**

### Properties

* Closed, convex
* $K_1 \subseteq K_2 \Rightarrow K_2^* \subseteq K_1^*$ 
* $K^{**}$ is the closure of the convex hull of $K$

### Inequalities

* $x \le_K y \Leftrightarrow \forall\lambda\ge_{K^*} 0, \lambda^Tx \le \lambda^Ty$
* $x <_K y \Leftrightarrow  \forall \lambda \ge_K^*0, \lambda \ne 0, \lambda^Tx < \lambda^Ty$

### Minimum/minimal element

* $x$ is minimum in $S$, w.r.t. $\le_K$ if and only if 

  $\forall \lambda \ge_{K^*} 0$ $x$ is the unique minimizer of $\lambda^Tz, z\in S$

* $x$ is minimal in $S$, w.r.t. $\le_K$ if

  $\exist \lambda >_{K^*} 0$ $x$ is the unique minimizer of $\lambda^Tz, z\in S$

### Example: Pareto optimal production frontier

* Resource vector $x = (x_1, x_2, \cdots, x_n)$ is the consumption on different resources.

* Production set $P = \{x\} \sube \mathbb R^n$ is the possible resource vectors (to achieve some target, e.g., produce an item)

* For two resource vectors $x, y$, $x$ is better than $y$ when $x <_{\mathbb R^+} y$ (denoted as $x \prec y$), 

  i.e., for all resources $x$ cosumes less than $y$

* We have: $x$ is minimal if exists $\lambda \in \mathbb R_+$, that $\lambda^Tx \preceq \lambda^T y$ for all $y \in P$. $\lambda$ can be interpreted as price. I.e., exists a possible price assignment, such that $x$ minimizes the production cost. If there are some $y \prec x$, $y$ will always cost less.

# Convex Function

$f(\theta x + (1 -\theta)y) \le \theta f(x) + (1-\theta)y$ when $\theta \in [0, 1]$

* Concave: $-f$ is convex

## Conditions

### First-order

$f(y) > f(x) + \nabla f(x)^T(y-x)$: grows faster than linear

### Second order

$\nabla^2 f(x) \succeq 0$ ($h^T\nabla^2f(x)h > 0$)

## Examples

* Exponential: $e^{ax}$
* Powers: $x^a$ $x > 0, a \ge 1 \text{ or } a \le 0$ (concave when $0 \le a \le 1$)
* Power of absolute value $|x|^p, p \ge 1$
* $\log x$ is concave
* Negative entropy $x \log x$

* Norms on $\mathbb R^n$
* $\max \{x_1, \cdots, x_n\}$
* Quadratic-over-linear $f(x, y) = x^2/y$
* Log-sum-exp $\log \sum_{i=1}^n e^{x_i}$ on $\mathbb R^n$
* Geometric mean $\left(\prod_{i=1}^n x_i \right)^{1/n}$ on $\mathbb R^n_{++}$
* Log-determinant $\log \det X$ on $S_{++}^n$ (Positive-definite matrices)

## Epigraph

$\text{epi } f = \{(x, t)\mid f(x)\le t\}$

Epigraph of $f$ is convex if and only if $f$ is convex.

## Jensen's inequality

$f(\mathbb E x) \le \mathbb E f(x)$ for convex functions (Let $p(x_1) = \theta, p(x_2) = 1-\theta$ is the same as the convex definition)

### Extensions

* Arithmetic-geometric mean inequality: $a + b < \sqrt{ab}$
* $\log \dfrac{a + b}2 \ge \dfrac{\log a + \log b}2$
* Holder inequality: $\sum x_iy_i \le \left(\sum |x_i|^p \right)^{1/p} \left(\sum |y_i|^q\right)^{1/q}$

## Convexity-preserving operations

### Non-negative weighted sum

$f = w_1f_1 + \cdots + w_mf_m$

### Composition with an affine mapping

$g(x) = f(Ax + b)$

### Pointwise maximum/supremum

$f(x) = \max \{f_1(x), \cdots, f_m(x)\}$

* Pointwise-linear $f(x) = \max \{a_1^Tx + b_1, \cdots, a_L^Tx + b_L\}$
* Sum of $r$ largest components $f(x_1, \cdots, x_n) = \sum_{i = 1}^r x_{[i]}$, where $x_{[i]}$ is the $i$-th largest, as it can view as the pointwise max of  $n \choose r$ linear functions.
* Support function of a set $S_C(x) = \sup\{x^Ty \mid y\in C\}$
* Distance to farthest point of a set $f(x) = \sup_{y\in C} \Vert x - y \Vert $
* Maximum eigenvalue of symmetric matrix $f(X) = \sup\{y^TXy\mid \Vert y\Vert_2 = 1\}$
* Norm of matrix $f(X) = \sup\{u^TXv\mid \Vert u\Vert_2=1,\Vert v\Vert_2 = 1\}$

### Composition

$f = h\circ g : f(x) = h(g(x))$

**Scalar composition**: $f''(x) = h''(g(x))g'(x)^2 + h'(g(x))g''(x) $

| h                       | g       | f       |
| ----------------------- | ------- | ------- |
| convex, non-decreasing  | convex  | convex  |
| convex, non-increasing  | concave | concave |
| concave, non-decreasing | concave | concave |
| concave, non-increasing | convex  | concave |

**Simple examples**

* $g$ is convex, then $e^{g(x)}$ is convex

* $g$ is concave and positive, then $\log g(x)$ is concave
* $g$ is concave and positive, then $1/g(x)$ is convex
* $g$ is convex and nonnegative, $p \ge 1$, then $g(x)^p$ is convex
* $g$ is convex and negative, then $-\log[-g(x)]$ is convex

**Extend to vector-valued functions**

The non-increasing/non-decreasing conditions become element-wise

**Vector examples**

* $h(z_1, \cdots, z_n) = z_{[1]} + \cdots + z_{[k]}$: sum of top-k elements. $h$ is element-wise non-decreasing and convex. Then let $z_i = g_i(x)$ be convex, then $h \circ g = h(g_1(x), \cdots, g_n(x))$ is conex

* $\log \sum e^{g_i(x)}$ is convex as long as $g_i$ is convex
* $[\sum g_i(x)^p]^{1/p}$ is concave when $p \in [0, 1]$ and $g_i$ is concave & nonnegative, and is convex when $p >1$ and $g_i$ is convex & nonnegative
* Geometric mean $(\prod_{i=1}^k g_i(x))^{1/k}$ is concave if $g_i$ is concave & nonnegative

### Perspective of a function

$g(x, t) = tf(x/t)$ is a perspective of x

**Examples**

* Squared norm: $g(x, t) = t(x/t)^T(x/t) = \dfrac{x^Tx}{t}$
* Relative entropy: $g(x,t) = -t \log (x/t) = t\log t - t\log x$

## Conjugate function

$f^*(y) = \sup_{x\in \text{dom} f} (y^Tx - f(x))$

# Convex optimization problems

## Optimization problems

$\min f_0(x)$ : objective function

s.t. 

$f_i(x) \le 0$ (inequality constraints)

 $g_i(x) = 0$ (equality constraints)



* Domain: $\displaystyle \bigcap \text{dom } f_i \cap \bigcap \text{dom } g_i$
* Optimal value:  $p^* = \inf\{ f_0(x)| f_i(x)\le0, g_i(x) = 0 \}$
* Optimal point: $f_0(x^*) = p^*$, and satisfies the conditions
* Optimal set: set of all optimal points
* Solvable: optimal point is attained/achieved, if optimal set is not empty.

### Equivalent problems

* Change of variable: $\tilde f(z) = f(\phi(z))$, where $\phi$ maps $z$ to the original variable $x$, i.e., $\phi(z) = x$
* Transformation on objective/constraint functions. E.g., $\psi$ monotonic increasing, then $\tilde \psi(f(x))$

## Convex optimization problem

$\min f_0(x)$ : objective function

s.t. 

$f_i(x) \le 0$ (inequality constraints)

 $a_i^Tx = b_i$ (equality constraints)

where $f_0, f_i, \cdots$ are convex functions

* Local optima: $f_0(x) = \inf \{f_0(z)\mid z \text{ is feasible and } \Vert z - x \Vert < R\}$ ($x$ minimizes $f_0$ in its $R$-neighborhood)
* Global optima

### Optima criterion

$\nabla f_0(x)^T (y -x) \ge 0$ for all $y$ in feasible set

* $\nabla f_0(x) = 0$
* $-\nabla f_0(x)$ defines a support plane to the feasible set at $x$ ($x$ lies in the border and any decrease direction is towards out of the feasible set)

### Unconstrained

$\nabla f_0(x) = 0$

* Quadratic: $(1/2)x^TPx +qx$ ($P\in S_+$), optimal point $x$ satisfies $Px + q = 0$

  * If $q \notin \mathcal R(P)$, unbounded/no solution. (This case, $q^TPq = 0$)
  * If $P \succ 0$, then there is a original solution $x = P^{-1}q$
  * If $P$ is singular, and $q \in \mathcal R(P)$, the optimal points are a affine set

  
