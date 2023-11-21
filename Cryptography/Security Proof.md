# Simulation-based

*How To Simulate It – A Tutorial on the Simulation Proof Technique* Lindell

## Terminology

* Input from two parties: $x, y$ respectively
* The function to compute: $f$

* Two party protocol $\pi$

* The view of party $i$: $\mathsf{view}^{\pi}_i(x, y, n) = (w, r^i;m_1^i,\cdots, m_t^i)$,

  where $n$ is the **security parameter**, $r^i$ is the **internal random tape** of party $i$, $m_j^i$ is the $j$-th message party $i$ received.

* The output of party $i$: $\mathsf{output}_i^\pi(x, y, n)$, can be computed from its own view of the execution. (If the two parties have the same output, then $i$ can be omitted)

## Semi-honest

### Correctness

$\text{Pr}[\mathsf{output}^\pi(x, y, n)\ne f(x, y)] \le \mu(n)$, 

where $\mu(n)$ is some negligible function (smaller than 1 / 'any polynomial').

### Security

$\pi$ securely computes $f$ in the presence of static semi-honest adversaries if there exists P.P.T. (probabilistic polynomial-time algorithm) $\mathcal S_1, \mathcal S_2$ such that

* $\{(\mathcal S_1(1^n, x, f_1(x,y), f(x, y)\}_{x,y,n} \stackrel{c}\equiv \{\mathsf{view}_1^\pi(x, y, n), \mathsf{output}^\pi(x, y, n)\}$
* $\{(\mathcal S_2(1^n, x, f_1(x,y), f(x, y)\}_{x,y,n} \stackrel{c}\equiv \{\mathsf{view}_2^\pi(x, y, n), \mathsf{output}^\pi(x, y, n)\}$

*Notice that the $f(x, y)$ considered for probabilistic protocol*

## Examples

### Concrete

#### Oblivious Transfer for Semi-Honest Adversaries

From *How To Simulate It – A Tutorial on the Simulation Proof Technique* Lindell

* Basic idea: If the simulated view can be distinguished, then the hard-core predicate can be guessed.



### No formal proof

#### An Efficient and Probabilistic Secure Bit-Decomposition [Asia CCS 2013]

* Basic idea: convert the bit-decomposition in plaintext to ciphertext, with a protocol to extract the lowest bit (two-party).
* Proof method: Informal, "uniform because of the semantic security of Paillier"

#### Secureml: A system for scalable privacy-preserving machine learning [S&P 2017 Mohassel]

* Proof method: directly from arithmetic secret sharing

