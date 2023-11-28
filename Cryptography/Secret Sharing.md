# Origin

## Shamir



# Application in MPC

## ABY

[ABY-A framework for efficient mixed-protocol secure two-party computation  NDSS 2015](https://encrypto.de/papers/DSZ15.pdf)

Two party semi-honest setting.

Arithmetic(A-) sharing for ADD/MUL, Boolean(B-)/Yao sharing for non-linear functions.

### Arithmetic Sharing

* Additive Sharing, i.e., $x = \langle x \rangle_0 + \langle x\rangle_1$.
* Using Beaver triples for multiplication

### Boolean Sharing

* $\langle x \rangle_0 \oplus \langle x \rangle_1 = x$
* Like the A-sharing on ring $\mathbb Z_2$

### Conversions