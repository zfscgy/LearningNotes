# Origin

## Shamir

# Application in MPC

## ABY

[ABY-A framework for efficient mixed-protocol secure two-party computation  NDSS 2015](https://encrypto.de/papers/DSZ15.pdf)

Two party semi-honest setting.

Arithmetic(A-) sharing for ADD/MUL, Boolean(B-)/Yao sharing for non-linear functions.

### Arithmetic Sharing

* Additive Sharing, i.e., $x = \langle x \rangle^A_0 + \langle x\rangle^A_1$.
* Using Beaver triples for multiplication

### Yao Sharing

Party 0 generates the garbled table for circuit $\langle x \rangle^A_0 + \langle x\rangle^A_1$

The garbled values of the output wire: $k_1^w = k_0^w \oplus R$

* $\lang x\rang_0^Y = k_0^w, \lang x \rang_1^Y = k_0^w\oplus xR = k_b^w$

Here $x$ is a **boolean** value. To share a integer is to share all the bits of the integer.

> Seems the free-xor optimization is used.

### Boolean Sharing

* $\langle x \rangle_0^B \oplus \langle x \rangle_1^B = x$
* The computation protocols are ike the A-sharing on ring $\mathbb Z_2$

### Conversions

### A2Y

To evaluate a garbled circuit $\langle x \rangle_0 + \langle x \rangle_1$.

Party 0 generates the garbled table, and party 1 maintain the garbled value of $x$.

### Y2B

If two bits $x = x'$, then $\lang x \rang_0^Y \oplus \lang  x\rang_1^Y = \bar 0 0...0$.

Otherwise $\lang x \rang_0^Y \oplus \lang x \rang_0^Y = \bar 1 R_{1:\kappa}$

So that $$

The first bit is the permutation bit, so that it **always different** when the the wire value is different.

**Permutation bit**: The point-and-permute trick for the garbled circuits, where a bit is associated to the wires, to indicate the row to encrypt for the garbled output (instead of encrypting all 4 rows and checking which decryption is the correct garbled output).

See: [A Gentle Introduction to Yao's Garbled Circuits   Page5](https://web.mit.edu/sonka89/www/papers/2017ygc.pdf).

### A2B

A2Y $\to$ Y2B

### B2A

Use **OT** to generate additive shares.

Notice that $x = \sum_{i=0}^l (\lang x^i \rang_0^B \oplus \langle x^i\rang_1^B)\cdot 2^i$. Here $x^i$ is the $i$-th bit in $x$.

* $s_{i,0} = (1 - \lang x^i\rang_0^B) 2^i - r_i$ ($r_i$ is a random number), corresponding to $\lang x^i \rang_1^B = 1$

* $s_{i, 1} = \lang x^i\rang_0^B 2^i - r_i$., corresponding to $\lang x^i \rang_1^B = 0$

So that we have: $s_{i, \lang x^i \rang_1^B} = (\lang x^i \rang_0^B \oplus \langle x^i\rang_1^B)\cdot 2^i - r_i$

* $x = \sum s_{i, \lang x^i \rang_1^B}+ \sum r_i$

Thus, party 0 can compute $s_{i,0}, s_{i,1}$ and be the OT sender, and party 1 is the receiver with choice bit $\lang x\rang_1^B$.

* Party 0 computes $- \lang x\rang_1^A = \sum r_i$

* Party 1 computes $\lang x \rang_0^A = \sum s_{i, \lang x^i \rang_1^B}$

Notice: although for every bit an OT is required, they can **be conducted at the same time**.

### Y2A

Although using a circuit $x - r$ can generate arithmetic shares of $x$, 

it is more efficient by Y2B $\to$ B2A

### B2Y

* Party 0 samples $k_0$ and $k_1 = k_0 \oplus R$ as the garbled value for $x$.

* If $\lang x \rang_0^B = 0$, then party 0 set messages = $(k_0, k_1)$, else the messages = $(k_1, k_0)$
* Party 0 acts as the OT sender and party 1 acts as the receiver with the choice bit $\lang x \rang_1^B$

Notice: although for every bit an OT is required, they can **be conducted at the same time**.

