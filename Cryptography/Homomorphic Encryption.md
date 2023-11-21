# Fully Homomorphic Encryption

## Ring-LWE

### CKKS

Mainly from [CKKS explained, Part 3: Encryption and Decryption (openmined.org)](https://blog.openmined.org/ckks-explained-part-3-encryption-and-decryption/)

* Basis

  * Plaintext - native: $\mathbb C^{N/2}$

  * Plaintext - encoded: $\mathbb Z[X]/(X^N+ 1)$ polynomial of degree $N$

  * Ciphertext: $\left(\mathbb Z_q[X] / (X^N +1)\right)^2$

* Encoding

  Roots of $X^N + 1$: $\xi, \xi^3, \cdots, \xi^{2N-1}$,

  We use $m(X)$ to encode $z_1, \cdots, z_{N}$, where $m(\xi^{2i + 1}) = z_i$. Also, consider only the real parts, the number of encoded elements is halved.

  The problem of finding $m$ is just an matrix multiplication: $\sum_{j=0}^{N-1} \alpha_i \xi^{(2i - 1)j} = z_i$

* Public Key:  $p = (-a\cdot s + e, a)$, where $a$ is random, $s$ is the secret key, $e$ is a small error. Both of them are ring elements.

* Encrypt: $c = (m - a\cdot s + e, a)$

* Add: simply add two coordinates ($m_1 - a_1\cdot s + m_2 - a_2 \cdot s, a_1 + a_2$)
* Multiply: 
  * Ciphertext-plaintext multiplication is simply multiply the ciphertext coordinates
  * Ciphertext-ciphertext multiplication: CMult - Relin (Relinearization)
  
* Scale coefficients: $\Delta$ = decimal scale, $q_0 \gg \Delta$ = the initial coefficient (prime), $q_l / q_{l-1} \approx \Delta$ is the coefficient at level $l$, level $l-1$

