# Partly used Cryptography

## Flawed Works

### GELU-Net

[GELU-Net   IJCAI 2018](https://www.ijcai.org/Proceedings/2018/0547.pdf)

For a layer in neural network $f(x) = \sigma(wx + b)$

1. The client has $x$ and encrypt it using Paillier and gets $[[x]]$, then sends it to the server
2. The server computes $[[wx + b]] = w\otimes [[x]]\oplus b$ and sends it back to the client.
3. The client decrypts $[[wx + b]]$ and then compute $\sigma(wx + b)$

### BAYHENN

[BAYHENN   IJCAI 2019](https://www.ijcai.org/Proceedings/2019/0671.pdf)

When server computes $[[wx + b]]$, the $w$ is actually **sampled**, instead of the real $w$. So that the client can not solve the model weights even with multiple iterations.

## Critics

### Learning Model with Error

[Learning Model with Error   IJCAI 2020](https://www.ijcai.org/proceedings/2020/0488.pdf)

Against:

* GELU-NET
  * With several pairs of $(x, wx + b)$, it is very easy to reveal $w$ and $b$
* BAYHENN
  * The client can design queries such as $x= [0, 0, \cdots]$ to extract bias, and then use $x = [1, 0, 0, \cdots]$ to extract the specific weight.
  * BYHENN is not forming a real LWE (Learning With Errors) problem

