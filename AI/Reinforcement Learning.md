The following content is mainly from [Monte Carlo vs Temporal Difference Learning - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit2/mc-vs-td)

# Overview

* **Environment**: each time step is a **state**
* **Agent**: make **observations** on the environment state, then use the policy to take an **action**.
* **Policy**: $\pi(a|s)$, the conditional probability of the action on the state.
* **Reward**: after the agent take an action, the environment could give a **reward**.

* **Episode**: $s_0a_0r_0s_1a_1r_1\cdots$, a sequence of state-action-rewards

$s_i, a_i, r_i$ are the state, action, reward on time step $i$.

* **Value function**: discounted reward function start from a state $s$ using the policy $\pi$.

   $v_\pi(s) = \mathbb E_\pi [R_{t+1} + \gamma R_{t+1} + \gamma^2 R_{t+3} + \cdots|S_t = s]$

# Value-based Learning

## Value Estimation

### Bellman Equation

$V_\pi(s) = \mathbb E_\pi [R_{t+1} + \gamma V_\pi(S_{t+1})|S_t = s]$

This is a simple recursion definition of the value function

### Monte-Carlo method

Let return $G_t = \sum_{i=t}^\infty r_i$ (the cumulative reward from step $t$)

Then, after one episode is finished, update $V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$, where $\alpha$ is the learning rate. I.e., using $G_t$ to approximate $V(S_t)$

### Temporal Difference method

$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$

Learn the value at each step. It is not accurate at the beginning since all $V(\cdot)$ is randomly initialized.

## Q-Learning

**Q function:** action-value function: $Q_\pi*(s, a) = \mathbb E_{\pi}[G_t|S_t = s, A_t = a]$

**Update rule:** $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t} + \gamma \max_a Q(S_{t + 1}, a) - Q(S_t, A_t))$

**Running a episode: ** Using policy derived from $Q$, e.g., $\epsilon$-greedy (choose $\text{argmax}_a Q(a, S_t)$ with probability $1 - \epsilon$, and random with probability $\epsilon$)

###  Off-policy & On-policy

* Off-policy: different policy used for acting(inference) and updating(training), like in Q-Learning, in updating using the $\epsilon$-greedy for exploration, but greedy for acting.
* On-policy: the same policy for acting and updating.

## Deep Q-Learning

In Q-Learning, the Q function is a table function of state-action pairs, which needs to store the value for every state.

This is infeasible because in real-applications, **the number of state can be (almost) infinite.**

In this case, use a **neural network** instead of a table function as the Q function.

* Update rule: $\text{Loss}(\theta) = \Vert \underbrace{r_t + \gamma \max_{a} Q_\theta(s_{t+1}, a_{t+1})}_\text{estimated Q func by Temporal Difference} - Q_\theta(s_t, a_t) \Vert$

  Here $\theta$ is the network parameter. The update can be done by SGD. The estimation is produced by Temporal Difference method (instead of Monte-Carlo since it is unefficient)

* **Experience Replay**: store the $(s_t, a_t, r_t, s_{t+1})$ tuple inside a buffer, then randomly pick it to update.

  This is to make the distribution more uniform, i.e., when performing update, the gradient will not be correlated.

References:

1. [Playing Atari with Deep Reinforcement Learning (original paper of DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
2. [DeepRL系列(7): DQN(Deep Q-learning)算法原理与实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/97856004)
3. 

