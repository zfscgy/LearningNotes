# Transformer structure

*[Attention is all you need](https://proceedings.neurips.cc/paper/7181-attention-is-all), NIPS 2017*

## Attention

### Sealed dot-product attention

$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$

$Q$: query, dimension is $L \times d_q$

$K$: keys, dimension is $L \times d_k = d_q$

$V$: values, dimension is $L \times d_v$

$L$ is the temporal/positional dimension in the sequence

$QK^T$: the similarity between pairs of  queries and keys, size $L\times L$, can be interpreted as each query's projection on the key space.

Hence, the attention is kind of like a query, where we first project the query into key space, then fetch the corresponding values of the key and sum them with the weight.

The output size is $L\times d_v$

### Multi-head attention

Have multiple attention heads: $\text{head} = \text{Attention}(QW^Q , KW^K,VW^V)$, then concatenate the outputs together

Notice that each $W$ reduces the dimension from $d_\text{model}$ (the original embedding dimension) to $d_\text{model}/n$, where $n$ is the number of heads. When concatenate all $n$ heads together, the dimension returns to $d_\text{model}$. By this, the multi-head attention has the same input and output shapes.

## Encoder

Input

> $\to$ Embeddings ($L \times d$)，where $d$ is the embedding dimension 
> $\to$ Multi-head attention ($Q, K, V$ is **the same embedding**) $\to$ Add & Norm (with skip connection to before the attention) 
> $\to$ Feed forward (Linear-ReLU-Linear) $\to$ Add & Norm (adding the skip connection to before the attention and normalizing)

Repeat $N$ times

Output shape $L\times d_v$

## Decoder

### Masked attention

In $QK^T$ of size $L \times L$, appending a mask like 

$\begin{bmatrix} 0 & 0 & 0 & 0 & 0 & \cdots\\ 
1 & 0 & 0 & 0 & 0 & \cdots \\ 
1 & 1 & 0 & 0 & 0 & \cdots \\ \cdots\end{bmatrix}$ (here $0$ actually use $-\infty$ since the next layer is Softmax)

By this, the output of encoder in position $i$ only depends on the positions before $i$, i.e., $1, \cdots, i-1$.

Input

> $\to$ Embeddings (Outputs shifted right)
>
> $\to$ **Masked** multi-head attention $\to$ Add & Norm
>
> $\to$ Multi-head attention ($Q, K$ are from the encoder, $V$ is from the previous layer) $\to$ Add & Norm 
>
> $\to$ Feed forward $\to$ Add & Norm

Repeat $N$ times

Output shape $L\times d_v$

> $\to$ Linear & Softmax (shape $L \times \text{vocabulary size}$)

## Computation process

### Seq2seq-like

The encoder takes all the input sequence, while the decoder takes the shifted output sequence as input.

Thus, the sequence generation process is like

$y'_1 = f([x_1,\cdots, x_n], [])$ (left is the input sequence, right is the output sequence)

$y'_2 = f([x_1, \cdots, x_n], [y_1])$

...

In an iterative predicting way

## Opensource Code

### PyTorch official implementation

[transformer.py - pytorch/pytorch - GitHub](https://github.com/pytorch/pytorch/blob/HEAD/torch/nn/modules/transformer.py) (adding '1s' after the 'github' opens an online vscode for any github repository)

# BERT

[*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  ACL 2019*](https://arxiv.org/pdf/1810.04805.pdf)

Using the encoder (which is bi-directional since there is no mask) in the Transformer.

## Training

### Input format

[CLS] my dog is cute [SEP] he likes play ##ing [SEP] (CLS is the classification token)

Token embeddings: $E_\text{[CLS]}, E_\text{my}, E_\text{dog}, \cdots$

Segment embeddings: $E_A,\cdots, E_B, \cdots$

Positional embeddings: $E(0), E(1), E(2), \cdots$

### Task1: Masked Language Model

Mask some tokens with the special token [MASK], the get the embeddings of those masked token and use it to predict the actual token. (Using a Softmax layer)

### Task2: Next Sentence Prediction

Using the [CLS] embedding of two sentences to predict their relation (is the next sentence or not)

### Number of parameters

Bert-base: $L \text{ (\#layers)} = 12, H \text{ (hidden dimension)}=768, A \text{ (\#attention heads)}=12$

Total parameters = 110M

Calculation

One encoder contains totally $11 \times 768^2 \approx 6.5M$:

* each attention head: $3\times 768 \times 768/12$ (three matrices to encode $Q, K, V$)
* Totally 12 heads: $3\times 768 \times 768$
* Feedforward: $2 \times (4\times 768) \times 768$ since the middle layer in the feedforward network is 4 times larger.

Total 12 encoders: $\approx 78M$

Word embedding size: $\approx 30000\times 768 \approx 23M$

Total $\approx 101M$ [Different from 110M in the original paper?]

## Claimed advantage against OPENAI GPT

Bi-direcitonal: uses both left and right context

## Huggingface implementation

[modeling_bert.py - huggingface/transformers - GitHub](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)

# GPT

[*Improving Language Understanding by Generative Pre-Training* OpenAI]([Improving language understanding with unsupervised learning (openai.com)](https://openai.com/research/language-unsupervised))

Using transformer decoder

## Training

### Loss

**Prediction loss**

Token $\mathcal U = \{(u_1, u_2, \cdots, u_n)\}$

$L_1(\mathcal U) = \sum_i \log P(u_i|u_{i-1},\cdots, u_{i-k};\Theta)$ where $\Theta$ means the parameter

(Predicting next token)

**Supervised loss**

Token and label $\mathcal C = \{(x_1, \cdots, x_m, y)\}$

$L_2(\mathcal C) = \sum \log P(y|x_1, \cdots, x_m)$

**Pre-training task**

Only use $L_1$

**Supevised fine-tuning task**

Use   $L_2 + \lambda L_1$

## Process

* Step 1: generate embeddings

   $h_0 = UW_e + W_p$ where $W_e$ is the word embedding, $W_p$ is the positional embedding

* Step 2: transformer decoder for $n$ layers

  $h_l = \text{TransformerDecoder} (h_{l-1})$

* Step 3: predict next token

  $P(u) = \text{softmax}(h_nW_e^T)$ (Using the same token  embedding at step 1)

##  GPT-2

[*Language Models are Unsupervised Multitask Learners* OpenAI](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

Model is very similar to GPT with a few modifications.

No explicit multi-task, the task is like a certain sentece, i.e., "Translate this into french"

###  Byte Pair Encoding (BPE)

Get frequent segments in vocabulary.

## GPT-3

[*Language Models are Few-Shot Learners*    OpenAI](https://arxiv.org/abs/2005.14165)

Also use the transformer architecture

### Few Shot Learning

For a few shot task, when evaluating one example, randomly drawing $K$ examples from training set as conditioning, using newlines as delimiters.

### Beam Search for Generation

* The greedy search for generation is to always output the token with largest predicted probability.

* The beam search maintains a list of $n$ token sequences.

* During each step, it searches the best $n$ sequences (start from $n$ previous bests)
* The 'score' can be computed by multiplying the logits. ($p(t_2,t_1|t_0) = p(t_2|t_1,t_0)p(t_1|t_0)$)

Reference: [如何通俗的理解beam search？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/82829880)



# LLMs

## ChatGLM

[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360)  ACL 2022

Official implementation  [ChatGLM introduction - Jupyter Notebook.pdf](C:\Users\zf\Downloads\ChatGLM introduction - Jupyter Notebook.pdf)

Some modifications on positional encoding and attention mask.

* The two positional embeddings are like [1, 2, 3, ..., n (`[gmask]`), n(`<sop>`), n, n, ...] and [0, 0, 0, ..., 0(`[gmask]`), 1([`<sop>`]), 2, 3, ...
* In first positional embedding, the generated tokens shares the same embedding (all = n). In second positional embedding, the original tokens have the same embedding (all = 0).
* No attention mask is applied to the original sequence, while in generated sequences the casual mask is applied.

Here, `[gmask]` means the start of generation? `<sop>` means the sentence start.

### Open-source code

[THUDM/ChatGLM-6B: ChatGLM-6B: An Open Bilingual Dialogue Language Model | 开源双语对话语言模型 (github.com)](https://github.com/THUDM/ChatGLM-6B)

**Notice**: Installing the requirements.txt could cause troubles! Better try directly installing huggingface libs from conda!



## Reinforcement Learning Human Feedback

* Reinforcement learning guide: [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit1/introduction)

[InstructGPT   OpenAI 2022](https://arxiv.org/pdf/2212.10560.pdf)

RLHF contains 3 steps:

1. Supervised Fine-tuning: The labeler (human) create prompt and desired responses ($\{x, y\}$) . The LLM fine-tunes on the set $(x, y)$.
2. Given a prompt $x$, sample the LLM's outputs $y_1, y_2, \cdots$. 





# Related Technologies

## Fine-tuning

### LoRA - Low-Rank Adaptation

[*LoRA: Low-Rank Adaptation of Large Language Models*    ICLR 2022](https://arxiv.org/pdf/2106.09685.pdf)

* Basic idea: for one dense layer $W \in \mathbb R^{d\times d}$, such that $d$ is large, using a low-rank matrix $\Delta W = AB^T,\quad A,B\in\mathbb R^{d\times d'}$ to modify it: $W' = W + \Delta W$

* In transformer, the attention weights are modified: $W_q, W_k, W_v$. In multi-head attention, $W_q, ...$ can still be viewed as one, because:

  * $\begin{bmatrix} W_{q1}h \\ W_{q2}h \end{bmatrix} = \begin{bmatrix} W_{q1} \\ W_{q_2}\end{bmatrix} h$  (When computing self-attention, the embedding will be split into multiple heads)

  So LoRA can be directly applied.

### Reinforcement Learning Human Feedback



## Transformer Variation

### Local Self-Attention

When computing attention, only consider the nearby tokens, like CNNs.

See: [Image Transformer  ICML 2018](http://proceedings.mlr.press/v80/parmar18a/parmar18a.pdf)  

(Using transformer with local attention to generate image pixel by pixel. Similar to PixelRNN/Pixel CNN)

See also: [Pixel Recurrent Neural Networks   ICML 2016](http://proceedings.mlr.press/v48/oord16.pdf)

# Optimization

## Unlimited Length Sequence

[Unlimiformer: Long-Range Transformers with Unlimited Length Input  NeurIPS 2023]([2305.01625.pdf (arxiv.org)](https://arxiv.org/pdf/2305.01625.pdf))

* When computing attention between the $i$-th and $j$-th token, i.e., $h_i W_Q W_K^Th_j^T$, considering it as a query, 

  where the *key* is $h_j$, and the *value* is $h_iW_QW_K^T$.

* Only compute the k-nearest neighbors' attentions. Given query $h_iW_QW_K^T$, only find $k$-closest $h_j$ to it and compute the attention. Hence, not the whole sequence will be computed.

* For fast k-nearest neighbor search, using the **product quantization**

### Product Quantization

[Product Quantization for Nearest Neighbor Search   T-PAMI 2010](https://inria.hal.science/file/index/docid/514462/filename/paper_hal.pdf)

Consider a $D$-dimensional vector, divide it into $D/d$ sub-vectors of dimension $d$. Then quantize each $d$ sub-vector by clustering them into several centroids.

* Use the centroid index to generate the code
* Generate reverse index
* Find 'approximate' k nearest neighbors.

