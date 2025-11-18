# CS336

## Basics

**Goal**: get a basic version of the full pipeline working

**Components**: tokenization, model architecture, training

---

**data不一定越多越好，有效的data才是**

### Tokenization

将String转换成Integer sequence

![image-20251118121346826](./Images/image-20251118121346826.png)

也有部分不用tokenizer的方法，直接使用Byte

本实验使用Byte Pair Encoding

### Architecture

基础Transformer架构：

![image-20251118121655336](./Images/image-20251118121655336.png)

### <a href="../assignment1-basics">Assignment</a>

- Implement **BPE** tokenizer

- Implement **Transformer, cross-entropy loss, AdamW optimizer, training loop**

- Train on ***TinyStories*** and ***OpenWebText***

- Leaderboard: minimize ***OpenWebText*** perplexity(困惑度) given 90 minutes on a H100



## Systems

**Goal:** squeeze the most out of the hardware 

**Components:** kernels, parallelism, inference

---

### Kernel

在芯片中，数据在**DRAM**和**SRAM+Compute Unit**中来回移动，这带来了Bandwith Cost。所以应该减少这样的移动从而带来最大化GPU利用率

### Parallelism

而当涉及到多卡的时候，GPU之间的通信成为新的bottleneck，同时**最小化数据移动**仍然重要。但是在多GPU的情况下，有一些额外的操作：使用collective operations，shard分片等

### Inference

即给定`prompt`，模型逐个生成后续`tokens`。推理不是训练，而是应用模型！

Globally, inference compute (every use) exceeds training compute (one-time cost)

Inference分为两部分：prefill和decode

![image-20251118145556866](./Images/image-20251118145556866.png)

- **prefill** phase 预填充，**一次性**对输入的`prompt`进行处理，是**compute-bound**
- **decoding** phase 解码，逐个生成新`token`，每步都基于上一步，是**memory-bound**。decode的优化方式：Use cheaper model，Speculative decoding，Systems optimizations
  - Use cheaper model，指通过**Pruning**剪枝移除不重要权重、**Quantization**量化降低为FP16/INT8精度、**Distillation**蒸馏为小模型
  - Speculative decoding，指用一个小的*draft model*进行预测多个token，然后用模型进行并行打分，若合理则采用
  - Systems optimizations，通过KV Cache和Batch批处理等方式

### <a href="../assignment2-systems">Assignment</a>

- Implement a **fused(融合的) RMSNorm** kernel in Triton

- Implement **distributed** data parallel training

- Implement optimizer state **sharding**    

- Benchmark and profile the implementations



## Scaling laws

Goal: do experiments **at small scale**, predict hyperparameters/loss at large scale

## Data

## Alignment