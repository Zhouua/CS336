# CS336

[TOC]

## Basics

**Goal**: get a basic version of the full pipeline working

**Components**: tokenization, model architecture, training

---

data不一定越多越好，有效的data才是

### Tokenization

将String转换成Integer sequence

![image-20251118121346826](./Images/image-20251118121346826.png)

也有部分不用tokenizer的方法，直接使用Byte

本实验使用Byte Pair Encoding

### Architecture

基础Transformer架构：

![image-20251118121655336](./Images/image-20251118121655336.png)

### Assignment

- Implement BPE tokenizer

- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop

- Train on TinyStories and OpenWebText

- Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100

## Systems

**Goal: **squeeze the most out of the hardware 

**Components:** kernels, parallelism, inference

## Scaling laws

## Data

## Alignment