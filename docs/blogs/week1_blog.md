# 从图注意力到自注意力

------

## 一、两种注意力的数学对照（本文灵魂）

在开始动手之前，我花了一天时间认真研究 Transformer。看完之后我愣了——**Transformer 里的自注意力，和我硕士做了两年的 GAT，本质上是同一件事**。

### 1.1 我熟悉的 GAT 注意力

GAT（Graph Attention Network）里，节点 $i$ 聚合邻居信息时的注意力权重是这样算的：

$$ \alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [W\mathbf{h}_i | W\mathbf{h}*j]))}{\sum*{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [W\mathbf{h}_i | W\mathbf{h}_k]))} $$

然后对邻居做加权求和：

$$ \mathbf{h}*i' = \sigma\left(\sum*{j \in \mathcal{N}*i} \alpha*{ij} W\mathbf{h}_j\right) $$

抽象一下，这就是两步：

1. **算"节点 i 和节点 j 有多相关"**
2. **softmax 归一化后加权求和**

硕士两年我几乎每天和这两个公式打交道。

### 1.2 我新学的 Transformer 自注意力

Transformer 里的自注意力定义：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

抽象一下：

1. **算"Query i 和 Key j 有多匹配"**（这里是内积）
2. **softmax 归一化后加权求和（对 Value 加权）**

等等——**这不就是 GAT 吗？**

### 1.3 它们到底是不是同一件事

把两种架构放在一起对比：

| 维度       | GAT                             | Transformer Self-Attention |
| ---------- | ------------------------------- | -------------------------- |
| 数据结构   | 图（节点 + 边）                 | 序列（token 列表）         |
| 邻接约束   | 只聚合 $\mathcal{N}_i$ 内的邻居 | 聚合所有 token（全连接图） |
| 相似度函数 | LeakyReLU(a · [Wh_i ‖ Wh_j])    | q · k / √d                 |
| 节点表示   | 共享一个 Wh                     | 分成 Q / K / V 三个角色    |
| 归一化     | softmax over 邻居               | softmax over 所有 token    |
| 聚合       | Σ α · Wh                        | Σ A · V                    |

**本质洞察：**

- **Transformer = 全连接图上的 GAT**（把每个 token 当节点，两两相连）
- **GAT = 稀疏图上的 Transformer**（用邻接矩阵屏蔽非邻居的注意力）
- 两者唯一核心差异是**相似度函数的形式**：GAT 用"拼接 + 单层 MLP"，Transformer 用"点积"

### 1.4 为什么"点积相似度"最终胜出？

这是一个工程问题。

GAT 的相似度计算是 `LeakyReLU(a · [Wh_i ‖ Wh_j])`——每一对节点都要**单独**做一次 MLP 前向。这种计算**难以打包成矩阵乘法**，只能一对一算。

Transformer 的相似度是 `q · k`——所有 Query 和所有 Key 的相似度**可以一次性矩阵乘法打包**算完：`Q @ K.T` 的每个元素就是一对 (Query, Key) 的相似度。

矩阵乘法是 GPU 最擅长的运算。**这让 Transformer 能 scale 到千亿参数，而 GAT 很难**。

更深一层的统计学洞察：Transformer 在点积后除以 $\sqrt{d_k}$，这是一个非常漂亮的细节。假设 q 和 k 的每个分量独立零均值、方差为 1，那么 $\mathbf{q} \cdot \mathbf{k}$ 的方差等于 $d_k$。除以 $\sqrt{d_k}$ 是把方差拉回 1，让 softmax 工作在合理区间。**纯粹的统计学救场**。

这个发现对我最大的意义是：**硕士两年的工作没白做**。当别人从 0 学 Transformer 时，我是在"换一个角度看老朋友"。

------

## 二、Week 1 我亲手跑通的东西

数学理解只是半边。另一半是动手。Week 1 我从零搭起了整个开发链路。

### 2.1 环境搭建：从 0 到 Qwen 在我电脑上说话

先说点坦诚的——**环境配置比想象中折腾**。

具体来说踩的坑：

- Conda 默认走 Anaconda 官方源，第一次装包就撞上 Terms of Service 错误，换清华镜像解决
- PyTorch 要根据有没有 NVIDIA 显卡选 CPU / CUDA 版，选错得重装
- VSCode 在 Windows Administrator 账户下找不到 conda 环境（这个坑我最后选择放弃 VSCode 自动识别，直接用 Anaconda Prompt 跑代码）
- Gradio 4.44 有个 `bool is not iterable` bug 和 jinja2 不兼容（这个我最后绕过去了——**放弃网页界面，改用 CLI 版**）

这些坑**不会写在任何教程里**。但它们构成了一个工程师的日常。

最终我搭起来的栈：

```
Python 3.10 (conda env: zhijiao)
  + PyTorch 2.11 (CPU 版)
  + Transformers 4.44
  + ModelScope (国内模型下载)
  + Git 2.53
```

然后从魔搭下载了 Qwen2.5-1.5B-Instruct，3GB 模型文件，10 分钟下完。

### 2.2 第一次让 Qwen 说话

我写了一个 30 行的脚本让模型加载、推理、输出。当我第一次看到终端里打出："**列表推导式是 Python 中快速构建列表的简洁语法**…" 那一刻，有一种奇妙的感觉——

**15 亿参数的大模型，此刻正在我的电脑 CPU 上为我做 15 亿次数学运算**。

用的是 AMD Ryzen 7 9700X，纯 CPU 推理速度是 **4.94 token/s**。对于一个没有 NVIDIA 显卡的本地开发环境来说，这个速度**刚好能接受**——每次回答等个 30 秒，可以用来实验 Prompt，可以用来测试多轮对话记忆。

Week 1 的最后，我用 Gradio 踩坑失败，转而写了一个命令行版的多轮对话界面。我试了这组测试：

第 1 轮：`请用一句话解释什么是列表推导式` 第 2 轮：`能再给 3 个更复杂的例子吗？` 第 3 轮：`把你刚才第二个例子改写成普通 for 循环`

模型回答得相当流畅，**第 3 轮能准确引用第 2 轮生成的代码**——这就是 In-Context Learning 在起作用。

### 2.3 关于 Tokenizer 的意外发现

我做了一个小实验，把几段不同的文本喂给 Qwen 的 tokenizer，看它怎么切：

| 原文                               | 字符数 | Token 数 | 比例 |
| ---------------------------------- | ------ | -------- | ---- |
| 南昌职业大学数智学院               | 10     | **6**    | 1.67 |
| I love teaching AI.                | 19     | **5**    | 3.80 |
| 🚀 人工智能 Artificial Intelligence | 30     | **5**    | 6.00 |
| 在2026年，大模型变得无处不在       | 16     | **13**   | 1.23 |

一些观察：

- **英文 token 效率远高于中文**：一个 token 平均覆盖 3-4 个英文字符，但只覆盖 1.5-2 个汉字
- **"南昌职业大学"被切成 `南昌 / 职业 / 大学`**（3 个 token），但"**数智学院**"被拆成 `数 / 智 / 学院`（3 个 token）——因为"数智"是新词，tokenizer 训练时没见过
- **中英混杂 + 数字标点**的切分效率最差

这些观察对我后续做微调数据设计很有用——尽量**避免大量标点、数字、emoji 混杂**，能压缩 token 消耗。

------

## 三、10 周计划与下周预告

Week 1 的 5 个 Commit 已经推到 GitHub 了，项目叫**职教通（Zhijiaotong）**——一个面向职业院校的 AI 教学助手。

一些阶段规划：

- **Week 2**：Prompt Engineering 与评测基础（建立双轨评测体系）
- **Week 3**：批改 Agent MVP（能自动批改 Python 代码作业）
- **Week 4-5**：LoRA 微调实战（给 Qwen 装上"出题能力"）
- **Week 6**：RAG 与向量数据库
- **Week 7**：多 Agent 编排（LangGraph）
- **Week 8-9**：前端集成 + Docker 部署
- **Week 10**：作品集整理 + 论文投稿

一个具体目标：把这个项目做成能直接给职业院校老师用的工具——数据不出校园，Docker 一键部署，支持批改、出题、生成教案三大场景。

------

## 结尾的一点感想

回头看，Week 1 真正的收获不是"我学会了 LLM"——一周的时间学不会什么新东西。真正的收获是**重新认识了自己过去两年的工作**。

**过去做 GAT 时，我从未想过它会成为我进入 LLM 世界的钥匙。**

硕士期间有一个深夜，我对着 GAT 论文公式看了很久，终于理解了注意力权重是怎么运作的。当时以为这就是图神经网络的一个小 trick。没想到几年后，**同样的机制在序列上被放大，最终重塑了整个 AI 产业**。

图上的注意力和序列上的注意力，最终都指向同一个朴素的问题——**哪些信息更值得我关注？**

------

## 关于我

- **GitHub**：[github.com/lizhiwei-stat/zhijiaotong](https://github.com/lizhiwei-stat/zhijiaotong)

  