# GDesigner 项目梳理文档

## 📋 项目概述

**GDesigner** 是一个基于图神经网络（GNN）的多智能体协作框架，用于解决复杂的AI任务。该项目通过构建智能体之间的图结构，实现空间连接（spatial）和时间连接（temporal）的动态优化，从而提升任务解决能力。

### 核心特性
- 🎯 **多智能体协作**：支持多种智能体类型（分析、代码编写、数学求解等）
- 🕸️ **图结构优化**：使用GCN（图卷积网络）动态优化智能体间的连接
- 🔄 **空间-时间连接**：支持空间连接（同轮次内）和时间连接（跨轮次）的优化
- 📊 **多数据集支持**：支持MMLU、HumanEval、GSM8K等数据集
- 🎨 **多种拓扑结构**：支持全连接、链式、星型、网格等多种图拓扑

---

## 📁 项目结构

```
Gdesigner/
├── GDesigner/              # 核心算法实现
│   ├── agents/            # 智能体实现
│   │   ├── analyze_agent.py      # 分析智能体
│   │   ├── code_writing.py       # 代码编写智能体
│   │   ├── math_solver.py        # 数学求解智能体
│   │   ├── adversarial_agent.py  # 对抗智能体
│   │   ├── final_decision.py     # 最终决策智能体
│   │   └── agent_registry.py     # 智能体注册表
│   │
│   ├── graph/             # 图结构实现
│   │   ├── graph.py       # 图类（核心）
│   │   └── node.py        # 节点类
│   │
│   ├── gnn/               # 图神经网络
│   │   └── gcn.py         # GCN模型实现
│   │
│   ├── llm/               # 大语言模型接口
│   │   ├── llm.py         # LLM抽象基类
│   │   ├── gpt_chat.py    # GPT聊天接口
│   │   ├── profile_embedding.py  # 角色嵌入
│   │   └── llm_registry.py      # LLM注册表
│   │
│   ├── prompt/            # 提示词管理
│   │   ├── prompt_set.py  # 提示词集基类
│   │   ├── mmlu_prompt_set.py    # MMLU提示词
│   │   ├── humaneval_prompt_set.py  # HumanEval提示词
│   │   ├── gsm8k_prompt_set.py     # GSM8K提示词
│   │   └── prompt_set_registry.py  # 提示词注册表
│   │
│   ├── tools/             # 工具集
│   │   ├── coding/        # 代码执行工具
│   │   ├── search/        # 搜索工具（arXiv、Wiki）
│   │   ├── reader/        # 文件读取工具
│   │   ├── web/           # 网页工具
│   │   └── vgen/          # 图像生成工具
│   │
│   └── utils/             # 工具函数
│       ├── const.py       # 常量定义
│       ├── globals.py     # 全局变量
│       └── utils.py       # 工具函数
│
├── experiments/           # 实验脚本
│   ├── run_mmlu.py       # MMLU数据集运行脚本
│   ├── run_humaneval.py  # HumanEval数据集运行脚本
│   ├── run_gsm8k.py      # GSM8K数据集运行脚本
│   ├── train_mmlu.py     # MMLU训练脚本
│   ├── evaluate_mmlu.py  # MMLU评估脚本
│   └── accuracy.py       # 准确率计算
│
├── datasets/              # 数据集
│   ├── mmlu_dataset.py   # MMLU数据集加载
│   ├── humaneval/        # HumanEval数据
│   ├── gsm8k/            # GSM8K数据
│   └── MMLU/             # MMLU数据
│
├── requirements.txt       # 依赖包
├── template.env          # 环境变量模板
└── README.md             # 项目说明
```

---

## 🔧 核心组件详解

### 1. Graph（图结构）
**位置**: `GDesigner/graph/graph.py`

**核心功能**：
- 管理多个智能体节点及其连接关系
- 支持空间连接（spatial）和时间连接（temporal）的动态构建
- 使用GCN优化连接概率
- 支持拓扑排序执行

**关键方法**：
- `construct_spatial_connection()`: 构建空间连接
- `construct_temporal_connection()`: 构建时间连接
- `run()`: 同步执行图
- `arun()`: 异步执行图（使用GCN优化）

### 2. Node（节点）
**位置**: `GDesigner/graph/node.py`

**核心功能**：
- 表示图中的一个智能体节点
- 管理前驱和后继节点
- 处理输入输出和记忆

**关键属性**：
- `spatial_predecessors/successors`: 空间前驱/后继
- `temporal_predecessors/successors`: 时间前驱/后继
- `inputs/outputs`: 输入输出
- `last_memory`: 上一轮次的记忆

### 3. Agents（智能体）
**位置**: `GDesigner/agents/`

**智能体类型**：
- **AnalyzeAgent**: 分析智能体，用于问题分析
- **CodeWriting**: 代码编写智能体
- **MathSolver**: 数学求解智能体
- **AdverarialAgent**: 对抗智能体，用于生成对抗性答案
- **FinalDecision**: 最终决策智能体（多种策略：Refer、Direct、WriteCode、MajorVote）

### 4. GCN（图卷积网络）
**位置**: `GDesigner/gnn/gcn.py`

**功能**：
- 基于节点特征和角色连接关系学习连接概率
- 使用节点角色描述和查询的嵌入特征
- 输出空间连接的logits

### 5. Prompt Sets（提示词集）
**位置**: `GDesigner/prompt/`

**功能**：
- 为不同领域（MMLU、HumanEval、GSM8K）提供专门的提示词
- 定义角色描述和连接关系
- 提供few-shot示例

---

## 🚀 使用方法

### 1. 环境配置

```bash
# 创建conda环境
conda create -n gdesigner python=3.10
conda activate gdesigner

# 安装依赖
pip install -r requirements.txt

# 配置API密钥
# 复制 template.env 为 .env 并填入API密钥
cp template.env .env
# 编辑 .env 文件，填入：
# BASE_URL = "your_openai_base_url"
# API_KEY = "your_openai_api_key"
```

### 2. 下载数据集

需要下载以下数据集：
- **MMLU**: 多任务语言理解数据集
- **HumanEval**: 代码生成评估数据集
- **GSM8K**: 数学问题求解数据集

### 3. 运行实验

#### MMLU数据集
```bash
python experiments/run_mmlu.py \
    --mode FullConnected \
    --batch_size 4 \
    --agent_nums 6 \
    --num_iterations 10 \
    --num_rounds 1 \
    --optimized_spatial
```

#### GSM8K数据集
```bash
python experiments/run_gsm8k.py \
    --mode FullConnected \
    --batch_size 4 \
    --agent_nums 4 \
    --num_iterations 10 \
    --num_rounds 1 \
    --optimized_spatial
```

#### HumanEval数据集
```bash
python experiments/run_humaneval.py \
    --mode FullConnected \
    --batch_size 4 \
    --agent_nums 4 \
    --num_iterations 10 \
    --num_rounds 1 \
    --optimized_spatial
```

### 4. 参数说明

**主要参数**：
- `--mode`: 图拓扑模式
  - `FullConnected`: 全连接
  - `Chain`: 链式
  - `Star`: 星型
  - `Mesh`: 网格
  - `Debate`: 辩论模式
  - `Layered`: 分层
  - `Random`: 随机
  - `DirectAnswer`: 直接回答（无连接）
  
- `--agent_nums`: 每个智能体类型的数量
- `--agent_names`: 智能体类型列表（如 `AnalyzeAgent`）
- `--num_iterations`: 优化迭代次数
- `--num_rounds`: 每个查询的执行轮数
- `--optimized_spatial`: 启用空间连接优化
- `--optimized_temporal`: 启用时间连接优化
- `--decision_method`: 最终决策方法（`FinalRefer`, `FinalDirect`, `FinalWriteCode`, `FinalMajorVote`）
- `--llm_name`: 使用的LLM名称（默认 `gpt-4o`）
- `--domain`: 数据集领域（`mmlu`, `humaneval`, `gsm8k`）

---

## 🔄 工作流程

### 1. 图构建流程
```
初始化Graph
  ↓
创建节点（根据agent_names）
  ↓
初始化潜在边（potential_spatial/temporal_edges）
  ↓
构建角色邻接矩阵（基于prompt_set）
  ↓
构建节点特征（基于角色描述嵌入）
  ↓
初始化GCN和MLP模型
  ↓
初始化空间/时间连接的logits和masks
```

### 2. 执行流程（arun方法）
```
输入查询
  ↓
构建新特征（节点特征 + 查询嵌入）
  ↓
通过GCN计算空间连接logits
  ↓
For each round:
  ├─ 构建空间连接（基于logits和masks）
  ├─ 构建时间连接（基于temporal_logits）
  ├─ 拓扑排序执行节点
  │   └─ 每个节点：
  │       ├─ 获取空间前驱信息
  │       ├─ 获取时间前驱信息
  │       └─ 执行任务
  └─ 更新记忆
  ↓
连接决策节点
  ↓
执行决策节点
  ↓
返回最终答案
```

### 3. 空间连接构建
- 基于`spatial_logits`（可训练或GCN输出）
- 使用sigmoid函数计算连接概率
- 检查循环依赖
- 根据概率随机采样连接

### 4. 时间连接构建
- 基于`temporal_logits`（可训练）
- 仅在round > 0时构建
- 允许节点访问上一轮次的信息

---

## 🎯 关键概念

### 空间连接（Spatial Connection）
- **定义**：同一轮次内节点之间的连接
- **用途**：允许节点在同一轮次内相互通信和协作
- **优化**：可通过GCN或可训练logits优化

### 时间连接（Temporal Connection）
- **定义**：跨轮次的节点连接
- **用途**：允许节点访问上一轮次的信息，实现迭代改进
- **优化**：通过可训练的logits优化

### 角色（Role）
- 每个节点有一个角色（如"Normal"、"Fake"等）
- 角色决定节点的提示词和连接偏好
- 通过`prompt_set`定义角色描述和连接关系

### 决策节点（Decision Node）
- 图执行完成后，所有节点连接到决策节点
- 决策节点负责整合所有节点的输出，生成最终答案
- 支持多种决策策略（参考、直接、写代码、多数投票）

---

## 📊 支持的拓扑模式

1. **FullConnected**: 所有节点相互连接（除自己）
2. **Chain**: 链式连接（1→2→3→...）
3. **Star**: 星型连接（中心节点连接所有其他节点）
4. **Mesh**: 网格连接（上三角矩阵）
5. **Debate**: 无空间连接，只有时间连接（辩论模式）
6. **Layered**: 分层连接（节点分为多层，层间连接）
7. **Random**: 随机连接
8. **DirectAnswer**: 无连接（直接回答）

---

## 🔬 优化机制

### 空间连接优化
- **GCN优化**：使用图卷积网络基于查询动态计算连接概率
- **可训练优化**：通过梯度下降优化连接logits
- **剪枝**：支持基于logits的连接剪枝

### 时间连接优化
- **可训练优化**：通过梯度下降优化时间连接logits
- **剪枝**：支持基于logits的连接剪枝

### 训练流程
- 使用训练集优化连接参数
- 支持批量训练
- 使用验证集评估性能

---

## 🛠️ 扩展开发

### 添加新智能体
1. 在`GDesigner/agents/`创建新的智能体类
2. 继承`Node`类并实现`_execute`和`_async_execute`方法
3. 使用`@AgentRegistry.register("AgentName")`注册

### 添加新数据集
1. 在`datasets/`创建数据集加载器
2. 在`GDesigner/prompt/`创建对应的提示词集
3. 继承`PromptSet`类并实现所有抽象方法
4. 使用`@PromptSetRegistry.register("dataset_name")`注册

### 添加新工具
1. 在`GDesigner/tools/`相应目录下创建工具
2. 在智能体中调用工具

---

## 📝 注意事项

1. **API配置**：确保正确配置`.env`文件中的API密钥
2. **数据集路径**：确保数据集文件放在正确的位置
3. **依赖版本**：注意PyTorch和CUDA版本的兼容性
4. **内存管理**：大批量或大量节点可能消耗较多内存
5. **异步执行**：推荐使用`arun`方法进行异步执行以提高效率

---

## 🔗 相关资源

- 参考项目：[GPTSwarm](https://github.com/metauto-ai/GPTSwarm)
- 论文代码：本项目是相关论文的实现代码

---

## 📄 许可证

请查看项目根目录的LICENSE文件（如果存在）。

---

*最后更新：2024年*

