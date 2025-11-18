# GDesigner 源码学习路径指南

## 🎯 学习目标
通过系统化的学习路径，从基础概念到核心实现，全面理解GDesigner的工作原理。

---

## 📚 推荐学习顺序

### 第一阶段：基础概念（理解核心抽象）

#### 1. **从 Node 开始** ⭐⭐⭐⭐⭐
**文件**: `GDesigner/graph/node.py`

**为什么先看这个？**
- Node是项目最基础的抽象，所有智能体都继承自它
- 理解Node的结构有助于理解整个系统的工作方式

**重点理解**：
- `spatial_predecessors/successors` 和 `temporal_predecessors/successors` 的区别
- `get_spatial_info()` 和 `get_temporal_info()` 如何获取信息
- `execute()` 和 `async_execute()` 的执行流程
- `_execute()` 和 `_async_execute()` 是抽象方法，需要子类实现

**学习建议**：
- 先看类的注释和属性定义
- 理解空间连接和时间连接的概念
- 理解输入输出的流转过程

---

#### 2. **理解 LLM 接口** ⭐⭐⭐⭐
**文件**: 
- `GDesigner/llm/llm.py` (抽象基类)
- `GDesigner/llm/gpt_chat.py` (具体实现)
- `GDesigner/llm/format.py` (消息格式)

**为什么看这个？**
- 所有智能体都需要调用LLM，这是基础能力
- 理解LLM接口有助于理解智能体如何工作

**重点理解**：
- `LLM` 抽象基类定义了 `gen()` 和 `agen()` 方法
- `GPTChat` 是具体实现，调用OpenAI API
- `Message` 类的结构（role, content）
- 注册机制：`@LLMRegistry.register()`

**学习建议**：
- 先看 `llm.py` 了解接口定义
- 再看 `gpt_chat.py` 看具体实现
- 理解注册机制如何工作

---

#### 3. **理解 Prompt 系统** ⭐⭐⭐⭐
**文件**:
- `GDesigner/prompt/prompt_set.py` (抽象基类)
- `GDesigner/prompt/mmlu_prompt_set.py` (具体实现示例)
- `GDesigner/prompt/common.py` (通用函数)

**为什么看这个？**
- Prompt决定了智能体的行为方式
- 理解如何组织提示词对理解智能体很重要

**重点理解**：
- `PromptSet` 定义了各种提示词方法
- `get_role()` 返回角色描述
- `get_role_connection()` 定义角色间的连接关系
- 不同领域（MMLU、HumanEval、GSM8K）有不同的PromptSet实现

**学习建议**：
- 先看 `prompt_set.py` 了解接口
- 再看一个具体实现（如 `mmlu_prompt_set.py`）
- 理解角色和连接关系的概念

---

### 第二阶段：智能体实现（理解具体工作方式）

#### 4. **学习 AnalyzeAgent** ⭐⭐⭐⭐⭐
**文件**: `GDesigner/agents/analyze_agent.py`

**为什么看这个？**
- 这是最常用的智能体类型
- 展示了如何实现Node的抽象方法
- 展示了如何处理空间和时间信息

**重点理解**：
- `_process_inputs()` 如何整合原始输入、空间信息、时间信息
- `_async_execute()` 如何调用LLM并返回结果
- 如何处理"Fake"角色（对抗性智能体）
- Wiki搜索的集成方式

**学习建议**：
- 仔细看 `_process_inputs()` 的逻辑
- 理解如何构建system_prompt和user_prompt
- 理解spatial_str和temporal_str的构建

---

#### 5. **学习其他智能体** ⭐⭐⭐
**文件**:
- `GDesigner/agents/code_writing.py`
- `GDesigner/agents/math_solver.py`
- `GDesigner/agents/final_decision.py`

**为什么看这些？**
- 了解不同类型智能体的实现差异
- 理解最终决策节点如何工作

**重点理解**：
- 不同智能体的 `_process_inputs()` 实现差异
- `FinalDecision` 的多种决策策略
- 如何整合多个智能体的输出

---

#### 6. **理解注册机制** ⭐⭐⭐
**文件**:
- `GDesigner/agents/agent_registry.py`
- `GDesigner/llm/llm_registry.py`
- `GDesigner/prompt/prompt_set_registry.py`

**为什么看这个？**
- 理解如何通过名称获取实例
- 理解工厂模式的应用

**重点理解**：
- `@AgentRegistry.register()` 装饰器的作用
- `AgentRegistry.get()` 如何创建实例
- 注册机制如何支持依赖注入

---

### 第三阶段：图结构核心（理解系统架构）

#### 7. **深入理解 Graph 类** ⭐⭐⭐⭐⭐
**文件**: `GDesigner/graph/graph.py`

**这是最核心的文件！建议分多次学习：**

**7.1 初始化部分（__init__）**
- 理解如何创建节点
- 理解 `potential_spatial_edges` 和 `potential_temporal_edges`
- 理解 `spatial_logits`、`spatial_masks`、`temporal_logits`、`temporal_masks`
- 理解GCN和MLP的初始化

**7.2 图构建部分**
- `construct_adj_matrix()`: 如何基于角色连接构建邻接矩阵
- `construct_features()`: 如何基于角色描述构建节点特征
- `construct_new_features()`: 如何加入查询特征

**7.3 连接构建部分**
- `construct_spatial_connection()`: 如何构建空间连接
  - 理解logits到概率的转换
  - 理解循环检测
  - 理解mask的作用
- `construct_temporal_connection()`: 如何构建时间连接
  - 理解为什么只在round > 0时构建

**7.4 执行部分**
- `run()`: 同步执行流程
  - 拓扑排序执行
  - 如何更新记忆
- `arun()`: 异步执行流程（使用GCN优化）
  - GCN如何计算连接logits
  - 异步执行的优势

**学习建议**：
- 这是最复杂的文件，建议分块学习
- 可以画图理解连接构建过程
- 可以单步调试理解执行流程

---

#### 8. **理解 GCN 优化** ⭐⭐⭐⭐
**文件**: `GDesigner/gnn/gcn.py`

**为什么看这个？**
- 理解如何用图神经网络优化连接
- 理解节点特征如何影响连接概率

**重点理解**：
- GCN的输入输出
- 如何将节点特征转换为连接logits
- MLP的作用

**学习建议**：
- 如果对GNN不熟悉，可以先了解GCN的基本原理
- 理解特征维度的变化

---

### 第四阶段：实验和工具（理解如何使用）

#### 9. **学习实验脚本** ⭐⭐⭐⭐
**文件**: `experiments/run_mmlu.py`

**为什么看这个？**
- 理解如何实际使用Graph
- 理解各种参数的含义
- 理解不同拓扑模式的实现

**重点理解**：
- `get_kwargs()` 函数如何生成不同拓扑的mask
- 如何创建Graph实例
- 训练和评估流程

**学习建议**：
- 可以运行这个脚本，观察实际运行
- 尝试修改参数，观察效果

---

#### 10. **学习数据集加载** ⭐⭐⭐
**文件**: `datasets/mmlu_dataset.py`

**为什么看这个？**
- 理解数据格式
- 理解如何加载和预处理数据

---

#### 11. **学习训练和评估** ⭐⭐⭐⭐
**文件**:
- `experiments/train_mmlu.py`
- `experiments/evaluate_mmlu.py`

**为什么看这个？**
- 理解如何优化连接参数
- 理解如何评估性能

**重点理解**：
- 训练循环
- 梯度计算和更新
- 评估指标

---

## 🗺️ 完整学习路径图

```
第一阶段：基础概念
├── 1. Node (graph/node.py) ⭐⭐⭐⭐⭐
├── 2. LLM接口 (llm/) ⭐⭐⭐⭐
└── 3. Prompt系统 (prompt/) ⭐⭐⭐⭐

第二阶段：智能体实现
├── 4. AnalyzeAgent (agents/analyze_agent.py) ⭐⭐⭐⭐⭐
├── 5. 其他智能体 (agents/) ⭐⭐⭐
└── 6. 注册机制 (registry文件) ⭐⭐⭐

第三阶段：图结构核心
├── 7. Graph类 (graph/graph.py) ⭐⭐⭐⭐⭐
└── 8. GCN优化 (gnn/gcn.py) ⭐⭐⭐⭐

第四阶段：实验和工具
├── 9. 实验脚本 (experiments/run_mmlu.py) ⭐⭐⭐⭐
├── 10. 数据集加载 (datasets/) ⭐⭐⭐
└── 11. 训练评估 (experiments/) ⭐⭐⭐⭐
```

---

## 💡 学习建议

### 1. 循序渐进
- 不要跳过前面的阶段直接看Graph
- 每个阶段都要理解透彻再进入下一阶段

### 2. 动手实践
- 每看完一个文件，尝试运行相关代码
- 修改参数，观察效果
- 添加print语句，观察数据流转

### 3. 画图理解
- 画出Node的连接关系
- 画出Graph的执行流程
- 画出GCN的计算过程

### 4. 调试技巧
- 使用IDE的调试功能
- 在关键位置添加断点
- 观察变量的值

### 5. 阅读顺序建议
```
Day 1: Node + LLM接口
Day 2: Prompt系统 + AnalyzeAgent
Day 3: Graph的初始化和连接构建
Day 4: Graph的执行流程
Day 5: GCN优化
Day 6: 实验脚本和训练评估
```

---

## 🔍 关键概念速查

### 空间连接 vs 时间连接
- **空间连接（Spatial）**: 同一轮次内节点间的连接
- **时间连接（Temporal）**: 跨轮次的连接，允许访问上一轮的信息

### Logits vs Masks
- **Logits**: 连接的概率分数（可训练或GCN计算）
- **Masks**: 固定掩码，控制哪些连接可以存在

### 角色（Role）
- 每个节点有一个角色
- 角色决定提示词和连接偏好
- 通过PromptSet定义

### 拓扑模式
- **FullConnected**: 全连接
- **Chain**: 链式
- **Star**: 星型
- **Mesh**: 网格
- 等等...

---

## 📖 推荐阅读顺序（详细版）

### 第一次阅读（理解整体架构）
1. `graph/node.py` - 理解Node基础
2. `graph/graph.py` - 快速浏览，理解整体结构
3. `experiments/run_mmlu.py` - 看如何使用

### 第二次阅读（深入理解）
1. `llm/` - 理解LLM接口
2. `prompt/` - 理解Prompt系统
3. `agents/analyze_agent.py` - 理解智能体实现
4. `graph/graph.py` - 深入理解每个方法

### 第三次阅读（优化机制）
1. `gnn/gcn.py` - 理解GCN优化
2. `experiments/train_mmlu.py` - 理解训练流程
3. `graph/graph.py` - 理解arun中的GCN应用

---

## 🎓 学习检查清单

完成每个阶段后，检查自己是否理解：

### 第一阶段检查
- [ ] 能解释Node的空间连接和时间连接的区别
- [ ] 能解释LLM接口如何工作
- [ ] 能解释PromptSet的作用

### 第二阶段检查
- [ ] 能解释AnalyzeAgent如何处理输入
- [ ] 能解释如何添加新的智能体类型
- [ ] 能解释注册机制如何工作

### 第三阶段检查
- [ ] 能解释Graph如何构建连接
- [ ] 能解释拓扑排序执行流程
- [ ] 能解释GCN如何优化连接

### 第四阶段检查
- [ ] 能运行实验脚本
- [ ] 能修改参数并观察效果
- [ ] 能理解训练和评估流程

---

## 🚀 下一步

完成学习后，可以尝试：
1. 添加新的智能体类型
2. 添加新的数据集支持
3. 修改GCN结构
4. 尝试新的拓扑模式
5. 优化训练流程

---

**祝你学习愉快！** 🎉


