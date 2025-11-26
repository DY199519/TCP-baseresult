# TPC-MSF v3.1 实施计划

## 一、总体改动评估

### 模块改动量汇总

| 模块 | 改动程度 | 改动量 | 关键工作 |
|------|---------|--------|---------|
| **Agents** | 完全不变 | 0% | 仅使用方式改变，内部逻辑不变 |
| **AgentRegistry** | 完全不变 | 0% | 注册机制不变 |
| **LLM** | 小幅修改 | 10-15% | 添加 token 计数功能 |
| **Prompt** | **大幅修改** | **60-70%** | **新增 PromptMutator，实现动态重写** |
| **Graph** | 较大修改 | 60% | 保留执行逻辑，添加 Genome 转换，移除 GCN 优化 |
| **执行层** | **新增模块** | **100%** | **MASExecutor + 种群级并发支持** |
| **演化引擎** | 完全新增 | 100% | CoevolutionEngine + 任务感知初始化 |

**总工作量估算**：约 6-8 周（单人），考虑并发和 Prompt 重写的复杂度

---

## 二、详细实施计划

### Phase 0: 基础数据结构层（1-2 周）

#### 0.1 实现不可变基因组（SystemGenome）
**文件**：`GDesigner/evolution/genome.py`（新建）

**核心类**：
```python
@dataclass(frozen=True)
class SystemGenome:
    topology_dna: Tuple[Tuple[int, int, str], ...]  # (u, v, edge_type)
    prompts_dna: Tuple[Tuple[int, str], ...]       # (node_id, prompt_text)
    
    @staticmethod
    def create(topo, prompts):
        # 规范化排序，确保可哈希
        sorted_topo = tuple(sorted(topo, key=lambda x: (x[0], x[1], x[2])))
        sorted_prompts = tuple(sorted(prompts.items()))
        return SystemGenome(sorted_topo, sorted_prompts)
    
    def get_hash(self, config_id: str) -> str:
        # 版本化哈希，用于缓存
        raw = json.dumps({
            "schema_v": 1,
            "topo": self.topology_dna,
            "prompts": self.prompts_dna,
            "conf": config_id
        }, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()
```

**验收标准**：
- [ ] SystemGenome 不可变（frozen=True）
- [ ] 相同拓扑和 Prompt 生成相同 hash
- [ ] 可以正确序列化/反序列化

---

#### 0.2 实现执行结果（ExecutionResult）
**文件**：`GDesigner/evolution/result.py`（新建）

**核心类**：
```python
@dataclass(frozen=True)
class ExecutionResult:
    accuracy: float
    total_tokens: int
    complexity: float  # 基于图结构计算
    error_type: str = "none"  # "Timeout", "Invalid", "none"
    execution_time: float = 0.0
```

**验收标准**：
- [ ] 包含所有评估指标
- [ ] 不可变，可用于缓存键

---

#### 0.3 实现演化个体（Individual）
**文件**：`GDesigner/evolution/individual.py`（新建）

**核心类**：
```python
class Individual:
    def __init__(self, genome: SystemGenome):
        self.genome = genome
        self.fitness_opt: Optional[Tuple[float, float, float]] = None
        # (accuracy, -cost, -complexity) 用于 NSGA-II
        self.stats: Dict[str, Any] = {}
        self.cached_result: Optional[ExecutionResult] = None
```

**验收标准**：
- [ ] 可以存储 fitness 值
- [ ] 支持缓存执行结果

---

### Phase 1: Prompt 动态重写系统（2-3 周）⭐ **关键模块**

#### 1.1 实现 PromptMutator 核心类
**文件**：`GDesigner/prompt/prompt_mutator.py`（新建）

**核心功能**：
```python
class PromptMutator:
    """
    负责根据拓扑变化动态重写 Prompt，实现强耦合共演化
    """
    def __init__(self, llm_name: str):
        self.llm = LLMRegistry.get(llm_name)
    
    def mutate_for_topology_change(
        self,
        genome: SystemGenome,
        node_id: int,
        new_connections: List[Tuple[int, str]],  # [(target_id, edge_type), ...]
        removed_connections: List[Tuple[int, str]] = None
    ) -> str:
        """
        当节点连接变化时，重写该节点的 Prompt
        
        关键逻辑：
        1. 分析当前 Prompt
        2. 分析新的连接关系
        3. 使用 LLM 生成语义一致的 Prompt
        """
        current_prompt = dict(genome.prompts_dna).get(node_id, "")
        connection_desc = self._describe_connections(new_connections)
        
        rewrite_prompt = f"""
        原 Prompt: {current_prompt}
        
        新的连接关系: {connection_desc}
        - 新增连接: {new_connections}
        - 移除连接: {removed_connections or []}
        
        请重写 Prompt，使其：
        1. 保留原有的核心职责描述
        2. 明确说明如何处理新的连接关系
        3. 如果接收来自其他 Agent 的输入，说明如何处理
        4. 如果发送输出给其他 Agent，说明输出格式
        5. 确保语义与拓扑结构一致
        """
        new_prompt = self.llm.gen([{
            'role': 'system',
            'content': 'You are a prompt engineering expert.'
        }, {
            'role': 'user',
            'content': rewrite_prompt
        }])
        return new_prompt.strip()
    
    def _describe_connections(self, connections: List[Tuple[int, str]]) -> str:
        """将连接关系转换为自然语言描述"""
        # 实现连接描述逻辑
        pass
```

**验收标准**：
- [ ] 能够根据拓扑变化生成语义一致的 Prompt
- [ ] 生成的 Prompt 包含连接关系说明
- [ ] 保留原有 Prompt 的核心职责

---

#### 1.2 实现 Prompt 模板系统
**文件**：`GDesigner/prompt/prompt_template.py`（新建）

**功能**：
- 提供 Prompt 模板库
- 支持基于角色的模板选择
- 支持模板参数化（连接关系、角色等）

**验收标准**：
- [ ] 支持从模板生成基础 Prompt
- [ ] 支持模板参数化

---

#### 1.3 修改 Agent 初始化支持自定义 Prompt
**文件**：`GDesigner/agents/*.py`

**改动**：
```python
# 修改所有 Agent 的 __init__
def __init__(self, 
             id: str | None = None,
             role: str = None,
             domain: str = "",
             llm_name: str = "",
             custom_constraint: str = None):  # 新增参数
    super().__init__(id, "AgentName", domain, llm_name)
    self.llm = LLMRegistry.get(llm_name)
    self.prompt_set = PromptSetRegistry.get(domain)
    self.role = self.prompt_set.get_role() if role is None else role
    
    # 优先使用自定义 constraint，否则从 PromptSet 获取
    if custom_constraint is not None:
        self.constraint = custom_constraint
    else:
        self.constraint = self.prompt_set.get_constraint(self.role)
```

**影响文件**：
- `analyze_agent.py`
- `code_writing.py`
- `math_solver.py`
- `adversarial_agent.py`
- `final_decision.py`

**验收标准**：
- [ ] 所有 Agent 支持自定义 constraint
- [ ] 向后兼容（不传 custom_constraint 时使用默认）

---

### Phase 2: 执行引擎层（1-2 周）

#### 2.1 实现 MASExecutor
**文件**：`GDesigner/evolution/executor.py`（新建）

**核心功能**：
```python
class MASExecutor:
    """
    多智能体系统执行器
    负责将 Genome 实例化为 Graph 并执行
    """
    def __init__(self, 
                 domain: str,
                 llm_name: str,
                 max_turns: int = 10,
                 token_limit: int = 100000):
        self.domain = domain
        self.llm_name = llm_name
        self.max_turns = max_turns
        self.token_limit = token_limit
        self.cache: Dict[str, ExecutionResult] = {}
    
    async def run(self, 
                  genome: SystemGenome,
                  task_batch: List[Dict[str, str]]) -> ExecutionResult:
        """
        执行单个 Genome
        
        关键约束：
        1. max_turns: 轮次上限熔断
        2. token_limit: 成本熔断
        3. 缓存：相同 hash 直接返回
        """
        # 检查缓存
        cache_key = genome.get_hash(f"{self.domain}_{len(task_batch)}")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 从 Genome 实例化 Graph
        graph = self._genome_to_graph(genome)
        
        # 执行任务批次
        total_tokens = 0
        total_accuracy = 0.0
        error_type = "none"
        
        try:
            for task in task_batch:
                answers, log_probs = await graph.arun(
                    task,
                    num_rounds=self.max_turns
                )
                # 计算 accuracy（需要数据集评估函数）
                accuracy = self._evaluate_answer(answers, task)
                total_accuracy += accuracy
                
                # 累计 tokens（需要从 LLM 调用处获取）
                total_tokens += self._get_tokens_used()
                
                # 检查熔断
                if total_tokens > self.token_limit:
                    error_type = "TokenLimit"
                    break
                    
        except asyncio.TimeoutError:
            error_type = "Timeout"
        except Exception as e:
            error_type = "Invalid"
        
        # 计算复杂度
        complexity = self._compute_complexity(genome)
        
        result = ExecutionResult(
            accuracy=total_accuracy / len(task_batch),
            total_tokens=total_tokens,
            complexity=complexity,
            error_type=error_type
        )
        
        # 缓存结果
        self.cache[cache_key] = result
        return result
    
    def _genome_to_graph(self, genome: SystemGenome) -> Graph:
        """
        从 Genome 实例化 Graph
        
        关键步骤：
        1. 提取拓扑结构
        2. 创建节点（使用 custom_constraint）
        3. 构建连接
        4. 设置决策节点
        """
        # 实现 Genome -> Graph 转换
        pass
    
    async def evaluate_population(
        self,
        genomes: List[SystemGenome],
        task_batch: List[Dict[str, str]],
        max_concurrent: int = 10
    ) -> List[ExecutionResult]:
        """
        并发评估整个种群
        
        关键优化：
        1. 使用 asyncio.Semaphore 控制并发数
        2. 处理 API rate limit
        3. 实现重试机制
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_limit(genome):
            async with semaphore:
                return await self.run(genome, task_batch)
        
        tasks = [evaluate_with_limit(g) for g in genomes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                final_results.append(ExecutionResult(
                    accuracy=0.0,
                    total_tokens=0,
                    complexity=0.0,
                    error_type="Exception"
                ))
            else:
                final_results.append(r)
        
        return final_results
```

**验收标准**：
- [ ] 能够从 Genome 实例化 Graph
- [ ] 支持熔断机制（turns, tokens）
- [ ] 支持结果缓存
- [ ] 支持种群级并发评估
- [ ] 处理并发限制和异常

---

#### 2.2 实现 Graph 的 Genome 转换方法
**文件**：`GDesigner/graph/graph.py`

**新增方法**：
```python
def to_genome(self) -> SystemGenome:
    """
    从 Graph 提取 Genome
    
    关键步骤：
    1. 提取拓扑结构（spatial + temporal）
    2. 提取所有节点的 Prompt
    3. 规范化并创建 SystemGenome
    """
    topo = []
    prompts = {}
    
    # 提取 spatial 连接
    for node_id, node in self.nodes.items():
        for successor in node.spatial_successors:
            topo.append((node_id, successor.id, "spatial"))
        # 提取 Prompt（从 constraint）
        prompts[node_id] = node.constraint
    
    # 提取 temporal 连接
    for node_id, node in self.nodes.items():
        for successor in node.temporal_successors:
            topo.append((node_id, successor.id, "temporal"))
    
    return SystemGenome.create(topo, prompts)
```

**验收标准**：
- [ ] 能够正确提取拓扑结构
- [ ] 能够正确提取 Prompt
- [ ] 生成的 Genome 可以正确转换回 Graph

---

#### 2.3 修改 LLM 接口添加 Token 计数
**文件**：`GDesigner/llm/llm.py`, `GDesigner/llm/gpt_chat2.py`

**改动**：
```python
class LLM(ABC):
    def __init__(self):
        self.last_token_count = 0  # 记录最后一次调用的 token 数
    
    async def agen(self, ...) -> Union[List[str], str]:
        # 调用后更新 self.last_token_count
        pass
```

**验收标准**：
- [ ] 所有 LLM 实现都记录 token 使用量
- [ ] 可以从 Executor 获取 token 计数

---

### Phase 3: 演化引擎核心（2-3 周）

#### 3.1 实现任务感知的景观探测
**文件**：`GDesigner/evolution/landscape_probe.py`（新建）

**核心功能**：
```python
class LandscapeProbe:
    """
    景观探测：替代 GNN 的任务感知功能
    """
    def __init__(self, executor: MASExecutor):
        self.executor = executor
    
    async def probe(
        self,
        task_batch: List[Dict[str, str]],
        num_prototypes: int = 5
    ) -> Dict[str, Any]:
        """
        探测任务景观
        
        返回：
        - ruggedness: 崎岖度（fitness 方差）
        - task_role_match: 任务与角色的匹配度
        - recommended_params: 推荐的初始参数
        """
        # 1. 提取任务特征（替代 GNN 的 query embedding）
        task_embedding = self._extract_task_features(task_batch[0]['task'])
        
        # 2. 生成差异化原型
        prototypes = self._generate_prototypes(
            num_prototypes,
            task_aware=True,  # 考虑任务特征
            task_embedding=task_embedding
        )
        
        # 3. 评估原型
        results = await self.executor.evaluate_population(
            prototypes,
            task_batch[:3]  # 使用小批次快速探测
        )
        
        # 4. 计算崎岖度
        fitness_values = [r.accuracy for r in results]
        ruggedness = np.std(fitness_values)
        
        # 5. 计算任务-角色匹配度
        task_role_match = self._compute_task_role_match(
            task_embedding,
            prototypes
        )
        
        # 6. 推荐初始参数
        recommended_params = self._recommend_params(
            ruggedness,
            task_role_match
        )
        
        return {
            'ruggedness': ruggedness,
            'task_role_match': task_role_match,
            'recommended_params': recommended_params,
            'prototypes': prototypes
        }
    
    def _extract_task_features(self, task: str) -> np.ndarray:
        """
        提取任务特征（替代 GNN 的 query embedding）
        """
        from GDesigner.llm.profile_embedding import get_sentence_embedding
        return get_sentence_embedding(task)
    
    def _compute_task_role_match(
        self,
        task_embedding: np.ndarray,
        prototypes: List[SystemGenome]
    ) -> float:
        """
        计算任务与角色的匹配度
        
        逻辑：
        1. 提取所有原型中的角色描述
        2. 计算角色描述与任务特征的相似度
        3. 返回平均相似度
        """
        # 实现匹配度计算
        pass
    
    def _recommend_params(
        self,
        ruggedness: float,
        task_role_match: float
    ) -> Dict[str, float]:
        """
        根据景观特征推荐初始参数
        """
        if ruggedness > 0.2 and task_role_match < 0.5:
            # 崎岖且不匹配 -> 激进探索
            return {
                'mutation_rate': 0.3,
                'crossover_rate': 0.8,
                'population_size': 50
            }
        elif ruggedness < 0.1 and task_role_match > 0.7:
            # 平坦且匹配 -> 精细开发
            return {
                'mutation_rate': 0.05,
                'crossover_rate': 0.5,
                'population_size': 30
            }
        else:
            # 中等参数
            return {
                'mutation_rate': 0.15,
                'crossover_rate': 0.6,
                'population_size': 40
            }
```

**验收标准**：
- [ ] 能够正确计算崎岖度
- [ ] 能够计算任务-角色匹配度
- [ ] 能够推荐合理的初始参数
- [ ] 生成的原型具有差异化

---

#### 3.2 实现强耦合共演化引擎
**文件**：`GDesigner/evolution/coevolution_engine.py`（新建）

**核心功能**：
```python
class CoevolutionEngine:
    """
    强耦合共演化引擎
    """
    def __init__(self,
                 executor: MASExecutor,
                 prompt_mutator: PromptMutator,
                 pop_size: int = 50,
                 elite_size: int = 5):
        self.executor = executor
        self.prompt_mutator = prompt_mutator
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.population: List[Individual] = []
        self.mutation_rate = 0.1
        self.crossover_rate = 0.6
    
    async def adaptive_initialize(self, task_batch: List[Dict[str, str]]):
        """
        Phase 0: 自适应初始化（景观探测）
        """
        probe = LandscapeProbe(self.executor)
        probe_result = await probe.probe(task_batch)
        
        # 设置初始参数
        params = probe_result['recommended_params']
        self.mutation_rate = params['mutation_rate']
        self.crossover_rate = params['crossover_rate']
        self.pop_size = params['population_size']
        
        # 使用探测到的原型填充初始种群
        prototypes = probe_result['prototypes']
        self.population = [Individual(g) for g in prototypes]
        
        # 填充随机个体
        self._fill_random_population()
        
        print(f"Landscape Probe Complete:")
        print(f"  Ruggedness: {probe_result['ruggedness']:.3f}")
        print(f"  Task-Role Match: {probe_result['task_role_match']:.3f}")
        print(f"  Mutation Rate: {self.mutation_rate}")
    
    def _co_mutate(self, genome: SystemGenome) -> SystemGenome:
        """
        强耦合变异：G 和 P 必须协同修改
        """
        topo = list(genome.topology_dna)
        prompts = dict(genome.prompts_dna)
        
        if np.random.random() < 0.5:
            # A. 结构驱动：加节点必加 Prompt
            new_id = max(prompts.keys()) + 1 if prompts else 0
            target = np.random.choice(list(prompts.keys())) if prompts else None
            
            if target is not None:
                topo.append((new_id, target, "spatial"))
                # 关键：必须同步生成 Prompt
                new_connections = [(target, "spatial")]
                prompts[new_id] = self.prompt_mutator.mutate_for_topology_change(
                    genome, new_id, new_connections
                )
        else:
            # B. 语义驱动：改 Prompt 微调连接
            node_id = np.random.choice(list(prompts.keys()))
            
            # 修改 Prompt（使用 LLM 重写）
            current_connections = [
                (v, t) for u, v, t in topo if u == node_id
            ]
            prompts[node_id] = self.prompt_mutator.mutate_for_topology_change(
                genome, node_id, current_connections
            )
            
            # 检查连接合理性，必要时添加连接
            if not any(u == node_id for u, v, t in topo):
                src = np.random.choice([n for n in prompts if n != node_id])
                topo.append((src, node_id, "spatial"))
        
        return SystemGenome.create(topo, prompts)
    
    def _structure_aware_crossover(
        self,
        parent1: SystemGenome,
        parent2: SystemGenome
    ) -> SystemGenome:
        """
        模块级交叉：交换 Agent Bundle（节点ID + Prompt + 连边）
        """
        # 实现模块级交叉
        pass
    
    def _repair_graph(self, genome: SystemGenome) -> SystemGenome:
        """
        修复算子：先修连通性，后罚 Fitness
        """
        # 1. 检查连通性
        # 2. 修复孤立节点
        # 3. 修复循环依赖
        pass
    
    async def next_generation(self, task_batch: List[Dict[str, str]]):
        """
        演化一代
        """
        # 1. 评估所有个体
        genomes = [ind.genome for ind in self.population]
        results = await self.executor.evaluate_population(genomes, task_batch)
        
        # 2. 更新 fitness
        for ind, result in zip(self.population, results):
            ind.fitness_opt = (
                result.accuracy,
                -result.total_tokens / 1000.0,  # 归一化成本
                -result.complexity
            )
            ind.cached_result = result
        
        # 3. NSGA-II 选择
        elites = self._nsga2_select(self.elite_size)
        
        # 4. 繁殖
        offspring = []
        parents = self._select_parents()
        for p1, p2 in parents:
            child_genome = self._structure_aware_crossover(p1.genome, p2.genome)
            
            # 强耦合变异
            if np.random.random() < self.mutation_rate:
                child_genome = self._co_mutate(child_genome)
            
            # 修复
            if not self._is_valid(child_genome):
                child_genome = self._repair_graph(child_genome)
            
            offspring.append(Individual(child_genome))
        
        # 5. 更新种群
        self.population = elites + offspring
```

**验收标准**：
- [ ] 能够完成景观探测初始化
- [ ] 强耦合变异正确同步 G 和 P
- [ ] NSGA-II 选择正确
- [ ] 修复算子能够修复无效图

---

#### 3.3 集成 NSGA-II
**文件**：`GDesigner/evolution/nsga2.py`（新建）

**功能**：
- 实现非支配排序
- 实现拥挤距离计算
- 实现多目标选择

**验收标准**：
- [ ] 能够正确进行非支配排序
- [ ] 能够计算拥挤距离
- [ ] 能够选择精英个体

---

### Phase 4: 集成与测试（1 周）

#### 4.1 集成所有组件
- 连接 CoevolutionEngine 和 MASExecutor
- 连接 PromptMutator 和演化引擎
- 实现主循环

#### 4.2 性能优化
- 优化并发执行
- 优化缓存策略
- 优化 Prompt 重写（减少 LLM 调用）

#### 4.3 测试验证
- 单元测试
- 集成测试
- 性能测试

---

## 三、关键技术难点

### 1. Prompt 动态重写的质量保证
**挑战**：确保 LLM 生成的 Prompt 语义正确且与拓扑一致

**解决方案**：
- 使用模板约束生成
- 实现 Prompt 验证逻辑
- 支持人工审核机制

### 2. 种群级并发执行
**挑战**：API rate limit、资源消耗、异常处理

**解决方案**：
- 使用 Semaphore 控制并发数
- 实现重试和退避策略
- 监控资源使用

### 3. 任务感知的替代方案
**挑战**：移除 GNN 后如何保持任务感知能力

**解决方案**：
- 使用 embedding 相似度计算
- 在景观探测阶段完成任务分析
- 将任务特征注入初始化参数

---

## 四、实施时间表

| 阶段 | 时间 | 关键交付物 |
|------|------|-----------|
| Phase 0 | 1-2 周 | SystemGenome, ExecutionResult, Individual |
| Phase 1 | 2-3 周 | PromptMutator, Agent 修改 |
| Phase 2 | 1-2 周 | MASExecutor, Graph 转换, LLM Token 计数 |
| Phase 3 | 2-3 周 | CoevolutionEngine, LandscapeProbe, NSGA-II |
| Phase 4 | 1 周 | 集成、测试、优化 |
| **总计** | **7-11 周** | 完整 TPC-MSF 系统 |

---

## 五、风险与应对

### 风险 1: Prompt 重写质量不稳定
**应对**：实现 Prompt 验证和模板约束

### 风险 2: 并发执行导致 API 限制
**应对**：实现智能并发控制和重试机制

### 风险 3: 演化收敛速度慢
**应对**：优化初始化和选择策略

---

## 六、验收标准

### 功能验收
- [ ] 能够完成景观探测初始化
- [ ] 能够进行强耦合共演化
- [ ] 能够并发评估种群
- [ ] Prompt 与拓扑语义一致

### 性能验收
- [ ] 一代演化时间 < 30 分钟（50 个体）
- [ ] 缓存命中率 > 50%
- [ ] 内存使用稳定

### 质量验收
- [ ] 代码覆盖率 > 80%
- [ ] 所有测试通过
- [ ] 文档完整

