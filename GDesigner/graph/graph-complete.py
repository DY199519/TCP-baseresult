import shortuuid
import asyncio
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np

# 移除 torch 和 GNN 相关依赖
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

# 引入 Genome 定义 (可选，如果不存在则 to_genome 方法会返回 None)
try:
    from GDesigner.evolution.genome import SystemGenome
except ImportError:
    SystemGenome = None

class Graph(ABC):
    """
    TPC-MSF Version: A deterministic, executable graph container.
    
    This class manages the execution of a network of nodes. Unlike the previous version,
    it does NOT handle topology generation (GNN/VGAE). Instead, it receives a fixed
    topology and prompt configuration from the Evolutionary Engine (MASExecutor)
    and executes it efficiently.
    """

    def __init__(self, 
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 node_kwargs: List[Dict] = None,
                 # 移除了所有 GNN 相关参数: optimized_spatial, masks, probabilities 等
                 ):
        
        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        
        # 决策节点保持不变
        self.decision_node: Node = AgentRegistry.get(decision_method, **{"domain": self.domain, "llm_name": self.llm_name})
        self.nodes: Dict[str, Node] = {}
        
        # 节点参数，可能包含由演化算法生成的 custom_constraint
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        # 初始化节点 (此时节点是孤立的，连接将由外部 Executor 通过 add_edge 设置)
        self.init_nodes()
        
        self.prompt_set = PromptSetRegistry.get(domain)
        
        # 移除了: construct_adj_matrix, construct_features, init_potential_edges

    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among {[n.id for n in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        # 保持原有 ID 生成逻辑，如果 node 已有 ID 则复用（用于 Genome 映射）
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        如果 node_kwargs 中包含 'custom_constraint'，AgentRegistry 会将其传递给 Agent 实例。
        这是 TPC-MSF 中 Prompt 变异生效的关键入口。
        """
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                # 注入全局配置
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                
                # 实例化 Agent (kwargs 中的 custom_constraint 会被 Agent.__init__ 接收)
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def add_edge(self, u_id: str, v_id: str, edge_type: str):
        """
        手动添加边。在 TPC-MSF 中，拓扑结构由 MASExecutor 根据 Genome 确定后调用此方法构建。
        替代了原有的 construct_spatial_connection 概率采样。
        
        Args:
            u_id: 源节点ID
            v_id: 目标节点ID
            edge_type: 边类型，必须是 "spatial" 或 "temporal"
            
        Raises:
            ValueError: 当节点不存在或边类型无效时抛出异常
        """
        if u_id not in self.nodes:
            raise ValueError(f"Source node '{u_id}' not found in graph. Available nodes: {list(self.nodes.keys())}")
        if v_id not in self.nodes:
            raise ValueError(f"Target node '{v_id}' not found in graph. Available nodes: {list(self.nodes.keys())}")
            
        u_node = self.nodes[u_id]
        v_node = self.nodes[v_id]
        
        if edge_type == "spatial":
            u_node.add_successor(v_node, "spatial")
        elif edge_type == "temporal":
            u_node.add_successor(v_node, "temporal")
        else:
            raise ValueError(f"Invalid edge_type '{edge_type}'. Must be 'spatial' or 'temporal'")

    def to_genome(self):
        """
        Phase 2.2: 从当前 Graph 实例提取 SystemGenome。
        用于将运行时的图状态转回基因组，以便进行后续的演化操作。
        
        Returns:
            SystemGenome: 如果 SystemGenome 类可用，返回基因组对象；否则返回 None
            
        Raises:
            ImportError: 如果 SystemGenome 类未定义
        """
        if SystemGenome is None:
            raise ImportError(
                "SystemGenome class not found. Please create GDesigner/evolution/genome.py "
                "with SystemGenome class definition."
            )
            
        topo = []
        prompts = {}

        # 1. 提取 Spatial 连接 (拓扑结构)
        for node_id, node in self.nodes.items():
            for successor in node.spatial_successors:
                # 只记录属于 nodes 的边（排除 decision_node）
                if successor.id in self.nodes:
                    topo.append((node_id, successor.id, "spatial"))
            
            # 2. 提取 Prompt (作为基因的一部分)
            # 优先提取 custom_constraint（如果存在，说明是变异后的 prompt）
            # 否则使用 constraint 属性
            if hasattr(node, 'custom_constraint') and node.custom_constraint:
                prompts[node_id] = node.custom_constraint
            else:
                prompts[node_id] = getattr(node, 'constraint', "")

        # 3. 提取 Temporal 连接 (记忆路由)
        for node_id, node in self.nodes.items():
            for successor in node.temporal_successors:
                # 只记录属于 nodes 的边（排除 decision_node）
                if successor.id in self.nodes:
                    topo.append((node_id, successor.id, "temporal"))

        return SystemGenome.create(topo, prompts)

    def connect_decision_node(self):
        """将所有节点连接到决策节点 (通常在执行最后一步调用)"""
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def update_memory(self):
        for id, node in self.nodes.items():
            node.update_memory()
            
    # -------------------------------------------------------------------------
    # 执行逻辑 (Sync & Async)
    # -------------------------------------------------------------------------

    def run(self, inputs: Any, 
            num_rounds: int = 3, 
            max_tries: int = 3, 
            max_time: int = 600) -> List[Any]:
        """
        同步运行图。保留此方法用于调试或简单场景。
        """
        for round in range(num_rounds):
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) 
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                        tries += 1
                
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers

    async def arun(self, input: Dict[str, str], 
                   num_rounds: int = 3, 
                   max_tries: int = 3, 
                   max_time: int = 600) -> List[Any]:
        """
        异步运行图 (TPC-MSF 核心执行方法)。
        实现了真正的"层级并发" (Batch Parallelism)，而非逐个节点的异步调用。
        这对提高种群评估速度至关重要。
        
        Args:
            input: 输入字典，通常包含 'task' 键
            num_rounds: 执行轮数
            max_tries: 每个节点执行失败后的最大重试次数
            max_time: 每个节点执行的最大超时时间（秒）
            
        Returns:
            List[Any]: 最终答案列表
        """
        for round in range(num_rounds):
            # 计算初始入度
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            
            # 队列中存储的是可以直接运行的节点ID
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
            
            while zero_in_degree_queue:
                # [关键改动]：一次性取出当前所有可执行节点 (Batch)
                current_batch_ids = list(zero_in_degree_queue)
                zero_in_degree_queue.clear() # 清空队列，为下一层级准备
                
                # 构建并发任务列表（带重试和超时机制）
                tasks = []
                
                for node_id in current_batch_ids:
                    node = self.nodes[node_id]
                    
                    # 创建带超时和重试的包装任务
                    async def execute_with_retry(node_id: str, node: Node, tries_left: int):
                        """带重试机制的节点执行函数"""
                        last_exception = None
                        for attempt in range(tries_left):
                            try:
                                # 使用 asyncio.wait_for 实现超时控制
                                return await asyncio.wait_for(
                                    node.async_execute(input),
                                    timeout=max_time
                                )
                            except asyncio.TimeoutError:
                                last_exception = TimeoutError(
                                    f"Node {node_id} execution timed out after {max_time}s (attempt {attempt + 1}/{tries_left})"
                                )
                                if attempt < tries_left - 1:
                                    print(f"Warning: {last_exception}, retrying...")
                            except Exception as e:
                                last_exception = e
                                if attempt < tries_left - 1:
                                    print(f"Error during execution of node {node_id} (attempt {attempt + 1}/{tries_left}): {e}")
                        
                        # 所有重试都失败，记录错误但不抛出异常（让其他节点继续执行）
                        print(f"Error: Node {node_id} failed after {tries_left} attempts. Last error: {last_exception}")
                        return last_exception
                    
                    task = execute_with_retry(node_id, node, max_tries)
                    tasks.append(task)
                
                # [关键改动]：并发执行当前层级的所有节点
                if tasks:
                    # return_exceptions=True 防止单个节点报错导致整个图崩溃
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 检查并记录执行结果中的异常
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            node_id = current_batch_ids[i]
                            print(f"Warning: Node {node_id} execution failed: {result}")

                # [关键改动]：批次执行完后，统一更新后续节点入度
                # 只有当父节点执行完，子节点才可能进入下一轮队列
                for node_id in current_batch_ids:
                    node = self.nodes[node_id]
                    for successor in node.spatial_successors:
                        if successor.id not in in_degree: 
                            continue
                        
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        # 决策节点逻辑（带超时和重试）
        self.connect_decision_node()
        try:
            await asyncio.wait_for(
                self.decision_node.async_execute(input),
                timeout=max_time
            )
        except asyncio.TimeoutError:
            print(f"Warning: Decision node execution timed out after {max_time}s")
        except Exception as e:
            print(f"Error during execution of decision node: {e}")
        
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers