import shortuuid
import asyncio
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np

# 移除 torch 和 GNN 相关依赖
from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry

# 引入 Genome 定义 (暂时注释掉，等演化引擎准备好后再启用)
# try:
#     from GDesigner.evolution.genome import SystemGenome
# except ImportError:
#     SystemGenome = None

class Graph(ABC):
    """
    可执行的图容器，用于管理和执行节点网络。
    
    支持固定拓扑结构（全连接、链式、星型等），可以独立运行，不依赖演化引擎。
    """

    def __init__(self, 
                 domain: str,
                 llm_name: Optional[str],
                 agent_names: List[str],
                 decision_method: str,
                 node_kwargs: List[Dict] = None,
                 topology_type: Optional[str] = None,
                 ):
        """
        初始化图容器。
        
        Args:
            domain: 领域名称
            llm_name: LLM名称
            agent_names: Agent名称列表
            decision_method: 决策方法
            node_kwargs: 节点参数列表（可选）
            topology_type: 拓扑类型（可选），如果提供则自动建立拓扑
                         可选值: "full_connected", "chain", "star", "none"
                         如果为None，则需要手动调用 build_fixed_topology() 或 add_edge() 建立连接
        """
        self.id: str = shortuuid.ShortUUID().random(length=4)
        self.domain: str = domain
        self.llm_name: str = llm_name
        self.agent_names: List[str] = agent_names
        
        # 决策节点保持不变
        self.decision_node: Node = AgentRegistry.get(decision_method, **{"domain": self.domain, "llm_name": self.llm_name})
        self.nodes: Dict[str, Node] = {}
        
        # 节点参数
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        # 初始化节点 (此时节点是孤立的，需要通过 build_fixed_topology 或 add_edge 设置连接)
        self.init_nodes()
        
        self.prompt_set = PromptSetRegistry.get(domain)
        
        # 如果提供了拓扑类型，自动建立拓扑（方便测试）
        if topology_type is not None:
            self.build_fixed_topology(topology_type)

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
        """
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                # 创建 kwargs 的副本，避免修改原始字典
                node_kwargs = kwargs.copy() if kwargs else {}
                # 注入全局配置
                node_kwargs["domain"] = self.domain
                node_kwargs["llm_name"] = self.llm_name
                
                # 实例化 Agent
                agent_instance = AgentRegistry.get(agent_name, **node_kwargs)
                self.add_node(agent_instance)

    def add_edge(self, u_id: str, v_id: str, edge_type: str):
        """
        手动添加边。用于构建图的拓扑结构。
        
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
    
    def build_fixed_topology(self, topology_type: str = "full_connected"):
        """
        构建固定拓扑结构。用于快速测试和流程验证。
        
        Args:
            topology_type: 拓扑类型，可选值：
                - "full_connected": 全连接（每个节点都连接到其他所有节点）
                - "chain": 链式（节点按顺序连接：0->1->2->...）
                - "star": 星型（第一个节点连接到所有其他节点）
                - "none": 不连接（节点之间无连接，只有到决策节点的连接）
        """
        node_ids = list(self.nodes.keys())
        if len(node_ids) == 0:
            return
        
        if topology_type == "full_connected":
            # 全连接：每个节点都连接到其他所有节点
            for i, u_id in enumerate(node_ids):
                for j, v_id in enumerate(node_ids):
                    if i != j:  # 不连接自己
                        self.add_edge(u_id, v_id, "spatial")
        
        elif topology_type == "chain":
            # 链式：0->1->2->...->n
            for i in range(len(node_ids) - 1):
                self.add_edge(node_ids[i], node_ids[i + 1], "spatial")
        
        elif topology_type == "star":
            # 星型：第一个节点连接到所有其他节点
            center_id = node_ids[0]
            for i in range(1, len(node_ids)):
                self.add_edge(center_id, node_ids[i], "spatial")
        
        elif topology_type == "none":
            # 不连接：节点之间无连接
            pass
        
        else:
            raise ValueError(
                f"Unknown topology_type '{topology_type}'. "
                f"Must be one of: 'full_connected', 'chain', 'star', 'none'"
            )

    # 暂时注释掉 to_genome 方法，等演化引擎准备好后再启用
    # def to_genome(self):
    #     """
    #     Phase 2.2: 从当前 Graph 实例提取 SystemGenome。
    #     用于将运行时的图状态转回基因组，以便进行后续的演化操作。
    #     """
    #     pass 

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
            max_time: int = 600) -> Tuple[List[Any], float]:
        """
        同步运行图。保留此方法用于调试或简单场景。
        
        Returns:
            Tuple[List[Any], float]: (final_answers, log_probs)
            为了保持与原始版本接口一致，log_probs 返回 0.0（占位符）
        """
        log_probs = 0.0  # 占位符，原始版本中用于 GNN 优化的 log_probs
        
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
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str, str], 
                   num_rounds: int = 3, 
                   max_tries: int = 3, 
                   max_time: int = 600) -> Tuple[List[Any], float]:
        """
        异步运行图 (TPC-MSF 核心执行方法)。
        实现了真正的"层级并发" (Batch Parallelism)，而非逐个节点的异步调用。
        这对提高种群评估速度至关重要。
        
        Returns:
            Tuple[List[Any], float]: (final_answers, log_probs)
            为了保持与原始版本接口一致，log_probs 返回 0.0（占位符）
        """
        log_probs = 0.0  # 占位符，原始版本中用于 GNN 优化的 log_probs
        
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
            
        return final_answers, log_probs