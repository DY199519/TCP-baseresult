import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import argparse
from pathlib import Path
from typing import Union, Literal, List, Optional, Dict, Any

from GDesigner.graph.graph import Graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.evaluate_mmlu import evaluate
from GDesigner.utils.const import GDesigner_ROOT

# [EVOLUTION - Phase 3]: 演化引擎引用 (暂注释)
# from experiments.train_mmlu import train 
# from GDesigner.evolution.coevolution_engine import CoevolutionEngine (Future)
# from GDesigner.evolution.landscape_probe import LandscapeProbe (Future)


def parse_args():
    parser = argparse.ArgumentParser(description="MMLU Evaluation Script (Inference & Evolution)")

    # ==========================
    # 1. 基础推理参数 (Inference)
    # ==========================
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Chain', 'Star',
                                 'FakeFullConnected', 'FakeChain', 'FakeStar'],
                        help="Mode of operation / Initial Topology Seed")
    
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for evaluation")
    parser.add_argument('--llm_name', type=str, default="gpt-4o", help="Model name")
    parser.add_argument('--domain', type=str, default="mmlu", help="Domain/Dataset name")
    
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Agent roles in the graph')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Number of agents per role')
    
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="Inference depth (discussion rounds per question)")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="Decision method for the final node")
    
    parser.add_argument('--limit_questions', type=int, default=None,
                        help="Limit number of questions to evaluate (None = all)")

    # ==========================
    # 2. 演化参数占位 (TPC-MSF v3.1)
    # ==========================
    # [EVOLUTION] 种群规模
    # parser.add_argument('--pop_size', type=int, default=50, help="Population size for GA")
    
    # [EVOLUTION] 变异率
    # parser.add_argument('--mutation_rate', type=float, default=0.1, help="Probability of mutation")
    
    # [EVOLUTION] 交叉率
    # parser.add_argument('--crossover_rate', type=float, default=0.6, help="Probability of crossover")
    
    # [EVOLUTION] 演化代数
    # parser.add_argument('--generations', type=int, default=10, help="Number of generations to evolve")

    args = parser.parse_args()
    
    # [Fix 1] 安全的路径拼接 - 强制转换为 Path 对象
    result_path = Path(GDesigner_ROOT) / "result"
    os.makedirs(result_path, exist_ok=True)
    
    # 参数校验
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args


# 拓扑类型映射表
MODE_TO_TOPOLOGY: Dict[str, str] = {
    'FullConnected': 'full_connected',
    'Chain': 'chain',
    'Star': 'star',
    'DirectAnswer': 'none',
}


def mode_to_topology_type(mode: str) -> str:
    """
    映射 mode 参数到 Graph 类支持的拓扑类型
    
    Args:
        mode: 命令行传入的 mode 参数
        
    Returns:
        Graph 类支持的拓扑类型字符串
        
    Raises:
        ValueError: 当 mode 无法映射到有效拓扑类型时
    """
    # 处理 Fake 前缀
    if mode.startswith('Fake'):
        base_mode = mode.replace('Fake', '')
        if base_mode in MODE_TO_TOPOLOGY:
            return MODE_TO_TOPOLOGY[base_mode]
    
    # 直接映射
    if mode in MODE_TO_TOPOLOGY:
        return MODE_TO_TOPOLOGY[mode]
    
    # [Fix] 未知 mode 时抛出明确错误，而不是返回 None
    raise ValueError(
        f"Unknown mode '{mode}'. Supported modes: {list(MODE_TO_TOPOLOGY.keys())} "
        f"and their Fake variants (FakeFullConnected, FakeChain, FakeStar)"
    )


def get_node_kwargs(mode: str, N: int) -> List[Dict[str, Any]]:
    """
    根据 mode 生成节点配置列表
    
    Args:
        mode: 运行模式
        N: 节点数量
        
    Returns:
        长度为 N 的节点配置字典列表（DirectAnswer 模式除外，返回长度为 1）
        
    Raises:
        ValueError: DirectAnswer 模式要求 N == 1，或 Fake 模式在 N == 1 时给出警告
    """
    # DirectAnswer 模式：要求必须是单节点
    if mode == 'DirectAnswer':
        if N != 1:
            raise ValueError(
                f"DirectAnswer mode requires exactly 1 agent, but got {N}. "
                f"Please adjust --agent_nums to [1] when using DirectAnswer mode."
            )
        return [{'role': 'Normal'}]
    
    # Fake 模式：交替分配 Fake/Normal 角色，近似 50/50 分配
    if 'Fake' in mode:
        if N == 1:
            print(f"Warning: Fake mode with only 1 agent has no effect. Using Normal role.")
            return [{'role': 'Normal'}]
        
        # 更清晰的分配策略：奇数索引为 Fake，偶数索引为 Normal
        # 这确保了近似 50/50 的分配（如果 N 为奇数，Normal 会多一个）
        node_kwargs = [
            {'role': 'Fake'} if i % 2 == 1 else {'role': 'Normal'}
            for i in range(N)
        ]
        return node_kwargs
    
    # 默认模式：返回空配置，让 Agent 使用默认角色
    return [{} for _ in range(N)]


def check_dataset_exists() -> bool:
    """检查 MMLU 数据集是否已存在"""
    data_path = Path(GDesigner_ROOT) / "datasets" / "MMLU" / "data"
    if not data_path.exists():
        return False
    try:
        # 检查目录是否包含文件
        return any(data_path.iterdir())
    except (PermissionError, OSError) as e:
        print(f"Warning: Cannot access data directory {data_path}: {e}")
        return False


async def main():
    args = parse_args()
    
    # ==========================
    # 1. Agent 配置
    # ==========================
    agent_names = [
        name 
        for name, num in zip(args.agent_names, args.agent_nums) 
        for _ in range(num)
    ]
    
    # [Fix 2] DirectAnswer 模式下强制限制为单 agent
    if args.mode == 'DirectAnswer':
        if len(agent_names) > 1:
            print(f"Notice: DirectAnswer mode detected. Limiting agents from {len(agent_names)} to 1.")
        agent_names = agent_names[:1]
    
    # ==========================
    # 2. 拓扑与节点配置
    # ==========================
    topology_type = mode_to_topology_type(args.mode)
    node_kwargs = get_node_kwargs(args.mode, len(agent_names))
    
    # 配置信息输出
    print("=" * 50)
    print("GDesigner Inference Mode")
    print("=" * 50)
    print(f"  Topology: {args.mode} -> {topology_type}")
    print(f"  Agents: {len(agent_names)} ({args.agent_names})")
    print(f"  LLM: {args.llm_name}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Decision Method: {args.decision_method}")
    print("=" * 50)
    
    # ==========================
    # 3. 初始化推理图
    # ==========================
    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        node_kwargs=node_kwargs,
        topology_type=topology_type
    )
    
    # ==========================
    # 4. 数据集准备
    # ==========================
    # [Fix] 仅在数据不存在时下载
    if not check_dataset_exists():
        print("Downloading MMLU dataset...")
        try:
            download()
            # 下载后再次检查
            if not check_dataset_exists():
                raise RuntimeError("Dataset download completed but data directory is still empty.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise RuntimeError(f"Failed to download or extract MMLU dataset: {e}") from e
    else:
        print("MMLU dataset already exists, skipping download.")
    
    # 初始化数据集并验证
    try:
        dataset_val = MMLUDataset('val')
        if len(dataset_val) == 0:
            raise RuntimeError("MMLU validation dataset is empty. Please check the data directory.")
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        raise RuntimeError(f"Failed to load MMLU dataset: {e}") from e
    
    # 确定评估数量
    limit_questions = args.limit_questions
    if limit_questions is None:
        limit_questions = len(dataset_val)
    print(f"Evaluating on {limit_questions} questions (dataset size: {len(dataset_val)})")
    
    # ==========================
    # [EVOLUTION HOOK] 演化引擎入口 (Future)
    # ==========================
    # if args.generations > 0:
    #     engine = CoevolutionEngine(...)
    #     best_genome = await engine.evolve(...)
    #     graph = best_genome.to_graph()

    # ==========================
    # 5. 执行评估
    # ==========================
    print("\nStarting evaluation...")
    score = await evaluate(
        graph=graph,
        dataset=dataset_val,
        num_rounds=args.num_rounds,
        limit_questions=limit_questions,
        eval_batch_size=args.batch_size
    )
    
    # ==========================
    # 6. 结果输出
    # ==========================
    print("\n" + "=" * 50)
    print("Evaluation Complete")
    print("=" * 50)
    print(f"  Final Score: {score:.4f}" if isinstance(score, float) else f"  Final Score: {score}")
    print(f"  Questions: {limit_questions}")
    print(f"  Mode: {args.mode}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())