import math
import time
import asyncio
import copy
from typing import Optional, Iterator, List, Any
from tqdm import tqdm

from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens

async def evaluate(
        graph: Graph,
        dataset,
        num_rounds: int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        ) -> float:

    print(f"Evaluating gdesigner on {dataset.__class__.__name__} split {dataset.split}")
    
    # [Evolution Placeholder] 如果后续使用 GCN/MLP 辅助演化，在此处开启 eval 模式
    # if hasattr(graph, 'gcn') and graph.gcn: graph.gcn.eval()
    
    accuracy = Accuracy()

    # --- 数据加载器 (保持不变) ---
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions: break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records: yield records

    data_len = min(len(dataset), limit_questions) if limit_questions else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    # --- 主循环 ---
    for i_batch, record_batch in tqdm(enumerate(eval_loader(eval_batch_size)), total=num_batches):
        print(80*'-')
        start_ts = time.time()
        tasks = [] 

        for record in record_batch:
            # 1. 深度拷贝：隔离每道题的 Agent 记忆
            realized_graph = copy.deepcopy(graph)
            
            # [Evolution Placeholder] 如果使用 Neural-Guided 演化，需在此处共享模型权重
            # if hasattr(graph, 'gcn'): realized_graph.gcn = graph.gcn
            # if hasattr(graph, 'mlp'): realized_graph.mlp = graph.mlp

            # 2. 准备输入并创建任务
            input_dict = dataset.record_to_input(record)
            tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
        
        # 3. 并行推理
        raw_results = await asyncio.gather(*tasks)
        raw_answers, log_probs = zip(*raw_results)
        
        print(f"Batch time {time.time() - start_ts:.3f}s")
        
        # 4. 统计结果
        for raw_answer, record in zip(raw_answers, record_batch):
            answer = dataset.postprocess_answer(raw_answer)
            correct_answer = dataset.record_to_target_answer(record)
            
            print(f"Raw: {raw_answer} | Post: {answer} | Correct: {correct_answer}")
            accuracy.update(answer, correct_answer)
            accuracy.print()
        
        # 5. Token 消耗监控
        print(f"Cost: ${Cost.instance().value:.4f} | "
              f"In: {PromptTokens.instance().value} | "
              f"Out: {CompletionTokens.instance().value}")

    print("Done!")
    return accuracy.get()