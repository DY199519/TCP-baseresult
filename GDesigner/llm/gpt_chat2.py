"""
基于 OpenAI SDK 的 LLM 实现
使用标准的 OpenAI API 格式，支持从 .env 读取配置
"""
import os
from typing import List, Union, Optional, Dict
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI, AsyncOpenAI

from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry

# 加载环境变量
load_dotenv()
BASE_URL = os.getenv('BASE_URL', '')
API_KEY = os.getenv('API_KEY', '')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat_openai(
    model: str,
    messages: List[Dict[str, str]],
    base_url: str = None,
    api_key: str = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: int = 60,
) -> str:
    """
    使用 OpenAI SDK 异步调用 API
    
    Args:
        model: 模型名称
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        base_url: API 基础 URL
        api_key: API 密钥
        max_tokens: 最大 token 数
        temperature: 温度参数
        timeout: 超时时间（秒）
    
    Returns:
        API 返回的文本内容
    """
    base_url = base_url or BASE_URL
    api_key = api_key or API_KEY
    
    if not base_url or not api_key:
        raise ValueError("BASE_URL 和 API_KEY 必须在 .env 文件中配置")
    
    # 创建异步客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    
    try:
        # 调用 API
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        # 计算成本（使用 prompt 和 response）
        prompt_text = "".join([msg.get('content', '') for msg in messages])
        cost_count(prompt_text, content, model)
        
        return content.strip()
    finally:
        await client.close()


def chat_openai(
    model: str,
    messages: List[Dict[str, str]],
    base_url: str = None,
    api_key: str = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: int = 60,
) -> str:
    """
    使用 OpenAI SDK 同步调用 API
    
    Args:
        model: 模型名称
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        base_url: API 基础 URL
        api_key: API 密钥
        max_tokens: 最大 token 数
        temperature: 温度参数
        timeout: 超时时间（秒）
    
    Returns:
        API 返回的文本内容
    """
    base_url = base_url or BASE_URL
    api_key = api_key or API_KEY
    
    if not base_url or not api_key:
        raise ValueError("BASE_URL 和 API_KEY 必须在 .env 文件中配置")
    
    # 创建同步客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    
    try:
        # 调用 API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # 提取响应内容
        content = response.choices[0].message.content
        
        # 计算成本（使用 prompt 和 response）
        prompt_text = "".join([msg.get('content', '') for msg in messages])
        cost_count(prompt_text, content, model)
        
        return content.strip()
    finally:
        client.close()


def convert_messages(messages: Union[List[Message], str]) -> List[Dict[str, str]]:
    """
    将 Message 对象列表转换为 OpenAI API 格式
    
    Args:
        messages: Message 对象列表或字符串
    
    Returns:
        OpenAI API 格式的消息列表
    """
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    
    if isinstance(messages, list) and len(messages) > 0:
        if isinstance(messages[0], Message):
            # 如果是 Message 对象列表
            return [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
        elif isinstance(messages[0], dict):
            # 如果已经是字典格式
            return messages
    
    # 默认情况
    return [{"role": "user", "content": str(messages)}]


@LLMRegistry.register('GPTChat2')
class GPTChat2(LLM):
    """
    基于 OpenAI SDK 的 LLM 实现
    
    使用方式：
    1. 在 .env 文件中配置：
       BASE_URL=https://your-api-url.com/v1
       API_KEY=your-api-key
    
    2. 使用：
       llm = LLMRegistry.get('GPTChat2', model_name='gpt-4o')
       response = await llm.agen([Message(role='user', content='Hello')])
    """
    
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        """
        初始化 GPTChat2
        
        Args:
            model_name: 模型名称，如 'gpt-4o', 'claude-sonnet-4-20250514' 等
            base_url: API 基础 URL（可选，默认从 .env 读取）
            api_key: API 密钥（可选，默认从 .env 读取）
        """
        self.model_name = model_name
        self.base_url = base_url or BASE_URL
        self.api_key = api_key or API_KEY
        
        if not self.base_url or not self.api_key:
            raise ValueError(
                "BASE_URL 和 API_KEY 必须在 .env 文件中配置，"
                "或者在初始化时传入 base_url 和 api_key 参数"
            )
    
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        异步生成回复
        
        Args:
            messages: Message 对象列表
            max_tokens: 最大 token 数（默认使用 DEFAULT_MAX_TOKENS）
            temperature: 温度参数（默认使用 DEFAULT_TEMPERATURE）
            num_comps: 生成数量（当前实现只支持 1）
        
        Returns:
            生成的文本内容（字符串）
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        
        # 转换消息格式
        openai_messages = convert_messages(messages)
        
        # 调用异步 API
        response = await achat_openai(
            model=self.model_name,
            messages=openai_messages,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # 如果 num_comps > 1，返回列表（当前实现只返回单个结果）
        if num_comps and num_comps > 1:
            # 注意：当前实现不支持批量生成，返回单个结果的列表
            return [response]
        
        return response
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        同步生成回复
        
        Args:
            messages: Message 对象列表
            max_tokens: 最大 token 数（默认使用 DEFAULT_MAX_TOKENS）
            temperature: 温度参数（默认使用 DEFAULT_TEMPERATURE）
            num_comps: 生成数量（当前实现只支持 1）
        
        Returns:
            生成的文本内容（字符串）
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        
        # 转换消息格式
        openai_messages = convert_messages(messages)
        
        # 调用同步 API
        response = chat_openai(
            model=self.model_name,
            messages=openai_messages,
            base_url=self.base_url,
            api_key=self.api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # 如果 num_comps > 1，返回列表（当前实现只返回单个结果）
        if num_comps and num_comps > 1:
            # 注意：当前实现不支持批量生成，返回单个结果的列表
            return [response]
        
        return response

