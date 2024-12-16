import os
import sys
import openai
import yaml
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import requests
import tqdm
import random

class ReflectiveTranslator:
    # 模型配置
    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "max_tokens": 16384,  # 16K 上下文窗口
            "cost_per_1k_tokens": 0.01,  # 示例成本
            "recommended_chunk_size": 8000,  # 建议的分块大小
            "max_output_tokens": 8000,  # 最大输出token数
        }
    }

    # 语言代码映射
    LANGUAGE_CODES = {
        "中文": "zh",
        "英文": "en",
        "日文": "ja",
        "韩文": "ko",
        "法文": "fr",
        "德文": "de",
        "西班牙文": "es",
        "俄文": "ru",
        "阿拉伯文": "ar",
        "葡萄牙文": "pt",
        "意大利文": "it",
        "越南文": "vi",
        "泰文": "th"
    }

    # 支持的文件格式
    SUPPORTED_FORMATS = {
        '.pdf': 'PDF文档',
        '.md': 'Markdown文档',
        '.txt': '文本文档',
        '.rst': 'reStructuredText文档'
    }

    # 翻译提示词
    TRANSLATION_PROMPTS = {
        'initial_translation': """# Role: 资深跨语言翻译专家

## Profile:
- Author: 玄清
- 精通语言学、文学和跨学科翻译
- 深谙文化传播，具备文学创造能力
- 传承并超越"信、达、雅"的翻译理念

## Expertise:
- 专业领域文献翻译
- 学术论文本地化
- 技术文档转换
- 多语言文化适配

## Goals:
1. 准确识别文本类型和专业领域
2. 保持文档结构和格式完整
3. 准确传达原意与情感
4. 确保术语一致性与专业性
5. 注重文化适应性与等效性
6. 保证译文流畅性与自然度
7. 还原原文风格与语气
8. 规范处理引用和参考文献
9. 保留专有名词和术语原文

## Guidelines:
1. 遵循目标语言的表达习惯
2. 采用统一的术语表进行翻译
3. 保持段落和章节结构
4. 规范化处理图表和公式
5. 确保引用格式的一致性""",

        'reflective_review': """# Role: 翻译评审专家

## Profile:
- Author: 玄清
- Description: 专注于运用"信达雅"翻译理论，对翻译文本进行全面评审和优化。

## Evaluation Criteria:
1. 忠实度（信）：
   - 内容完整性
   - 术语准确性
   - 专业性保持
   - 引用规范性

2. 通达性（达）：
   - 语言流畅度
   - 表达自然度
   - 结构清晰度
   - 逻辑连贯性

3. 雅致度（雅）：
   - 语言优美性
   - 风格一致性
   - 文体适当性
   - 格式规范性

## Quality Standards:
1. 术语一致性检查：
   - 核对专业术语表
   - 验证术语翻译准确性
   - 确保术语使用统一

2. 格式规范性检查：
   - 段落结构完整
   - 标点符号规范
   - 排版格式统一
   - 引用格式标准

3. 文体特征评估：
   - 学术性文献规范
   - 技术文档标准
   - 文学作品风格
   - 通用文本特征""",

        'final_revision': """# Role: 资深翻译优化专家

## Profile:
- Author: 玄清
- Description: 专业翻译改进专家，致力于按照翻译优化建议逐条修改，将翻译初稿提升至最高质量水平。

## Objectives:
1. 全面落实评审建议
2. 确保全局一致性
3. 提升翻译质量
4. 完善文档规范

## Quality Metrics:
1. 准确性指标：
   - 术语使用准确度 ≥ 98%
   - 专业内容还原度 ≥ 95%
   - 格式规范遵循度 = 100%

2. 流畅性指标：
   - 语言自然度评分 ≥ 90%
   - 读者理解难度适中
   - 表达连贯性良好

3. 专业性指标：
   - 术语表使用符合度 = 100%
   - 专业规范遵循度 ≥ 95%
   - 引用格式规范度 = 100%""",

        'markdown_format': """# Role: 文档Markdown整理

## Profile:
- Author: 玄清
- Description: 专注于将各种文本内容高效转换为标准化的Markdown格式，优化文档结构和可读性。

## Goals:
1. 输出结构清晰、易读的Markdown文档
2. 删除冗余信息（如页眉、页脚、页码）
3. 保证文档逻辑连贯，确保无信息丢失
4. 使用标准Markdown语法，实现文档的交互性与易编辑性

## Skills:
1. 精通文本预处理，自动识别并保留原文语言，精准去除无关字符
2. 熟悉文本结构解析，能够智能准确转换标题、段落、列表等元素
3. 掌握Markdown语法，能高效转换文档元素为规范化Markdown格式
4. 熟练处理截断文本，确保文档逻辑完整且无信息遗漏
5. 严格执行质量控制，确保文档无语义偏差、格式准确

## Constrains:
1. 严格保留原文关键信息与结构，避免信息丢失
2. 确保处理截断文本时逻辑连贯，保留所有重要信息
3. 输出文档应段落间距合理，层次清晰，具有良好可读性
4. 不进行语言翻译或内容修改，严格遵守原文语言
5. 避免误删有意义的内容或结构信息，确保文档准确无误
6. 避免格式错乱或信息变形，保证格式规范
7. 文本转换时，段落格式需忠实于原文，不能随意增加标题层次"""
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens_per_chunk: int = 16000,  
        overlap_tokens: int = 100,
        cost_limit: float = 10.0,
        source_lang: str = "中文",
        target_lang: str = "英文"
    ):
        """初始化翻译器
        
        Args:
            api_key (str): OpenAI API密钥
            model_name (str, optional): 模型名称. Defaults to "gpt-4o-mini".
            temperature (float, optional): 温度参数. Defaults to 0.7.
            max_tokens_per_chunk (int, optional): 每个块的最大token数. Defaults to 16000.
            overlap_tokens (int, optional): 重叠token数. Defaults to 100.
            cost_limit (float, optional): 成本限制. Defaults to 10.0.
            source_lang (str, optional): 源语言. Defaults to "中文".
            target_lang (str, optional): 目标语言. Defaults to "英文".
        """
        # 验证模型支持
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {', '.join(self.MODEL_CONFIGS.keys())}")
        
        # 验证语言支持
        if source_lang not in self.LANGUAGE_CODES:
            raise ValueError(f"不支持的源语言: {source_lang}。支持的语言: {', '.join(self.LANGUAGE_CODES.keys())}")
        if target_lang not in self.LANGUAGE_CODES:
            raise ValueError(f"不支持的目标语言: {target_lang}。支持的语言: {', '.join(self.LANGUAGE_CODES.keys())}")
        
        # 获取模型配置
        self.model_config = self.MODEL_CONFIGS[model_name]
        
        # 设置分块大小
        if max_tokens_per_chunk is None:
            max_tokens_per_chunk = self.model_config['recommended_chunk_size']
        elif max_tokens_per_chunk > self.model_config['max_tokens']:
            raise ValueError(f"分块大小超过模型最大上下文长度: {max_tokens_per_chunk} > {self.model_config['max_tokens']}")
        
        # 加载环境变量
        load_dotenv()
        
        # 设置API配置
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("未提供API密钥，请通过参数传入或在环境变量中设置OPENAI_API_KEY")
            
        # 配置API
        openai.api_key = self.api_key
        openai.api_base = "https://api.gptsapi.net/v1"
        openai.api_version = None
        
        # 保存配置
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.cost_limit = cost_limit
        self.total_cost = 0
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 加载术语表
        self.glossary = self._load_glossary()
        
        # 创建进度保存目录
        self.progress_dir = Path("translation_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        # 初始化日志
        self._setup_logging()

        # 创建图片保存目录
        self.images_dir = "images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def _setup_logging(self):
        """设置日志配置"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器，防止重复
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler('translation.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # 防止日志传播到根记录器
        self.logger.propagate = False

    def _load_glossary(self) -> Dict:
        """加载术语表"""
        try:
            with open('glossary.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning("未找到术语表文件，将使用空术语表")
            return {}

    def _handle_rate_limit(self, error_message: str) -> int:
        """处理速率限制错误，返回等待时间

        Args:
            error_message (str): 错误信息

        Returns:
            int: 建议的等待时间（秒）
        """
        # 解析错误信息中的等待时间
        wait_time = 3600  # 默认等待1小时
        try:
            # 尝试从错误信息中提取等待时间
            if 'try again in' in error_message.lower():
                time_str = error_message.lower().split('try again in')[1].strip()
                if 'hour' in time_str:
                    hours = int(time_str.split()[0])
                    wait_time = hours * 3600
                elif 'minute' in time_str:
                    minutes = int(time_str.split()[0])
                    wait_time = minutes * 60
                elif 'second' in time_str:
                    seconds = int(time_str.split()[0])
                    wait_time = seconds
        except:
            pass
        
        return wait_time

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=lambda x: isinstance(x, (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError))
    )
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """调用 GPTsAPI，包含重试逻辑
        
        Args:
            system_prompt (str): 系统提示
            user_prompt (str): 用户提示
            
        Returns:
            str: API响应的文本内容
            
        Raises:
            Exception: API调用失败时抛出异常
        """
        max_retries = 5
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"正在发送API请求 (尝试 {attempt + 1}/{max_retries})...")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # 设置SSL验证选项
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # 配置请求
                import urllib3
                urllib3.disable_warnings()
                openai.verify_ssl_certs = False
                
                # 使用openai库调用API
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.model_config['max_output_tokens'],  # 设置最大输出token数
                    request_timeout=30,  # 设置30秒超时
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Connection": "keep-alive"
                    }
                )
                
                self.logger.info("API请求成功，正在处理响应...")
                
                # 验证响应格式
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("无效的API响应格式")
                
                if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
                    raise ValueError("响应中没有有效内容")
                
                result = response.choices[0].message['content'].strip()
                
                # 验证响应内容
                if not result:
                    raise ValueError("响应内容为空")
                
                self.logger.info(f"成功获取响应 (长度: {len(result)} 字符)")
                return result
                
            except Exception as e:
                self.logger.error(f"API请求错误: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.error(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    self.logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"API调用失败，已达到最大重试次数: {str(e)}")

    def _initial_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """初始翻译，保留专有名词
        
        Args:
            text (str): 要翻译的文本
            source_lang (str): 源语言
            target_lang (str): 目标语言
            
        Returns:
            str: 翻译结果
        """
        system_prompt = self.TRANSLATION_PROMPTS['initial_translation']
        user_prompt = f"""请将以下{source_lang}文本翻译成{target_lang}。
注意事项：
1. 保持原文的格式和结构
2. 保留专有名词的原文
3. 确保术语翻译的一致性
4. 保持学术/技术文献的专业性

原文：
{text}"""
        
        return self._call_openai(system_prompt, user_prompt)

    def _reflective_review(self, original_text: str, translation: str, source_lang: str, target_lang: str) -> str:
        """反思评审阶段"""
        system_prompt = self.TRANSLATION_PROMPTS['reflective_review']
        user_prompt = f"""请评审以下{source_lang}文本的{target_lang}翻译。
关注以下方面：
1. 翻译的准确性和完整性
2. 专业术语的使用
3. 语言的流畅度和自然度
4. 格式和结构的保持

原文：
{original_text}

译文：
{translation}"""

        return self._call_openai(system_prompt, user_prompt)

    def _final_revision(self, original_text: str, translation: str, review_feedback: str, source_lang: str, target_lang: str) -> str:
        """终稿修改阶段"""
        system_prompt = self.TRANSLATION_PROMPTS['final_revision']
        user_prompt = f"""请根据评审意见修改{target_lang}译文。
评审意见：
{review_feedback}

原文：
{original_text}

当前译文：
{translation}"""

        return self._call_openai(system_prompt, user_prompt)

    def _format_markdown(self, text: str, layout_info: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> str:
        """使用 GPT 优化 Markdown 格式"""
        system_prompt = self.TRANSLATION_PROMPTS['markdown_format']
        user_prompt = f"""请将以下文本转换为规范的Markdown格式，保持原有的结构和格式特征：

文本内容：
{text}

布局信息：
{json.dumps(layout_info, ensure_ascii=False, indent=2)}

图片信息：
{json.dumps(images, ensure_ascii=False, indent=2)}"""

        return self._call_openai(system_prompt, user_prompt)

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """执行多阶段翻译过程

        Args:
            text (str): 要翻译的文本
            source_lang (str): 源语言
            target_lang (str): 目标语言

        Returns:
            str: 翻译结果
        """
        # 初始翻译
        initial_translation = self._initial_translation(text, source_lang, target_lang)
        
        # 反思评审
        review_feedback = self._reflective_review(text, initial_translation, source_lang, target_lang)
        
        # 最终修改
        final_translation = self._final_revision(text, initial_translation, review_feedback, source_lang, target_lang)
        
        return final_translation

    def _estimate_cost(self, text: str) -> float:
        """估算翻译成本
        
        Args:
            text (str): 要翻译的文本
            
        Returns:
            float: 预估成本（美元）
        """
        try:
            # 使用tiktoken估算token数量
            encoder = tiktoken.encoding_for_model(self.model_name)
            token_count = len(encoder.encode(text))
        except Exception as e:
            logging.warning(f"无法使用tiktoken计算token: {str(e)}")
            # 使用简单的估算方法作为后备
            token_count = len(text) if any('\u4e00' <= c <= '\u9fff' for c in text) else len(text) // 4
        
        # 考虑到翻译和审查的多次API调用
        total_tokens = token_count * 3  # 初始翻译、反思评审、最终修订
        
        # 计算成本
        cost = (total_tokens / 1000) * self.model_config['cost_per_1k_tokens']
        
        return cost

    def _split_text(self, text: str) -> List[str]:
        """将文本分割成适合模型处理的片段，使用更智能的语义分割策略
        
        Args:
            text (str): 要分割的文本
            
        Returns:
            List[str]: 文本片段列表
        """
        if not text:
            return []
            
        # 首先保护特殊内容
        processed_text, placeholders = self._preserve_special_content(text)
        
        # 获取最优块大小
        optimal_chunk_size = self._get_optimal_chunk_size(processed_text)
        
        # 分割文本
        chunks = []
        current_chunk = []
        current_size = 0
        
        paragraphs = re.split(r'\n\s*\n', processed_text)
        
        for para in paragraphs:
            para_size = self._estimate_tokens(para)
            
            if para_size > optimal_chunk_size:
                # 如果段落太长，按句子分割
                sentences = re.split(r'([。！？.!?])', para)
                current_sentence = ''
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]  # 添加分隔符
                        
                    sentence_size = self._estimate_tokens(sentence)
                    
                    if current_size + sentence_size > optimal_chunk_size and current_chunk:
                        # 保存当前块
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
            else:
                # 如果当前段落加入会超过大小限制，保存当前块
                if current_size + para_size > optimal_chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(para)
                current_size += para_size
        
        # 保存最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # 还原每个块中的特殊内容
        restored_chunks = [self._restore_special_content(chunk, placeholders) for chunk in chunks]
        
        return restored_chunks

    def _merge_translations(self, translations: List[str], overlap_tokens: int) -> str:
        """合并翻译片段，使用更智能的合并策略
        
        Args:
            translations (List[str]): 翻译片段列表
            overlap_tokens (int): 片段间重叠的token数
            
        Returns:
            str: 合并后的翻译文本
        """
        if not translations:
            return ""
        
        # 使用第一个片段作为基础
        merged = translations[0]
        
        # 遍历后续片段
        for i in range(1, len(translations)):
            current = translations[i]
            
            # 寻找最长的重叠部分
            overlap = self._find_longest_overlap(merged, current)
            
            if overlap:
                # 使用重叠部分合并文本，并确保在合适的位置合并
                overlap_start = merged.rfind(overlap)
                if overlap_start != -1:
                    # 检查重叠部分的上下文，确保合并点合适
                    context_before = merged[max(0, overlap_start-50):overlap_start]
                    context_after = current[len(overlap):min(len(current), len(overlap)+50)]
                    
                    # 如果上下文显示这是一个好的合并点
                    if self._is_good_merge_point(context_before, context_after):
                        merged = merged[:overlap_start] + current
                    else:
                        # 尝试在句子边界合并
                        last_sentence_end = self._find_last_sentence_end(merged)
                        next_sentence_start = self._find_next_sentence_start(current)
                        if last_sentence_end > 0 and next_sentence_start > 0:
                            merged = merged[:last_sentence_end] + '\n' + current[next_sentence_start:]
                        else:
                            merged += '\n' + current
                else:
                    merged += '\n' + current
            else:
                # 如果没有找到重叠，尝试在句子边界合并
                last_sentence_end = self._find_last_sentence_end(merged)
                next_sentence_start = self._find_next_sentence_start(current)
                
                if last_sentence_end > 0 and next_sentence_start > 0:
                    merged = merged[:last_sentence_end] + '\n' + current[next_sentence_start:]
                else:
                    merged += '\n' + current
        
        return merged

    def _is_good_merge_point(self, context_before: str, context_after: str) -> bool:
        """判断是否是好的合并点
        
        Args:
            context_before (str): 合并点前的上下文
            context_after (str): 合并点后的上下文
            
        Returns:
            bool: 是否是好的合并点
        """
        # 定义不应该分开的模式
        bad_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # 人名
            r'\b[A-Z]+\b',  # 缩写词
            r'\b\d+\.\d+\b',  # 小数
            r'\([^)]*$',  # 未闭合的括号
            r'"[^"]*$',  # 未闭合的引号
            r'\[[^\]]*$',  # 未闭合的方括号
        ]
        
        # 检查是否有不应该分开的模式
        for pattern in bad_patterns:
            if re.search(pattern, context_before) or re.search(f'^[^{pattern[-1]}]*{pattern[-1]}', context_after):
                return False
        
        return True

    def _find_longest_overlap(self, text1: str, text2: str, min_length: int = 10) -> str:
        """找到两个文本中最长的重叠部分
        
        Args:
            text1 (str): 第一个文本
            text2 (str): 第二个文本
            min_length (int): 最小重叠长度
            
        Returns:
            str: 最长重叠部分
        """
        # 获取text1的后半部分和text2的前半部分
        text1_end = text1[-200:]  # 只检查最后200个字符
        text2_start = text2[:200]  # 只检查前200个字符
        
        longest_overlap = ""
        
        # 从最长可能的重叠开始检查
        for length in range(min(len(text1_end), len(text2_start)), min_length - 1, -1):
            for i in range(len(text1_end) - length + 1):
                substring = text1_end[i:i + length]
                if substring in text2_start:
                    return substring
        
        return longest_overlap

    def _find_last_sentence_end(self, text: str) -> int:
        """找到文本中最后一个句子的结束位置
        
        Args:
            text (str): 要检查的文本
            
        Returns:
            int: 最后一个句子的结束位置，如果没有找到返回-1
        """
        # 定义句子结束标记
        sentence_ends = ['. ', '。', '！', '? ', '？', '…']
        
        # 从后向前查找第一个句子结束标记
        for i in range(len(text) - 1, -1, -1):
            for end in sentence_ends:
                if text[i:i + len(end)] == end:
                    return i + len(end)
        
        return -1

    def _find_next_sentence_start(self, text: str) -> int:
        """找到文本中下一个句子的开始位置
        
        Args:
            text (str): 要检查的文本
            
        Returns:
            int: 下一个句子的开始位置，如果没有找到返回-1
        """
        # 跳过开头的空白字符
        i = 0
        while i < len(text) and text[i].isspace():
            i += 1
        
        return i if i < len(text) else -1

    def _extract_text(self, file_path: str) -> List[Dict]:
        """从文件中提取文本，保持原格式
        
        Args:
            file_path (str): 文件路径
            
        Returns:
            List[Dict]: 包含文本内容和格式信息的列表
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if file_ext == '.md':
                    return self._extract_markdown_format(content)
                else:  # .txt 或其他文本文件
                    return self._extract_text_format(content)
                    
            except Exception as e:
                logging.error(f"读取文件失败: {str(e)}")
                return []

    def _extract_markdown_format(self, content: str) -> List[Dict]:
        """从Markdown文件中提取文本和格式信息
        
        Args:
            content (str): Markdown文本内容
            
        Returns:
            List[Dict]: 包含文本内容和格式信息的列表
        """
        blocks = []
        current_block = {'type': 'text', 'content': '', 'format': {}}
        
        lines = content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 处理标题
            if line.startswith('#'):
                if current_block['content']:
                    blocks.append(current_block)
                
                level = len(line.split()[0])  # 计算#的数量
                current_block = {
                    'type': 'text',
                    'content': line,
                    'format': {
                        'is_title': True,
                        'level': level,
                        'original_format': 'markdown'
                    }
                }
                blocks.append(current_block)
                current_block = {'type': 'text', 'content': '', 'format': {}}
            
            # 处理代码块
            elif line.startswith('```'):
                if current_block['content']:
                    blocks.append(current_block)
                
                # 收集整个代码块
                code_block = [line]
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    code_block.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_block.append(lines[i])
                
                blocks.append({
                    'type': 'text',
                    'content': '\n'.join(code_block),
                    'format': {
                        'is_code': True,
                        'original_format': 'markdown'
                    }
                })
            
            # 处理列表
            elif line.strip().startswith(('- ', '* ', '+ ', '1. ')):
                if current_block['content'] and not current_block['format'].get('is_list'):
                    blocks.append(current_block)
                    current_block = {'type': 'text', 'content': '', 'format': {}}
                
                current_block['format']['is_list'] = True
                current_block['format']['original_format'] = 'markdown'
                current_block['content'] += line + '\n'
            
            # 处理普通段落
            else:
                if current_block['format'].get('is_list') and not line.strip().startswith(('- ', '* ', '+ ', '1. ')):
                    blocks.append(current_block)
                    current_block = {'type': 'text', 'content': '', 'format': {}}
                
                current_block['content'] += line + '\n'
            
            i += 1
        
        # 添加最后一个块
        if current_block['content']:
            blocks.append(current_block)
        
        return blocks

    def _extract_text_format(self, content: str) -> List[Dict]:
        """从纯文本文件中提取文本和格式信息
        
        Args:
            content (str): 文本内容
            
        Returns:
            List[Dict]: 包含文本内容和格式信息的列表
        """
        blocks = []
        current_block = {'type': 'text', 'content': '', 'format': {}}
        
        lines = content.split('\n')
        for line in lines:
            stripped_line = line.strip()
            
            # 检测可能的标题（全大写或数字开头的短行）
            if stripped_line.isupper() and len(stripped_line) < 100 and not stripped_line.startswith('#'):
                if current_block['content']:
                    blocks.append(current_block)
                
                blocks.append({
                    'type': 'text',
                    'content': line,
                    'format': {
                        'is_title': True,
                        'original_format': 'text'
                    }
                })
                current_block = {'type': 'text', 'content': '', 'format': {}}
            
            # 检测列表（使用常见的列表标记）
            elif stripped_line.startswith(('1.', '2.', '3.')):
                if current_block['content'] and not current_block['format'].get('is_list'):
                    blocks.append(current_block)
                    current_block = {'type': 'text', 'content': '', 'format': {'is_list': True, 'original_format': 'text'}}
                
                current_block['content'] += line + '\n'
            
            # 处理普通段落
            else:
                if current_block['format'].get('is_list') and not stripped_line.startswith(('1.', '2.', '3.')):
                    blocks.append(current_block)
                    current_block = {'type': 'text', 'content': '', 'format': {'original_format': 'text'}}
                
                # 保持原始缩进
                current_block['content'] += line + '\n'
        
        # 添加最后一个块
        if current_block['content']:
            blocks.append(current_block)
        
        return blocks

    def _extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """从PDF文件中提取文本和图片
        
        Args:
            file_path (str): PDF文件路径
            
        Returns:
            List[Dict]: 包含文本内容和格式信息的列表
        """
        content_blocks = []
        
        try:
            # 创建图片保存目录
            images_dir = os.path.join(os.path.dirname(file_path), 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # 打开PDF文件
            doc = fitz.open(file_path)
            
            # 遍历每一页
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本
                text_blocks = page.get_text("dict", sort=True)["blocks"]
                current_block_text = []
                current_block_rect = None
                
                for b in text_blocks:
                    if "lines" in b:
                        block_text = []
                        for line in b["lines"]:
                            line_text = []
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text.append(text)
                            if line_text:
                                block_text.append(" ".join(line_text))
                        if block_text:
                            text_content = " ".join(block_text)
                            content_blocks.append({
                                "type": "text",
                                "content": text_content,
                                "rect": b["bbox"],
                                "page": page_num + 1
                            })
                
                # 提取图片
                try:
                    for img_index, img in enumerate(page.get_images(full=True)):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            if base_image:
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                
                                # 保存图片
                                image_filename = f"image_p{page_num + 1}_{img_index + 1}.{image_ext}"
                                image_path = os.path.join(images_dir, image_filename)
                                
                                with open(image_path, "wb") as f:
                                    f.write(image_bytes)
                                
                                # 获取图片在页面中的位置
                                pix = fitz.Pixmap(doc, xref)
                                bbox = page.get_image_bbox(img)
                                if bbox:
                                    content_blocks.append({
                                        'type': 'image',
                                        'path': image_path,
                                        'page': page_num + 1,
                                        'width': pix.width,
                                        'height': pix.height,
                                        'relative_position': {
                                            'x1': bbox.x0 / page.rect.width,
                                            'y1': bbox.y0 / page.rect.height,
                                            'x2': bbox.x1 / page.rect.width,
                                            'y2': bbox.y1 / page.rect.height
                                        }
                                    })
                                pix = None  # 释放内存
                        except Exception as e:
                            logging.warning(f"处理图片时出错: {str(e)}")
                            continue
                except Exception as e:
                    logging.warning(f"获取页面图片列表时出错: {str(e)}")
                
                logging.info(f"已处理第 {page_num + 1}/{len(doc)} 页")
            
            doc.close()
            return content_blocks
            
        except Exception as e:
            logging.error(f"处理PDF文件时出错: {str(e)}")
            return []

    def _get_heading_level(self, font_size: float) -> Optional[int]:
        """根据字体大小确定标题级别
        
        Args:
            font_size (float): 字体大小
            
        Returns:
            Optional[int]: 标题级别，如果不是标题则返回 None
        """
        if font_size > 14:
            return 1
        elif font_size > 12:
            return 2
        elif font_size > 10:
            return 3
        return None

    def _get_alignment(self, bbox: List[float], page_width: float) -> str:
        """根据文本框位置判断对齐方式
        
        Args:
            bbox (List[float]): 文本框坐标
            page_width (float): 页面宽度
            
        Returns:
            str: 对齐方式 ('left', 'center', 'right')
        """
        x_center = (bbox[0] + bbox[2]) / 2
        relative_pos = x_center / page_width
        
        if relative_pos < 0.35:
            return 'left'
        elif relative_pos > 0.65:
            return 'right'
        else:
            return 'center'

    def _convert_to_markdown(self, text: str, layout_info: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> str:
        """将文本转换为Markdown格式，严格保持原PDF排版
        
        Args:
            text (str): 翻译后的文本
            layout_info (List[Dict[str, Any]]): 排版信息
            images (List[Dict[str, Any]]: 图片信息
            
        Returns:
            str: Markdown格式的文本
        """
        formatted_text = []
        current_page = None
        
        for page in layout_info:
            page_num = page['page']
            
            # 添加分页符（除第一页外）
            if current_page is not None:
                formatted_text.append('\n---\n')
            current_page = page_num
            
            # 按垂直位置排序页面元素
            elements = sorted(page['elements'], key=lambda x: (x['bbox'][1], x['bbox'][0]))
            
            # 跟踪当前垂直位置和标题层级
            current_y = 0
            img_index = 0
            last_heading_level = 0
            
            for element in elements:
                # 检查是否需要插入图片
                if page_num in image_positions:
                    while (img_index < len(image_positions[page_num]) and 
                           image_positions[page_num][img_index]['relative_position']['y1'] <= element['bbox'][1]):
                        img = image_positions[page_num][img_index]
                        img_path = os.path.relpath(img['path']).replace('\\', '/')
                        
                        # 计算图片宽度，保持合适的显示比例
                        max_width = 800
                        if 'width' in img and 'height' in img:
                            aspect_ratio = img['width'] / img['height']
                            img_width = min(max_width, img['width'])
                            img_height = int(img_width / aspect_ratio)
                            formatted_text.append(f"\n<div align='center'>\n<img src='{img_path}' width='{img_width}' height='{img_height}'>\n</div>\n")
                        else:
                            formatted_text.append(f"\n<div align='center'>\n<img src='{img_path}' width='{max_width}'>\n</div>\n")
                        
                        img_index += 1
                
                if element['type'] == 'text':
                    text_content = element['content'].strip()
                    if not text_content:
                        continue
                    
                    # 处理标题
                    font_size = element.get('size', 0)
                    if font_size > 12:  # 标题判断
                        heading_level = min(6, max(1, int(18 - font_size) // 2))
                        
                        # 确保标题层级合理
                        if last_heading_level > 0 and heading_level - last_heading_level > 1:
                            heading_level = last_heading_level + 1
                        
                        text_content = f"{'#' * heading_level} {text_content}"
                        last_heading_level = heading_level
                    else:
                        # 应用文本样式
                        if element.get('style', {}).get('bold'):
                            text_content = f'**{text_content}**'
                        if element.get('style', {}).get('italic'):
                            text_content = f'*{text_content}*'
                        if element.get('style', {}).get('underline'):
                            text_content = f'<u>{text_content}</u>'
                    
                    # 处理对齐方式
                    alignment = element.get('alignment', 'left')
                    if alignment == 'center':
                        text_content = f"<div align='center'>\n\n{text_content}\n\n</div>"
                    elif alignment == 'right':
                        text_content = f"<div align='right'>\n\n{text_content}\n\n</div>"
                    
                    # 添加适当的空行
                    if formatted_text and not formatted_text[-1].endswith('\n\n'):
                        formatted_text.append('\n\n')
                    
                    formatted_text.append(text_content)
                    current_y = element['bbox'][3]
            
            # 检查是否还有剩余的图片需要插入
            if page_num in image_positions:
                while img_index < len(image_positions[page_num]):
                    img = image_positions[page_num][img_index]
                    img_path = os.path.relpath(img['path']).replace('\\', '/')
                    if 'width' in img and 'height' in img:
                        aspect_ratio = img['width'] / img['height']
                        img_width = min(800, img['width'])
                        img_height = int(img_width / aspect_ratio)
                        formatted_text.append(f"\n<div align='center'>\n<img src='{img_path}' width='{img_width}' height='{img_height}'>\n</div>\n")
                    else:
                        formatted_text.append(f"\n<div align='center'>\n<img src='{img_path}' width='800'>\n</div>\n")
                    img_index += 1
        
        return '\n'.join(formatted_text)

    def _format_markdown(self, text: str, layout_info: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> str:
        """使用 GPT 优化 Markdown 格式

        Args:
            text (str): 要格式化的文本
            layout_info (List[Dict[str, Any]]): 排版信息
            images (List[Dict[str, Any]]: 图片信息

        Returns:
            str: 格式化后的 Markdown 文本
        """
        # 首先提取所有图片引用
        image_refs = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith('![') and line.endswith(')'):
                image_refs.append(line)

        # 进行基本的 Markdown 转换
        md_text = self._convert_to_markdown(text, layout_info, images)
        
        # 确保所有图片都被正确引用
        for img in images:
            img_path = os.path.relpath(img['path']).replace('\\', '/')
            img_width = min(800, img.get('width', 800))
            img_height = None
            
            # 计算图片宽度，保持合适的显示比例
            if 'width' in img and 'height' in img:
                aspect_ratio = img['width'] / img['height']
                display_width = min(img['width'], 800)
                display_height = int(display_width / aspect_ratio)
                
                # 添加图片标记
                img_ref = f"\n###### ![图片]({img_path})\n\n"
                if img_ref not in md_text:
                    # 根据图片在PDF中的位置插入图片引用
                    page_text = md_text.split('---')
                    if len(page_text) >= img['page']:
                        page_content = page_text[img['page']-1]
                        # 在页面内容的适当位置插入图片
                        lines = page_content.split('\n')
                        insert_pos = 0
                        for i, line in enumerate(lines):
                            if i < len(lines) - 1:
                                insert_pos = i
                                break
                        lines.insert(insert_pos, img_ref)
                        page_text[img['page']-1] = '\n'.join(lines)
                        md_text = '---'.join(page_text)
        
        # 使用 GPT 进行 Markdown 格式优化，同时保护图片引用
        system_prompt = self.TRANSLATION_PROMPTS['markdown_format']
        user_prompt = f"""请对以下Markdown文本进行格式优化，确保结构清晰、易读，并符合标准Markdown语法规范。

特别注意：
1. 保持所有图片引用的完整性和位置
2. 不要修改图片引用的格式和内容
3. 确保优化后的文本包含所有原有图片
4. 保持图片的相对位置不变

原文本：
{md_text}

要求：
1. 保持原有的标题层级和结构
2. 优化段落间距和排版
3. 保留所有图片引用，位置不变
4. 删除冗余的分隔符和空行
5. 确保列表格式统一规范
6. 不要改变图片引用的内容和格式
7. 保持图片与相关文本的上下文关系"""

        response = self._call_openai(system_prompt, user_prompt)
        
        # 验证所有图片引用是否都在优化后的文本中
        optimized_text = response.strip()
        for img in images:
            img_path = os.path.relpath(img['path']).replace('\\', '/')
            img_ref = f"![{img_filename}](images/{img_filename})"
            if img_ref not in optimized_text:
                logging.warning(f"图片引用在格式化过程中丢失: {img_ref}")
                # 将丢失的图片引用放回原位置
                page_text = optimized_text.split('---')
                if len(page_text) >= img['page']:
                    page_content = page_text[img['page']-1]
                    lines = page_content.split('\n')
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if i < len(lines) - 1:
                            insert_pos = i
                            break
                    lines.insert(insert_pos, f"\n###### ![图片]({img_path})\n")
                    page_text[img['page']-1] = '\n'.join(lines)
                    optimized_text = '---'.join(page_text)
        
        return optimized_text

    def _save_as_markdown(self, output_path: str, translations: List[Dict]) -> None:
        """将翻译结果保存为Markdown格式

        Args:
            output_path (str): 输出文件路径
            translations (List[Dict]): 翻译结果列表，每个字典包含类型、内容和格式信息
        """
        try:
            markdown_content = []
            current_list_type = None
            
            # 获取输出目录
            output_dir = os.path.dirname(output_path)
            
            for block in translations:
                if not block.get('content'):
                    continue
                
                # 清理文本内容
                if block.get('type') == 'text':
                    block['content'] = self._clean_text(block['content'])
                
                # 保持图片和表格块不变
                markdown_content.append(block['content'])
                
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            
            logging.info(f"成功保存翻译结果到: {output_path}")
            
        except Exception as e:
            logging.error(f"保存Markdown文件时出错: {str(e)}")
            raise

    def _process_table_content(self, table_text: str) -> str:
        """处理表格内容，保持格式的同时翻译单元格文本
        
        Args:
            table_text (str): 原始表格文本
            
        Returns:
            str: 处理后的表格文本
        """
        lines = table_text.split('\n')
        processed_lines = []
        
        # 识别表头分隔行
        header_separator_pattern = r'\|[\s\-:]+\|'
        
        for i, line in enumerate(lines):
            # 跳过空行
            if not line.strip():
                processed_lines.append(line)
                continue
                
            # 如果是表头分隔行，直接保留
            if re.match(header_separator_pattern, line.strip()):
                processed_lines.append(line)
                continue
            
            # 处理表格内容行
            cells = line.split('|')
            processed_cells = []
            
            for cell in cells:
                cell = cell.strip()
                if not cell:  # 跳过空单元格
                    processed_cells.append('')
                    continue
                    
                # 检查单元格是否包含代码
                if self._contains_code(cell):
                    # 如果包含代码，保持原样
                    processed_cells.append(cell)
                else:
                    # 翻译普通文本内容
                    translated_cell = self._translate_text(cell, self.source_lang, self.target_lang)
                    processed_cells.append(translated_cell)
            
            # 重建表格行，保持原有的格式
            processed_line = '|' + '|'.join(f' {cell} ' for cell in processed_cells) + '|'
            processed_lines.append(processed_line)
        
        return '\n'.join(processed_lines)
    
    def _contains_code(self, text: str) -> bool:
        """检查文本是否包含代码
        
        Args:
            text (str): 要检查的文本
            
        Returns:
            bool: 是否包含代码
        """
        # 代码相关的模式
        code_patterns = [
            r'`[^`]+`',          # 行内代码
            r'```[\s\S]*?```',   # 代码块
            r'~~~[\s\S]*?~~~',   # 另一种代码块标记
            r'<code>.*?</code>', # HTML代码标签
            r'\$.*?\$',          # 数学公式
            r'\\\(.*?\\\)',      # LaTeX行内公式
            r'\\\[.*?\\\]'       # LaTeX块级公式
        ]
        
        return any(re.search(pattern, text, re.DOTALL) for pattern in code_patterns)

    def _detect_content_type(self, text: str) -> str:
        """
        检测文本内容的类型
        返回: 'code', 'table', 'text'
        """
        # 检测代码块
        code_markers = ['```', '~~~', 'def ', 'class ', 'import ', 'from ']
        if any(marker in text for marker in code_markers):
            return 'code'
        
        # 检测表格
        if '|' in text and any(line.strip().startswith('|') for line in text.split('\n')):
            return 'table'
            
        return 'text'
        
    def _estimate_text_complexity(self, text: str) -> float:
        """
        估算文本的复杂度
        返回: 0.5-1.5之间的复杂度系数
        """
        complexity = 1.0
        
        # 1. 句子长度
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        avg_sentence_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length > 100:
            complexity += 0.2
        elif avg_sentence_length < 30:
            complexity -= 0.1
            
        # 2. 特殊字符密度
        special_chars = set('{}[]()<>@#$%^&*+=\\|~`')
        special_char_ratio = len([c for c in text if c in special_chars]) / len(text)
        complexity += special_char_ratio * 2
        
        # 3. 数字和符号密度
        symbol_ratio = len([c for c in text if not c.isalnum()]) / len(text)
        complexity += symbol_ratio
        
        # 4. 中文字符比例
        chinese_ratio = len([c for c in text if '\u4e00' <= c <= '\u9fff']) / len(text)
        complexity += chinese_ratio * 0.3
        
        # 限制复杂度范围
        return max(0.5, min(1.5, complexity))
        
    def _get_optimal_chunk_size(self, text: str) -> int:
        """
        根据内容类型和复杂度确定最优块大小
        """
        content_type = self._detect_content_type(text)
        complexity = self._estimate_text_complexity(text)
        
        # 基础块大小 - 使用更大的基础大小
        base_size = self.max_tokens_per_chunk // 2  # 预留一半空间给翻译结果
        
        # 根据内容类型调整
        type_multiplier = {
            'code': 0.8,  # 代码块需要保持完整，但通常较短
            'table': 0.7,  # 表格需要完整处理
            'text': 1.0   # 普通文本使用标准大小
        }
        
        # 计算最终大小
        optimal_size = int(base_size * type_multiplier[content_type] / complexity)
        
        # 确保不超过模型限制，但保持足够大的大小
        return min(max(optimal_size, 1000), self.model_config['max_tokens'] - 1000)  # 留出1000个token的缓冲

    def _optimize_overlap(self, prev_text: str, next_text: str) -> int:
        """
        优化两个文本块之间的重叠大小
        """
        # 基础重叠大小
        base_overlap = self.overlap_tokens
        
        # 检测内容类型
        prev_type = self._detect_content_type(prev_text)
        next_type = self._detect_content_type(next_text)
        
        # 如果两边都是代码或表格，增加重叠
        if prev_type in ['code', 'table'] or next_type in ['code', 'table']:
            base_overlap = int(base_overlap * 1.5)
            
        # 根据复杂度调整重叠
        prev_complexity = self._estimate_text_complexity(prev_text)
        next_complexity = self._estimate_text_complexity(next_text)
        avg_complexity = (prev_complexity + next_complexity) / 2
        
        # 复杂度越高，重叠越大
        overlap = int(base_overlap * avg_complexity)
        
        return min(overlap, self.model_config['max_tokens'] // 4)  # 限制最大重叠大小

    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        """
        # 使用tiktoken估算token数量
        try:
            encoder = tiktoken.encoding_for_model(self.model_name)
            return len(encoder.encode(text))
        except Exception as e:
            logging.warning(f"无法使用tiktoken计算token: {str(e)}")
            # 使用简单的估算方法作为后备
            return len(text) if any('\u4e00' <= c <= '\u9fff' for c in text) else len(text) // 4

    def _clean_text(self, text: str) -> str:
        """清理文本，移除不必要的空白字符和格式

        Args:
            text (str): 原始文本

        Returns:
            str: 清理后的文本
        """
        if not text:
            return ""
            
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除行首和行尾的空白字符
        text = text.strip()
        
        # 移除不可见字符，但保留换行符
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r')
        
        return text

    def _post_process_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """对翻译后的文本块进行后处理

        Args:
            blocks (List[Dict]): 翻译后的文本块列表

        Returns:
            List[Dict]: 处理后的文本块列表
        """
        processed_blocks = []
        for block in blocks:
            if not block.get('content'):
                continue
                
            # 清理文本内容
            if block.get('type') == 'text':
                block['content'] = self._clean_text(block['content'])
            
            # 保持图片和表格块不变
            processed_blocks.append(block)
            
        return processed_blocks

    def _is_header_footer(self, text: str, y_position: float, page_height: float) -> bool:
        """判断文本是否为页眉页脚

        Args:
            text (str): 文本内容
            y_position (float): 文本在页面中的垂直位置
            page_height (float): 页面高度

        Returns:
            bool: 是否为页眉页脚
        """
        # 如果文本为空，不是页眉页脚
        if not text:
            return False
            
        # 清理文本
        text = self._clean_text(text)
        
        # 页眉页脚的典型特征
        header_footer_patterns = [
            r'^\d+$',  # 纯数字（页码）
            r'^Page \d+$',  # "Page X" 格式
            r'^第\d+页$',  # 中文页码
            r'^Chapter \d+$',  # 章节标记
            r'^第\d+章$',  # 中文章节标记
        ]
        
        # 检查文本是否匹配页眉页脚模式
        for pattern in header_footer_patterns:
            if re.match(pattern, text):
                return True
                
        # 检查位置（页面顶部10%或底部10%的区域）
        margin_threshold = page_height * 0.1
        if y_position < margin_threshold or y_position > (page_height - margin_threshold):
            # 在页眉页脚区域的短文本更可能是页眉页脚
            if len(text) < 50:
                return True
                
        return False

    def _analyze_block(self, block: dict, prev_block: dict = None) -> dict:
        """分析文本块的特征

        Args:
            block (dict): 文本块信息
            prev_block (dict, optional): 上一个文本块信息. Defaults to None.

        Returns:
            dict: 文本块特征分析结果
        """
        analysis = {
            'is_header_footer': False,
            'is_title': False,
            'is_list_item': False,
            'indentation': 0,
            'font_size': 0,
            'font_style': set(),
            'text': '',
            'position': {
                'top': 0,
                'bottom': 0,
                'left': 0,
                'right': 0
            }
        }
        
        try:
            # 提取位置信息
            if 'bbox' in block:
                x0, y0, x1, y1 = block['bbox']
                analysis['position'] = {
                    'top': y0,
                    'bottom': y1,
                    'left': x0,
                    'right': x1
                }
                
                # 检查是否是页眉页脚
                if y0 < 100 or y1 > 500:
                    analysis['is_header_footer'] = True
            
            # 提取文本内容
            if 'lines' in block:
                text_parts = []
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            text_parts.append({
                                'text': text,
                                'font': span.get("font", ""),
                                'size': span.get("size", 12),
                                'flags': span.get("flags", 0)
                            })
                            
                            # 分析字体样式
                            if 'size' in span:
                                analysis['font_size'] = max(analysis['font_size'], span['size'])
                            
                            if 'flags' in span:
                                if span['flags'] & 2**0:  # 粗体
                                    analysis['font_style'].add('bold')
                                if span['flags'] & 2**1:  # 斜体
                                    analysis['font_style'].add('italic')
                                if span['flags'] & 2**2:  # 下划线
                                    analysis['font_style'].add('underline')
                
                analysis['text'] = ' '.join(line["text"] for line in text_parts)
                
                # 检查是否是标题
                if analysis['font_size'] > 12 or 'bold' in analysis['font_style']:
                    analysis['is_title'] = True
                
                # 检查是否是列表项
                text = analysis['text']
                if text.startswith(('•', '-', '*', '1.', '2.', '3.')) or \
                   any(text.startswith(f"{i}.") for i in range(1, 10)):
                    analysis['is_list_item'] = True
                
                # 计算缩进
                if 'lines' in block and block['lines']:
                    first_line = block['lines'][0]
                    if 'spans' in first_line and first_line['spans']:
                        first_span = first_line['spans'][0]
                        if 'bbox' in first_span:
                            analysis['indentation'] = first_span['bbox'][0]
            
        except Exception as e:
            logging.warning(f"分析文本块时出错: {str(e)}")
        
        return analysis

    def translate_file(self, file_path: str, source_lang: str = None, target_lang: str = None) -> str:
        """翻译文件
        
        Args:
            file_path (str): 文件路径
            source_lang (str, optional): 源语言. Defaults to None.
            target_lang (str, optional): 目标语言. Defaults to None.
            
        Returns:
            str: 翻译后的文件路径
        """
        if not source_lang:
            source_lang = self.source_lang
        if not target_lang:
            target_lang = self.target_lang
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        try:
            self.logger.info(f"开始翻译文件: {os.path.basename(file_path)}")
            self.logger.info(f"源语言: {source_lang}")
            self.logger.info(f"目标语言: {target_lang}")
            self.logger.info(f"使用模型: {self.model_name}")

            # 获取文件扩展名
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 根据文件类型处理内容
            if file_ext == '.pdf':
                # 1. 提取PDF内容并保持格式
                content_blocks = self._extract_pdf_content(file_path)
                
                # 2. 将内容分割成适合翻译的块
                translation_blocks = []
                for block in content_blocks:
                    if block['type'] == 'text':
                        # 翻译文本块
                        translated_content = self._initial_translation(block['content'], source_lang, target_lang)
                        block['translation'] = translated_content
                    else:
                        # 图片和其他类型的块保持不变
                        block['translation'] = block['content']
                    translation_blocks.append(block)
                    
                # 3. 转换为Markdown
                content = self._blocks_to_markdown(translation_blocks)
            
            elif file_ext in ['.txt', '.md']:
                # 直接读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 直接翻译内容
                content = self._initial_translation(content, source_lang, target_lang)
            
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
            
            # 生成输出文件路径
            output_dir = os.path.dirname(file_path)
            if not output_dir:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{target_lang}.md")
            
            # 保存翻译结果
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.logger.info(f"\n翻译完成！输出文件: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"翻译失败: {str(e)}")
            raise

    def _extract_pdf_content(self, pdf_path: str) -> List[Dict]:
        """从PDF提取内容,包括文本、图片、表格等
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含内容块的列表,每个块包含类型和内容
        """
        content_blocks = []
        
        try:
            # 创建图片保存目录
            images_dir = os.path.join(os.path.dirname(pdf_path), 'images')
            os.makedirs(images_dir, exist_ok=True)
            
            # 打开PDF文件
            doc = fitz.open(pdf_path)
            
            # 遍历每一页
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本
                text_blocks = page.get_text("dict", sort=True)["blocks"]
                page_text_blocks = []
                
                for b in text_blocks:
                    if "lines" in b:
                        block_text = []
                        for line in b["lines"]:
                            line_text = []
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text.append(text)
                            if line_text:
                                block_text.append(" ".join(line_text))
                        if block_text:
                            text_content = " ".join(block_text)
                            page_text_blocks.append({
                                "type": "text",
                                "content": text_content,
                                "rect": b["bbox"],
                                "page": page_num + 1
                            })
                
                # 合并同一页面中的相邻文本块
                if page_text_blocks:
                    # 按垂直位置排序
                    page_text_blocks.sort(key=lambda x: x["rect"][1])
                    
                    # 初始化合并后的块
                    merged_blocks = []
                    current_block = page_text_blocks[0]
                    merged_text = [current_block["content"]]
                    
                    # 遍历剩余的块
                    for next_block in page_text_blocks[1:]:
                        # 如果两个块在垂直方向上足够接近（比如小于行高的1.5倍）
                        if abs(next_block["rect"][1] - current_block["rect"][3]) < 20:  # 可以根据实际情况调整这个阈值
                            merged_text.append(next_block["content"])
                        else:
                            # 保存当前合并的块
                            merged_blocks.append({
                                "type": "text",
                                "content": " ".join(merged_text),
                                "rect": [
                                    current_block["rect"][0],
                                    current_block["rect"][1],
                                    next_block["rect"][2],
                                    next_block["rect"][3]
                                ],
                                "page": current_block["page"]
                            })
                            # 开始新的块
                            current_block = next_block
                            merged_text = [current_block["content"]]
                    
                    # 添加最后一个合并的块
                    merged_blocks.append({
                        "type": "text",
                        "content": " ".join(merged_text),
                        "rect": current_block["rect"],
                        "page": current_block["page"]
                    })
                    
                    # 将合并后的块添加到内容列表
                    content_blocks.extend(merged_blocks)
                
                # 提取图片
                images = page.get_images()
                for img_num, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image:
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            img_filename = f"image_p{page_num + 1}_{img_num + 1}.{image_ext}"
                            img_path = os.path.join(images_dir, img_filename)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            
                            content_blocks.append({
                                'type': 'image',
                                'content': f"![{img_filename}](images/{img_filename})",
                                'page': page_num + 1
                            })
                    except Exception as e:
                        self.logger.error(f"处理图片时出错: {str(e)}")
            
            # 按页码和位置排序
            content_blocks.sort(key=lambda x: (x['page'], x['rect'][1] if 'rect' in x else 0))
            
            return content_blocks
            
        except Exception as e:
            self.logger.error(f"提取PDF内容时出错: {str(e)}")
            raise
        finally:
            if 'doc' in locals():
                doc.close()
        
        return content_blocks

    def _blocks_to_markdown(self, blocks: List[Dict]) -> str:
        """将翻译后的块转换为完整的Markdown文档
        
        Args:
            blocks: 翻译后的内容块列表
            
        Returns:
            str: Markdown格式的文档
        """
        markdown_lines = []
        current_page = 1
        
        for block in blocks:
            # 添加分页符
            if block['page'] != current_page:
                current_page = block['page']
                if markdown_lines:  # 不是第一页时添加分隔符
                    markdown_lines.extend(["\n\n---\n\n"])
            
            content = block['translation']
            
            if block['type'] == 'text':
                markdown_lines.append(content)
                markdown_lines.append("\n\n")
            elif block['type'] == 'image':
                markdown_lines.append(block['content'])
                markdown_lines.append("\n\n")
        
        return "".join(markdown_lines)

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """调用 GPTsAPI，包含重试逻辑
        
        Args:
            system_prompt (str): 系统提示
            user_prompt (str): 用户提示
            
        Returns:
            str: API响应的文本内容
        """
        max_retries = 5
        retry_delay = 10
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"正在发送API请求 (尝试 {attempt + 1}/{max_retries})...")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # 设置SSL验证选项
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # 配置请求
                import urllib3
                urllib3.disable_warnings()
                openai.verify_ssl_certs = False
                
                # 使用openai库调用API
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.model_config['max_output_tokens'],  # 设置最大输出token数
                    request_timeout=30,  # 设置30秒超时
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Connection": "keep-alive"
                    }
                )
                
                self.logger.info("API请求成功，正在处理响应...")
                
                # 验证响应格式
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("无效的API响应格式")
                
                if not hasattr(response.choices[0], 'message') or not response.choices[0].message:
                    raise ValueError("响应中没有有效内容")
                
                result = response.choices[0].message['content'].strip()
                
                # 验证响应内容
                if not result:
                    raise ValueError("响应内容为空")
                
                self.logger.info(f"成功获取响应 (长度: {len(result)} 字符)")
                return result
                
            except Exception as e:
                self.logger.error(f"API请求错误: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.error(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    self.logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"API调用失败，已达到最大重试次数: {str(e)}")

    def _contains_special_content(self, text: str) -> Dict[str, bool]:
        """检查文本是否包含特殊内容
        
        Args:
            text (str): 要检查的文本
            
        Returns:
            Dict[str, bool]: 包含各种特殊内容的标记
        """
        patterns = {
            'code': [
                r'```[\s\S]*?```',   # Markdown代码块
                r'~~~[\s\S]*?~~~',   # 另一种代码块标记
                r'`[^`]+`',          # 行内代码
                r'<code>.*?</code>', # HTML代码标签
                r'(?m)^    .*$',     # 缩进代码块
                r'(?m)^\t.*$',       # Tab缩进代码块
            ],
            'formula': [
                r'\$\$[\s\S]*?\$\$',  # LaTeX数学公式块
                r'\$[^$\n]+\$',       # LaTeX行内公式
                r'\\\[[\s\S]*?\\\]',  # LaTeX显示公式
                r'\\\([\s\S]*?\\\)',  # LaTeX行内公式
                r'\\begin\{equation\}[\s\S]*?\\end\{equation\}',  # LaTeX方程环境
                r'\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}',  # LaTeX对齐环境
            ],
            'table': [
                r'\|[^\n]*\|',        # Markdown表格行
                r'\+[-=]+\+',         # ASCII表格边框
                r'┌[─┬]+┐',           # Unicode表格边框
                r'^\s*[\|\+][-\+]+[\|\+]\s*$',  # 表格分隔符
            ],
            'image': [
                r'!\[([^\]]*)\]\(([^\)]+)\)',  # Markdown图片
                r'<img[^>]+>',                  # HTML图片标签
            ],
            'diagram': [
                r'```(?:mermaid|dot|plantuml)[\s\S]*?```',  # Mermaid/Graphviz/PlantUML图表
                r'```(?:chart|graph)[\s\S]*?```',           # 其他图表格式
            ]
        }
        
        result = {}
        for content_type, pattern_list in patterns.items():
            result[content_type] = any(re.search(pattern, text, re.MULTILINE) for pattern in pattern_list)
            
        return result

    def _preserve_special_content(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """保护特殊内容，用占位符替换它们
        
        Args:
            text (str): 原始文本
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: 处理后的文本和特殊内容列表
        """
        placeholders = []
        processed_text = text
        
        # 定义需要保护的模式
        patterns = {
            'code': [
                (r'```[\s\S]*?```', 'CODE_BLOCK'),
                (r'~~~[\s\S]*?~~~', 'CODE_BLOCK'),
                (r'`[^`]+`', 'INLINE_CODE'),
            ],
            'formula': [
                (r'\$\$[\s\S]*?\$\$', 'MATH_BLOCK'),
                (r'\$[^$\n]+\$', 'INLINE_MATH'),
                (r'\\\[[\s\S]*?\\\]', 'MATH_DISPLAY'),
                (r'\\\([\s\S]*?\\\)', 'MATH_INLINE'),
                (r'\\begin\{equation\}[\s\S]*?\\end\{equation\}', 'EQUATION'),
                (r'\\begin\{align\*?\}[\s\S]*?\\end\{align\*?\}', 'ALIGN'),
            ],
            'table': [
                (r'(?m)^\|.*\|$[\n\r]*(?:\|[-:]+\|[-:\n\r]+)*', 'TABLE'),
            ],
            'image': [
                (r'!\[([^\]]*)\]\(([^\)]+)\)', 'IMAGE'),
                (r'<img[^>]+>', 'HTML_IMAGE'),
            ],
            'diagram': [
                (r'```(?:mermaid|dot|plantuml)[\s\S]*?```', 'DIAGRAM'),
            ]
        }
        
        placeholder_id = 0
        
        for content_type, pattern_list in patterns.items():
            for pattern, tag in pattern_list:
                matches = list(re.finditer(pattern, processed_text, re.MULTILINE | re.DOTALL))
                for match in reversed(matches):  # 从后向前处理，避免位置变化
                    content = match.group(0)
                    placeholder = f"<{tag}_{placeholder_id}>"
                    placeholders.append({
                        'id': placeholder_id,
                        'type': content_type,
                        'tag': tag,
                        'content': content,
                        'start': match.start(),
                        'end': match.end()
                    })
                    processed_text = processed_text[:match.start()] + placeholder + processed_text[match.end():]
                    placeholder_id += 1
        
        return processed_text, placeholders

    def _restore_special_content(self, text: str, placeholders: List[Dict[str, Any]]) -> str:
        """还原特殊内容
        
        Args:
            text (str): 包含占位符的文本
            placeholders (List[Dict[str, Any]]: 特殊内容列表
            
        Returns:
            str: 还原后的文本
        """
        restored_text = text
        
        # 按ID从大到小排序，避免位置冲突
        for placeholder in sorted(placeholders, key=lambda x: x['id'], reverse=True):
            pattern = f"<{placeholder['tag']}_{placeholder['id']}>"
            content = placeholder['content']
            
            # 特殊处理图片标题
            if placeholder['type'] == 'image' and '![' in content:
                # 提取图片标题和URL
                match = re.match(r'!\[(.*?)\]\((.*?)\)', content)
                if match:
                    title, url = match.groups()
                    # 只翻译标题，保持URL不变
                    if title:
                        translated_title = self._translate_text(title, self.source_lang, self.target_lang)
                        content = f'![{translated_title}]({url})'
            
            restored_text = restored_text.replace(pattern, content)
        
        return restored_text

    def _split_text(self, text: str) -> List[str]:
        """将文本分割成适合模型处理的片段，使用更智能的语义分割策略
        
        Args:
            text (str): 要分割的文本
            
        Returns:
            List[str]: 文本片段列表
        """
        if not text:
            return []
            
        # 首先保护特殊内容
        processed_text, placeholders = self._preserve_special_content(text)
        
        # 获取最优块大小
        optimal_chunk_size = self._get_optimal_chunk_size(processed_text)
        
        # 分割文本
        chunks = []
        current_chunk = []
        current_size = 0
        
        paragraphs = re.split(r'\n\s*\n', processed_text)
        
        for para in paragraphs:
            para_size = self._estimate_tokens(para)
            
            if para_size > optimal_chunk_size:
                # 如果段落太长，按句子分割
                sentences = re.split(r'([。！？.!?])', para)
                current_sentence = ''
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]  # 添加分隔符
                        
                    sentence_size = self._estimate_tokens(sentence)
                    
                    if current_size + sentence_size > optimal_chunk_size and current_chunk:
                        # 保存当前块
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
            else:
                # 如果当前段落加入会超过大小限制，保存当前块
                if current_size + para_size > optimal_chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(para)
                current_size += para_size
        
        # 保存最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # 还原每个块中的特殊内容
        restored_chunks = [self._restore_special_content(chunk, placeholders) for chunk in chunks]
        
        return restored_chunks

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='PDF文档翻译工具')
    parser.add_argument('input_file', help='输入PDF文件路径')
    parser.add_argument('--source-lang', default='英文', help='源语言，默认为英文')
    parser.add_argument('--target-lang', default='中文', help='目标语言，默认为中文')
    parser.add_argument('--model', default='gpt-4o-mini', help='使用的模型名称')
    args = parser.parse_args()

    try:
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

        # 创建翻译器实例
        translator = ReflectiveTranslator(
            api_key=api_key,
            model_name=args.model,
            source_lang=args.source_lang,
            target_lang=args.target_lang
        )

        # 开始翻译
        translator.logger.info(f"开始翻译文件: {args.input_file}")
        translator.logger.info(f"源语言: {args.source_lang}")
        translator.logger.info(f"目标语言: {args.target_lang}")
        translator.logger.info(f"使用模型: {args.model}")

        output_file = translator.translate_file(args.input_file)
        translator.logger.info(f"\n翻译完成！输出文件: {output_file}")

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
