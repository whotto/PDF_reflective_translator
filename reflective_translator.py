import os
import sys
import json
import time
import logging
import argparse
import PyPDF2
import openai
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)

class ReflectiveTranslator:
    # 模型配置
    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "max_tokens": 128000,  # 128K 上下文窗口
            "cost_per_1k_tokens": 0.01,  # 示例成本
            "recommended_chunk_size": 100000,  # 建议的分块大小
            "max_output_tokens": 20000,  # 最大输出token数
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

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens_per_chunk: int = 16000,  
        overlap_tokens: int = 100,
        cost_limit: float = 10.0
    ):
        """初始化翻译器
        
        Args:
            api_key (str): OpenAI API密钥
            model_name (str, optional): 模型名称. Defaults to "gpt-4o-mini".
            temperature (float, optional): 温度参数. Defaults to 0.7.
            max_tokens_per_chunk (int, optional): 每个块的最大token数. Defaults to 16000.
            overlap_tokens (int, optional): 重叠token数. Defaults to 100.
            cost_limit (float, optional): 成本限制. Defaults to 10.0.
        """
        # 验证模型支持
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {', '.join(self.MODEL_CONFIGS.keys())}")
        
        # 获取模型配置
        self.model_config = self.MODEL_CONFIGS[model_name]
        
        # 设置分块大小
        if max_tokens_per_chunk is None:
            max_tokens_per_chunk = self.model_config['recommended_chunk_size']
        elif max_tokens_per_chunk > self.model_config['max_tokens']:
            raise ValueError(f"分块大小超过模型最大上下文长度: {max_tokens_per_chunk} > {self.model_config['max_tokens']}")
        
        # 加载环境变量
        load_dotenv()
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        # 验证其他参数
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens 不能为负数")
        if overlap_tokens >= max_tokens_per_chunk:
            raise ValueError("overlap_tokens 必须小于 max_tokens_per_chunk")
        if cost_limit is not None and cost_limit <= 0:
            raise ValueError("cost_limit 必须大于0")
        
        # 设置API配置
        self.api_urls = [
            "https://api.gptsapi.net/v1",
        ]
        
        # 保存配置
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.cost_limit = cost_limit
        self.total_cost = 0
        
        # 加载术语表
        self.glossary = self._load_glossary()
        
        # 创建进度保存目录
        self.progress_dir = Path("translation_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        # 初始化日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志配置"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"translation_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.INFO)

    def _load_glossary(self) -> Dict:
        """加载术语表"""
        try:
            with open('glossary.yaml', 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning("未找到术语表文件，将使用空术语表")
            return {}

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _call_openai(self, messages: List[Dict]) -> str:
        """调用OpenAI API
        
        Args:
            messages (List[Dict]): 消息列表
            
        Returns:
            str: API响应的文本内容
        """
        last_error = None
        
        for api_url in self.api_urls:
            try:
                logging.info(f"尝试使用API端点: {api_url}")
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # 计算输入tokens并限制max_tokens
                input_tokens = sum(len(m.get('content', '')) for m in messages) * 1.5  # 粗略估计
                max_output_tokens = min(16000, self.max_tokens_per_chunk)  # 确保不超过模型限制
                
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": max_output_tokens,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
                
                response = requests.post(
                    api_url + "/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                logging.info(f"HTTP Request: {response.request.method} {response.request.url} {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        if 'message' in result['choices'][0]:
                            return result['choices'][0]['message']['content'].strip()
                        else:
                            raise ValueError(f"API响应缺少message字段: {result}")
                    else:
                        raise ValueError(f"API响应缺少choices字段: {result}")
                else:
                    raise ValueError(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
                    
            except Exception as e:
                last_error = e
                logging.warning(f"API调用失败: {str(e)}, 尝试下一个端点")
                continue
        
        raise Exception(f"所有API端点都调用失败。最后的错误: {str(last_error)}")

    def _initial_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """初始翻译，保留专有名词
        
        Args:
            text (str): 要翻译的文本
            source_lang (str): 源语言
            target_lang (str): 目标语言
            
        Returns:
            str: 翻译结果
        """
        prompt = f"""请将以下{source_lang}文本翻译成{target_lang}。

要求：
1. 以专业译者的水准进行翻译，确保译文通顺、准确、优雅
2. 保持专业术语的准确性，不翻译专有名词（如人名、地名、产品名等）
3. 遵循目标语言的表达习惯和写作风格
4. 保持原文的语气和语调
5. 数学公式、代码片段等专业内容保持原样
6. 对于复杂的长句，可以适当调整语序以提高可读性
7. 确保译文的连贯性和上下文的一致性

原文：
{text}"""

        messages = [
            {"role": "system", "content": "你是一位专业的翻译专家，精通多国语言，擅长学术和技术文献的翻译工作。你的翻译不仅准确，而且自然流畅，符合目标语言的表达习惯。"},
            {"role": "user", "content": prompt}
        ]

        response = self._call_openai(messages)
        return response.strip()

    def _reflective_review(self, original_text: str, translation: str, source_lang: str, target_lang: str) -> str:
        """反思评审阶段"""
        system_prompt = """# Role: 翻译评审助手

## Profile:
- 专注于运用"信达雅"翻译理论进行评审
- 致力于提升译文的忠实度、通达性和雅致度
- 确保译文准确传达原文内容和意图

## Goals:
1. 提升忠实度：识别并纠正译文中的偏差
2. 优化通达性：提高译文的可读性和流畅度
3. 增强雅致度：提升语言美感和表达优雅度

## Skills:
1. 忠实度分析：深入理解原文，识别信息遗漏或误解
2. 通达性分析：评估流畅度和易懂性，识别生硬表达
3. 雅致度分析：评估语言美感和风格统一性

## Constraints:
1. 忠实原文：避免曲解或偏离原意
2. 具体可行：提供实际可操作的修改方案
3. 保持完整：确保评审覆盖全部内容要素
"""

        user_prompt = f"""请对以下翻译进行系统性评审：

## 原文（{source_lang}）：
{original_text}

## 译文（{target_lang}）：
{translation}

请从"信达雅"三个维度进行评估：
1. 信（忠实度）：
   - 译文是否准确传达原文内容和意图
   - 是否存在信息遗漏或误解
   - 专业术语使用是否准确

2. 达（通达性）：
   - 译文是否流畅自然
   - 是否符合{target_lang}的表达习惯
   - 句式结构是否合理

3. 雅（优雅度）：
   - 用词是否优美得体
   - 风格是否统一
   - 是否符合文体要求

请提供具体的改进建议，包括：
1. 问题描述：说明问题所属维度和具体表现
2. 改进建议：提供具体的修改方案
3. 参考示例：给出修改示例
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._call_openai(messages)

    def _final_revision(self, original_text: str, translation: str, review_comments: str, source_lang: str, target_lang: str) -> str:
        """终稿修改阶段"""
        system_prompt = """# Role: 翻译改进专家

## Profile:
- 专业翻译改进专家
- 致力于将翻译初稿提升至最高质量水平
- 严格遵循翻译优化清单

## Goals:
1. 全面采纳评审建议
2. 确保术语准确性
3. 提升文本流畅度
4. 优化语言表达
5. 统一文体风格

## Skills:
1. 精准修改能力
2. 术语规范应用
3. 文体风格统一
4. 逻辑连贯性把控

## Constraints:
1. 严格依据评审意见
2. 保持原文意图
3. 确保修改可追溯
4. 避免引入新问题
"""

        user_prompt = f"""请根据评审意见对译文进行终稿修改：

## 原文（{source_lang}）：
{original_text}

## 当前译文（{target_lang}）：
{translation}

## 评审意见：
{review_comments}

请进行以下优化：
1. 根据评审意见逐条修改
2. 确保专业术语准确性和一致性
3. 优化文本结构和表达方式
4. 提升整体文字质量
5. 保持文体风格统一

请输出修改后的最终译文。
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._call_openai(messages)

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
        """将文本分割成适合模型处理的片段
        
        Args:
            text (str): 要分割的文本
            
        Returns:
            List[str]: 文本片段列表
        """
        # 按段落分割
        paragraphs = text.split('\n\n')
        segments = []
        current_segment = []
        current_length = 0
        max_length = self.max_tokens_per_chunk // 2  # 预留空间给翻译结果
        
        for paragraph in paragraphs:
            # 估算段落长度（中文字符计1，其他计0.5）
            para_length = sum(1 if '\u4e00' <= c <= '\u9fff' else 0.5 for c in paragraph)
            
            if current_length + para_length > max_length and current_segment:
                # 当前段落加入会超出限制，先保存当前片段
                segments.append('\n\n'.join(current_segment))
                current_segment = []
                current_length = 0
            
            current_segment.append(paragraph)
            current_length += para_length
        
        # 保存最后一个片段
        if current_segment:
            segments.append('\n\n'.join(current_segment))
        
        return segments

    def _merge_translations(self, translations: List[str], overlap_tokens: int) -> str:
        """合并翻译片段，处理重叠部分
        
        Args:
            translations (List[str]): 翻译片段列表
            overlap_tokens (int): 片段间重叠的token数
            
        Returns:
            str: 合并后的翻译文本
        """
        if not translations:
            return ""
        
        merged = translations[0]
        for i in range(1, len(translations)):
            current = translations[i]
            # 寻找重叠部分
            overlap = self._find_overlap(merged[-overlap_tokens:], current[:overlap_tokens])
            if overlap:
                merged = merged[:-len(overlap)] + current
            else:
                merged += '\n' + current
        
        return merged

    def _find_overlap(self, end: str, start: str) -> str:
        """找到两个文本片段的重叠部分
        
        Args:
            end (str): 第一个片段的结尾
            start (str): 第二个片段的开头
            
        Returns:
            str: 重叠部分
        """
        min_overlap = 10  # 最小重叠长度
        for i in range(len(end), min_overlap-1, -1):
            if end[-i:] == start[:i]:
                return end[-i:]
        return ""

    def _save_progress(self, file_path: str, translations: List[Dict]):
        """保存翻译进度
        
        Args:
            file_path (str): 原文件路径
            translations (List[Dict]): 已完成的翻译列表
        """
        try:
            # 确保进度目录存在
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存进度
            progress_file = self.progress_dir / f"{Path(file_path).stem}_progress.json"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.warning(f"保存进度失败: {str(e)}")

    def _load_progress(self, file_path: str) -> Optional[List[Dict]]:
        """加载翻译进度
        
        Args:
            file_path (str): 原文件路径
            
        Returns:
            Optional[List[Dict]]: 已完成的翻译列表，如果没有进度则返回None
        """
        try:
            progress_file = self.progress_dir / f"{Path(file_path).stem}_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logging.warning(f"加载进度失败: {str(e)}")
            return None

    def _save_as_markdown(self, output_path: str, translations: List[Dict]):
        """保存翻译结果为Markdown格式
        
        Args:
            output_path (str): 输出文件路径
            translations (List[Dict]): 翻译结果列表
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入标题
            f.write(f"# {Path(output_path).stem}\n\n")
            
            # 写入翻译内容
            for item in translations:
                if item['type'] == 'text':
                    if item['format'].get('is_title'):
                        # 标题使用对应的markdown标题级别
                        level = item['format'].get('title_level', 2)
                        f.write(f"{'#' * level} {item['translation']}\n\n")
                    elif item['format'].get('has_formula'):
                        # 数学公式
                        f.write(f"```math\n{item['content']}\n```\n\n")
                    else:
                        # 普通文本
                        f.write(f"{item['translation']}\n\n")
                elif item['type'] == 'image':
                    # 图片引用
                    f.write(f"![{item.get('caption', 'Image')}]\n\n")

    def _extract_text(self, file_path: str) -> List[Dict]:
        """从PDF文件中提取文本，保持原格式并识别图片位置
        
        Args:
            file_path (str): PDF文件路径
            
        Returns:
            List[Dict]: 包含文本内容、格式信息的列表
        """
        if not file_path.lower().endswith('.pdf'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return [{'type': 'text', 'content': f.read(), 'format': {}}]
        
        content_blocks = []
        current_text = []
        current_page = 0
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                logging.info(f"PDF共有 {total_pages} 页")
                
                for page_num in range(total_pages):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        # 分割文本为段落
                        paragraphs = []
                        current_para = []
                        
                        for line in text.split('\n'):
                            line = line.strip()
                            if not line:  # 空行表示段落结束
                                if current_para:
                                    paragraphs.append(' '.join(current_para))
                                    current_para = []
                            else:
                                # 检查是否是新段落的开始（通过缩进或其他特征）
                                if (line.startswith('    ') or  # 缩进
                                    line[0].isupper() or  # 大写字母开头
                                    any(line.startswith(p) for p in ['•', '-', '*', '1.', '2.'])):  # 列表标记
                                    if current_para:
                                        paragraphs.append(' '.join(current_para))
                                        current_para = []
                                current_para.append(line)
                        
                        if current_para:  # 添加最后一个段落
                            paragraphs.append(' '.join(current_para))
                        
                        # 处理段落
                        for para in paragraphs:
                            para = para.strip()
                            if not para:
                                continue
                                
                            # 检测是否为标题
                            is_title = (len(para) < 100 and 
                                      (para.isupper() or 
                                       para.startswith('#') or 
                                       any(char.isdigit() for char in para[:3]) or
                                       para.endswith(':') or
                                       para.endswith('?')))
                            
                            # 检测是否包含数学公式
                            has_formula = ('\\' in para or 
                                         '$' in para or 
                                         any(sym in para for sym in ['∑', '∫', 'π', '≈', '±']))
                            
                            # 如果是新页面，先保存之前的内容
                            if current_page != page_num + 1 and current_text:
                                content_blocks.append({
                                    'type': 'text',
                                    'content': '\n\n'.join(current_text),
                                    'format': {
                                        'page': current_page,
                                        'is_title': False,
                                        'has_formula': False
                                    }
                                })
                                current_text = []
                            
                            current_page = page_num + 1
                            
                            # 标题和公式单独作为块
                            if is_title or has_formula:
                                if current_text:  # 先保存之前的普通文本
                                    content_blocks.append({
                                        'type': 'text',
                                        'content': '\n\n'.join(current_text),
                                        'format': {
                                            'page': current_page,
                                            'is_title': False,
                                            'has_formula': False
                                        }
                                    })
                                    current_text = []
                                
                                content_blocks.append({
                                    'type': 'text',
                                    'content': para,
                                    'format': {
                                        'page': current_page,
                                        'is_title': is_title,
                                        'has_formula': has_formula
                                    }
                                })
                            else:
                                current_text.append(para)
                    
                    # 检查图片
                    if '/XObject' in page:
                        for obj in page['/XObject'].get_object().values():
                            if obj['/Subtype'] == '/Image':
                                # 先保存当前文本
                                if current_text:
                                    content_blocks.append({
                                        'type': 'text',
                                        'content': '\n\n'.join(current_text),
                                        'format': {
                                            'page': current_page,
                                            'is_title': False,
                                            'has_formula': False
                                        }
                                    })
                                    current_text = []
                                
                                content_blocks.append({
                                    'type': 'image',
                                    'format': {
                                        'page': current_page
                                    }
                                })
                    
                    logging.info(f"已处理第 {page_num + 1}/{total_pages} 页")
                
                # 保存最后的文本块
                if current_text:
                    content_blocks.append({
                        'type': 'text',
                        'content': '\n\n'.join(current_text),
                        'format': {
                            'page': current_page,
                            'is_title': False,
                            'has_formula': False
                        }
                    })
            
            return content_blocks
            
        except Exception as e:
            logging.error(f"处理PDF时出错: {str(e)}")
            raise

    def translate_file(self, file_path: str, source_lang: str, target_lang: str):
        """翻译文件
        
        Args:
            file_path (str): 文件路径
            source_lang (str): 源语言
            target_lang (str): 目标语言
        """
        try:
            # 提取文本和格式信息
            content_blocks = self._extract_text(file_path)
            logging.info(f"成功提取文本，共 {len(content_blocks)} 个内容块")
            
            # 准备输出文件
            output_path = str(Path(file_path).with_stem(f"{Path(file_path).stem}_{target_lang}"))
            if not output_path.endswith('.md'):
                output_path += '.md'
            
            # 加载已有进度或初始化新的翻译列表
            translations = []
            start_index = 0
            
            progress_file = self.progress_dir / f"{Path(file_path).stem}_progress.json"
            if progress_file.exists():
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        saved_translations = json.load(f)
                        if isinstance(saved_translations, list):
                            translations = saved_translations
                            start_index = len(saved_translations)
                except Exception as e:
                    logging.warning(f"加载进度失败: {str(e)}")
            
            # 翻译每个文本块
            for i, block in enumerate(content_blocks[start_index:], start=start_index):
                try:
                    logging.info(f"正在翻译第 {i + 1}/{len(content_blocks)} 个内容块")
                    
                    if block['type'] == 'text':
                        # 初始翻译
                        translation = self._initial_translation(
                            block['content'], source_lang, target_lang
                        )
                        
                        # 创建新的翻译块
                        translated_block = {
                            'type': 'text',
                            'content': block['content'],
                            'translation': translation,
                            'format': dict(block.get('format', {}))
                        }
                    else:
                        # 对于非文本块，创建新的字典
                        translated_block = {
                            'type': block['type'],
                            'content': block.get('content', ''),
                            'format': dict(block.get('format', {}))
                        }
                    
                    # 添加到翻译结果列表
                    translations.append(translated_block)
                    
                    # 定期保存进度
                    if (i + 1) % 5 == 0:
                        try:
                            self.progress_dir.mkdir(parents=True, exist_ok=True)
                            with open(progress_file, 'w', encoding='utf-8') as f:
                                json.dump(translations, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            logging.warning(f"保存进度失败: {str(e)}")
                    
                    # 控制API调用频率
                    if block['type'] == 'text':
                        time.sleep(1)
                        
                except Exception as e:
                    logging.error(f"处理第 {i + 1} 个内容块时出错: {str(e)}")
                    raise
            
            # 保存最终结果
            self._save_as_markdown(output_path, translations)
            logging.info(f"翻译完成！结果已保存到: {output_path}")
            
            # 清理进度文件
            if progress_file.exists():
                progress_file.unlink()
            
        except Exception as e:
            logging.error(f"翻译过程中出现错误: {str(e)}")
            raise

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='PDF文档翻译工具')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('--model', default='gpt-4o-mini', help='使用的模型名称')
    parser.add_argument('--cost_limit', type=float, default=10.0, help='成本上限（美元）')
    args = parser.parse_args()
    
    try:
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
            
        # 创建翻译器实例
        translator = ReflectiveTranslator(
            api_key=api_key,
            model_name=args.model,
            temperature=0.7,
            max_tokens_per_chunk=16000,
            overlap_tokens=100,
            cost_limit=args.cost_limit
        )
        
        # 开始翻译
        print(f"\n开始翻译：{args.pdf_path}")
        print(f"使用模型：{args.model}")
        print(f"成本上限：${args.cost_limit}\n")
        
        translator.translate_file(
            file_path=args.pdf_path,
            source_lang="英文",
            target_lang="中文"
        )
        
        print("\n翻译完成！输出文件保存在同目录下。")
        
    except ValueError as e:
        print(f"\n错误：{str(e)}")
    except Exception as e:
        print(f"\n发生错误：{str(e)}")

if __name__ == "__main__":
    main()
