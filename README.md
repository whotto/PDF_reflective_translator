# Reflective Translator

一个智能的 PDF 文档翻译工具，支持多语言互译，具有高质量的翻译输出和智能的文档结构保持能力。

作者：玄清

## 版本号

v1.1.0

## 应用场景

- 学术论文翻译：将英文学术论文翻译成中文，保持专业术语和格式
- 技术文档本地化：支持多种语言的技术文档翻译
- 书籍翻译：支持长文本的分块翻译，自动保存进度
- 多语言文档处理：支持13种主流语言之间的互译

## 创作流程图概览

```
PDF文档输入 -> 文本提取 -> 智能分块 -> 
翻译处理(OpenAI) -> 格式保持 -> Markdown输出
```

## 使用工具及链接

- Python 3.8+：主要开发语言
- OpenAI API：提供翻译服务
- PyMuPDF (fitz)：PDF文件处理
- python-dotenv：环境变量管理
- GitHub：代码托管平台

## 特性

- 支持多种语言互译（中文、英文、日文、韩文、法文等13种语言）
- 默认支持英文到中文的翻译（无需额外参数）
- 智能保持文档格式和结构
- 支持数学公式、代码片段的原样保留
- 自动分块处理长文本
- 具有翻译进度保存和恢复功能
- 支持多种文档格式（PDF、Markdown、TXT、RST）

## 系统要求

- Python 3.8 或更高版本
- 支持 Windows、macOS 和 Linux 系统
- 需要网络连接以访问翻译API

## 依赖说明

核心依赖：

- openai (0.28.0)：用于API调用，使用稳定版本确保兼容性
- PyMuPDF (1.23.8)：用于PDF文档处理
- PyYAML (6.0.1)：用于配置文件处理
- python-dotenv (1.0.0)：用于环境变量管理
- tiktoken (0.5.1)：用于token计数

工具库：

- requests (2.31.0)：用于HTTP请求
- tenacity (8.2.3)：用于API重试机制
- tqdm (4.66.1)：用于显示进度条

所有依赖都是跨平台兼容的，无需额外的平台特定依赖。

## 安装指南

1. 确保已安装Python 3.8+：

   ```bash
   python --version
   ```

2. 创建并激活虚拟环境（推荐）：

   Windows:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

4. 配置API密钥：

   - 复制 `.env.example` 为 `.env`

   - 在 `.env` 文件中设置你的API密钥：

     ```
     OPENAI_API_KEY=你的API密钥
     ```

## 快速开始

1. 翻译PDF文件（默认英文到中文）：

   ```bash
   python reflective_translator.py "document.pdf"
   ```

2. 翻译其他格式文件：

   ```bash
   python reflective_translator.py "document.txt"  # 文本文件
   python reflective_translator.py "document.md"   # Markdown文件
   ```

3. 指定翻译方向：

   ```bash
   python reflective_translator.py "document.pdf" --source 中文 --target 英文
   ```

## 输出说明

- 所有翻译结果都会以Markdown格式保存
- 输出文件命名格式：`原文件名_目标语言.md`
- 图片会被保存在 `images` 目录下
- 翻译日志保存在 `translation.log` 文件中

## 使用方法

### 基本用法（英文到中文）

```bash
python reflective_translator.py "你的PDF文件.pdf"
```

### 指定语言方向

```bash
python reflective_translator.py "你的PDF文件.pdf" --source 中文 --target 英文
```

### 其他参数

```bash
python reflective_translator.py "你的PDF文件.pdf" --model gpt-4o-mini --cost_limit 10.0
```

### 支持的语言

- 中文
- 英文
- 日文
- 韩文
- 法文
- 德文
- 西班牙文
- 俄文
- 阿拉伯文
- 葡萄牙文
- 意大利文
- 越南文
- 泰文

## 使用方法

基本用法：

```bash
python reflective_translator.py <input_file>
```

指定源语言和目标语言：

```bash
python reflective_translator.py --source_lang en --target_lang zh <input_file>  # 英文翻译为中文
python reflective_translator.py --source_lang zh --target_lang en <input_file>  # 中文翻译为英文
python reflective_translator.py --source_lang ja --target_lang zh <input_file>  # 日文翻译为中文
```

支持的语言代码：

- zh: 中文
- en: 英文
- ja: 日文
- ko: 韩文
- fr: 法文
- de: 德文
- es: 西班牙文
- it: 意大利文
- ru: 俄文
- pt: 葡萄牙文
- nl: 荷兰文
- ar: 阿拉伯文
- hi: 印地文

例如：

```bash
# 英文 PDF 翻译为中文
python reflective_translator.py document.pdf

# 中文文档翻译为英文
python reflective_translator.py --source_lang zh --target_lang en chinese_doc.txt

# 日文文档翻译为中文
python reflective_translator.py --source_lang ja --target_lang zh japanese_doc.md
```

## 文件格式说明

### 支持的输入格式

- PDF (*.pdf)：支持文本型 PDF，包括学术论文、技术文档等
- Markdown (*.md)：支持标准 Markdown 文档
- 文本 (*.txt)：支持纯文本文件
- reStructuredText (*.rst)：支持 RST 格式文档

### 输出格式

输出文件将保存为 Markdown 格式（*.md），具有以下特点：

- 保持原文档的标题层级结构
- 保留数学公式和代码块格式
- 图片引用保持不变
- 表格结构完整呈现
- 文件名格式：原文件名_中文.md（例如：example.pdf -> example_中文.md）

## 配置说明

- `model`: 使用的模型，默认为 'gpt-4o-mini'
- `cost_limit`: 成本上限（美元），默认为 10.0
- `source`: 源语言，默认为 '英文'
- `target`: 目标语言，默认为 '中文'

## 使用指南

> 重要提醒：为了安全和降低费用，将 OpenAI API base_url [https://api.openai.com](https://api.openai.com/) 替换为 [https://api.gptsapi.net](https://api.gptsapi.net)。费用低一半。

[注册链接](https://bewildcard.com/i/WHVIP)

## 注意事项

1. API 使用：
   - 默认使用 GPTs API (https://api.gptsapi.net/v1)
   - 相比官方 API 可节省 50% 的费用
   - 👉 [点击这里注册](https://bewildcard.com/i/WHVIP)
   - 请确保 API 密钥配置正确

2. 对于文件名包含空格的PDF，请使用引号：

```bash
python reflective_translator.py "What Is ChatGPT Doing.pdf"
```

3. 翻译结果会保存在与源文件相同目录下，格式为 Markdown

4. 翻译进度会自动保存，意外中断后可以继续之前的翻译

## 开发说明

- 使用 Python 3.8 或更高版本
- 主要依赖：PyPDF2、openai、python-dotenv
- 代码遵循 PEP 8 规范

## 未来计划

1. 支持更多文档格式（Word、Excel等）
2. 添加批量处理功能
3. 优化翻译质量和速度
4. 增加自定义词典功能
5. 添加GUI界面

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request

## 联系方式

- 博客：[天天悦读](https://yuedu.biz/)
- Email：[grow8org@gmail.com](mailto:grow8org@gmail.com)
- GitHub：[PDF_reflective_translator](https://github.com/whotto/PDF_reflective_translator)

## 最新更新

### v1.0.1

- 支持多种文件格式输入
- 统一 Markdown 输出
- 优化 API 调用逻辑
- 改进错误处理机制
- 添加详细日志记录
