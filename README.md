# Reflective Translator

一个智能的 PDF 文档翻译工具，支持多语言互译，具有高质量的翻译输出和智能的文档结构保持能力。

## 特性

- 支持多种语言互译（中文、英文、日文、韩文、法文等13种语言）
- 默认支持英文到中文的翻译（无需额外参数）
- 智能保持文档格式和结构
- 支持数学公式、代码片段的原样保留
- 自动分块处理长文本
- 具有翻译进度保存和恢复功能
- 支持多种文档格式（PDF、Markdown、TXT、RST）

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/whotto/PDF_reflective_translator.git
cd PDF_translation_workflow
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
   - 复制 `.env.example` 为 `.env`
   - 在 `.env` 文件中设置你的 API key：
```
OPENAI_API_KEY=你的API密钥
```

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

## 配置说明

- `model`: 使用的模型，默认为 'gpt-4o-mini'
- `cost_limit`: 成本上限（美元），默认为 10.0
- `source`: 源语言，默认为 '英文'
- `target`: 目标语言，默认为 '中文'

## 注意事项

1. 对于文件名包含空格的PDF，请使用引号：
```bash
python reflective_translator.py "What Is ChatGPT Doing.pdf"
```

2. 翻译结果会保存在与源文件相同目录下，格式为 Markdown

3. 翻译进度会自动保存，意外中断后可以继续之前的翻译

## 开发说明

- 使用 Python 3.8 或更高版本
- 主要依赖：PyPDF2、openai、python-dotenv
- 代码遵循 PEP 8 规范

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request

## 联系方式

- 博客：[阅读网](https://yuedu.biz)
- Email：grow8org@gmail.com
- GitHub：[PDF_reflective_translator](https://github.com/whotto/PDF_reflective_translator)
