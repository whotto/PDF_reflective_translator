# PDF Translation Workflow

一个强大的PDF文档翻译工具，支持将PDF文档从一种语言翻译成另一种语言，保持原文格式并生成高质量的译文。

## 特性

- 智能PDF文本提取，保持原文格式和结构
- 专业的翻译质量，支持学术和技术文献
- 保留数学公式、代码片段等专业内容
- 自动识别和保护专有名词
- 支持翻译进度保存和恢复
- 灵活的配置选项

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd pdf-translation-workflow
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并添加你的API密钥：
```
OPENAI_API_KEY=your-api-key-here
```

## 使用方法

基本用法：
```bash
python reflective_translator.py "your-pdf-file.pdf"
```

### 命令行参数

- `pdf_path`: PDF文件路径（必需）
- `--model`: 使用的模型名称（默认：gpt-4o-mini）
- `--cost_limit`: 成本上限（美元）（默认：10.0）

### 示例

翻译PDF文档：
```bash
python reflective_translator.py "document.pdf"
```

## 项目结构

```
.
├── .env                    # 环境变量配置
├── README.md              # 项目文档
├── reflective_translator.py # 主程序
├── requirements.txt       # 项目依赖
├── logs/                 # 日志目录
└── translation_progress/ # 翻译进度保存目录
```

## 配置选项

- 模型配置：
  * 模型名称：gpt-4o-mini（默认）
  * 温度参数：0.7
  * 最大令牌数：16000

- 翻译设置：
  * 源语言：英文（默认）
  * 目标语言：中文（默认）
  * 成本上限：$10.0（默认）

## 输出格式

翻译结果将保存为Markdown格式，包含：
- 保持原文的标题层级
- 保留数学公式
- 保持图片引用
- 清晰的段落结构

## 注意事项

1. 确保API密钥配置正确
2. 大文件翻译可能需要较长时间
3. 翻译过程中可以随时中断，进度会自动保存
4. 建议定期检查成本使用情况

## 依赖项

- PyPDF2：PDF解析
- openai：API调用
- python-dotenv：环境配置
- requests：HTTP请求
- tiktoken：Token计数

## 许可证

[选择合适的许可证]

## 贡献

欢迎提交问题和改进建议！
