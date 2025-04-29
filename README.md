# RAG_history


##  工作流程
[文档加载] --> [文本分割] --> [向量化处理] --> [向量存储] --> [问题检索] --> [答案生成]

##  快速使用
```bash
# 安装依赖
pip install langchain chromadb pypdf docx2txt sentence-transformers
```
```bash
from chatdoc import ChatDoc
# 初始化系统
assistant = ChatDoc()
assistant.doc = "history.docx"  # 设置文档路径

# 处理文档
assistant.splitSentences()      # 执行分块和向量化

# 智能问答
question = "请设计关于工业革命的2道选择题"
response = assistant.chatWithDoc(question)
print(response.content)
