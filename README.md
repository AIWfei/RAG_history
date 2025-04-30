# RAG_history


##  工作流程
[文档加载] --> [文本分割] --> [向量化处理] --> [向量存储] --> [问题检索] --> [答案生成]
## 1. 文档加载
**使用工具**：LangChain文档加载器  
```python
from langchain.document_loaders import (
    UnstructuredExcelLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader
)
# 根据文件类型选择加载器
loaders = {
    "txt": TextLoader,
    "docx": Docx2txtLoader,
    "pdf": PyPDFLoader,
    "xlsx": UnstructuredExcelLoader
}

# 示例PDF加载
loader = PyPDFLoader("docs/历史教材.pdf")
text = loader.load()
```
## 2. 文本分割
**使用工具**：LangChain文本分割器 
```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=150,    # 文本块长度
    chunk_overlap=20   # 块间重叠字符
)
split_texts = text_splitter.split_documents(text)
```
## 3. 向量化处理
**使用工具**：SentenceTransformers
```python
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 编码文本列表，返回嵌套列表形式的向量
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        # 编码单个文本，返回列表形式的向量
        return self.model.encode(text).tolist()

# 初始化向量模型
self.embeddings = SentenceTransformerEmbeddings("paraphrase-multilingual-MiniLM-L12-v2")
```
## 4. 向量存储
**使用工具**：Chroma向量数据库 
```python
from langchain.vectorstores import Chroma

def embeddingAndVectorDB(self,texts):
        # db = Chroma.from_documents(texts, self.embeddings, collection_name="example_collection", persist_directory="chroma_db_history")
        db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,  # 传入 Embeddings 对象
            collection_name="example_collection",
            persist_directory="chroma_db_history_1"
        )
        # 持久化数据库
        db.persist()
        return db
```
## 5. 问题检索
**使用工具**：Chroma向量数据库 
```python
def askAndFindFiles(self,question):
        return self.db.similarity_search(query=question, k=3)  # 返回最相关的3个文档
```
## 6. 答案生成
**使用工具**：LLM大语言模型（这里使用deepseek） 
```python
#用自然语言和文档聊天
    def chatWithDoc(self,question):
        _content = ""
        context = self.askAndFindFiles(question)
        # print("检索的内容是" + str(context))
        for i in context:
            _content += i.page_content

        messages = self.prompt.format_messages(context=_content,question=question)
        # print(messages)
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0,
            api_key=dsKey,
            base_url=dsUrl
            )
        return llm.invoke(messages)
```
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
