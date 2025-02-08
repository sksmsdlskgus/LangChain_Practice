from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PDFPlumberLoader

from base import BaseChain

# 문서 포맷팅
def format_docs(docs):
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content><page>{doc.metadata['page']}</page><source>{doc.metadata['source']}</source></document>"
        for doc in docs
    )

# RAG 체인 설정
class RagChain(BaseChain):

    def __init__(self, model: str = "llama3.1-instruct-8b:latest", temperature: float = 0.3, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or (
            "You are Snail, an AI assistant that helps foreigners settle, travel, and live in Korea. aYour default language is Korean, and you should respond in Korean unless the user speaks another language—then reply in that language. Your main goal is to introduce Korean culture and provide helpful information about living in Korea."
        )
        self.file_path = kwargs.get("file_path")

    def setup(self):
        if not self.file_path:
            raise ValueError("file_path is required")

        # 문서 로드
        loader = PDFPlumberLoader(self.file_path)

        # Splitter 설정 (여기에서 chunk_size와 chunk_overlap을 조정)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

        # 문서 분할
        docs = loader.load_and_split(text_splitter=text_splitter)

        # 임베딩 설정 (Ollama 모델 사용)
        embeddings = OllamaEmbeddings(model=self.model)

        # Chroma 벡터 DB 저장
        vectorstore = Chroma.from_documents(docs, embedding=embeddings)

        # 문서 검색기 설정
        retriever = vectorstore.as_retriever()

        # 프롬프트 로드
        prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")

        # Ollama 모델 지정
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        # 체인 생성
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
