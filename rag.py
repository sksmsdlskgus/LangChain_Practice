# import os
# import logging
# from typing import Optional
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.prompts import load_prompt
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama

# # Chroma DB Directory
# CHROMA_DB_DIR = "vectorstore"

# class RagChain:
#     def __init__(self, model: str = "llama3.1-instruct-8b:latest", temperature: float = 0.0, system_prompt: Optional[str] = None, **kwargs):
#         self.model = model
#         self.temperature = temperature
#         self.system_prompt = system_prompt or (
#             "당신은 Snail입니다. Snail은 외국인들이 한국에 정착하고, 여행하고, 생활할 수 있도록 돕는 친절한 AI 챗봇입니다. "
#             "당신의 주요 목표는 한국 문화를 소개하고, 한국에서 생활하는 데 필요한 유용한 정보를 제공하는 것입니다. "
#             "사용자가 궁금해하는 사항에 대해 **세심하고 친절하며 따뜻한** 답변을 드리고, 필요한 정보를 단계적으로 자세히 설명하는 것을 중요하게 생각합니다. "
#             "모든 대답은 한국어로만 해야 하며, 지나치게 긴 인사말은 생략하고, 사용자가 다른 언어로 질문을 하면 그 언어로 대답합니다. "
#             "모든 답변은 무조건 **존댓말**로 해야 하며, 항상 사용자가 편안하게 느낄 수 있도록 배려심을 담아 답변해 주세요."
#         )
#         self.file_path = kwargs.get("file_path")

#     def setup_chain(self):
#         if not self.file_path:
#             raise ValueError("file_path is required")

#         # PDF 문서 로딩
#         loader = PyPDFLoader(self.file_path)
#         documents = loader.load_and_split()

#         if not documents:
#             raise ValueError("No content loaded from the provided PDF.")

#         # 텍스트 분할 (임계값 150자, 중첩 30자)
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
#         docs = text_splitter.split_documents(documents)

#         if not docs:
#             raise ValueError("No documents split after text splitting.")

#         # Ollama 임베딩 설정
#         embeddings = OllamaEmbeddings(model=self.model)

#         # Chroma 벡터 DB 생성
#         os.makedirs(CHROMA_DB_DIR, exist_ok=True)
#         vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)
#         vectorstore.persist()

#         # 벡터 DB 검색기 생성
#         retriever = vectorstore.as_retriever()

#         # 프롬프트 로드
#         prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")

#         # Ollama 모델 설정
#         llm = ChatOllama(model=self.model, temperature=self.temperature)

#         # LLM 체인 생성
#         chain = (prompt | llm | StrOutputParser())

#         return retriever, chain

#     def get_answer(self, question: str):
#         # 체인 및 검색기 설정
#         retriever, chain = self.setup_chain()

#         # 쿼리로 검색한 결과
#         search_results = retriever.retrieve(question)

#         if not search_results:
#             return "관련된 정보를 찾을 수 없습니다."

#         # 검색된 결과가 있으면 첫 번째 항목으로 응답 생성
#         context = search_results[0]['content']

#         # 템플릿에 질문과 문맥 전달
#         final_answer = chain.run(input_variables={"question": question, "context": context})

#         return final_answer

# def initialize_directories():
#     """Chroma DB 디렉토리 초기화"""
#     os.makedirs(CHROMA_DB_DIR, exist_ok=True)
#     logging.info(f"Chroma DB directory initialized: {CHROMA_DB_DIR}")

# def store_files_in_chroma():
#     """PDF 파일을 벡터 스토어에 저장"""
#     documents = []

#     # PDF 파일 처리
#     for file in os.listdir("data"):  # 'data' 디렉토리가 맞는지 확인
#         if file.endswith(".pdf"):
#             pdf_path = os.path.join("data", file)
#             try:
#                 loader = PyPDFLoader(pdf_path)
#                 documents.extend(loader.load_and_split())
#             except Exception as e:
#                 logging.error(f"Error loading PDF {file}: {e}")
#                 continue

#     # 문서 분할(chunking) 및 벡터화
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
#     split_docs = text_splitter.split_documents(documents)

#     # 벡터 스토어에 chunked 문서 저장
#     if split_docs:
#         vectorstore = Chroma.from_documents(
#             split_docs, OllamaEmbeddings(model="llama3.1-instruct-8b:latest"), persist_directory=CHROMA_DB_DIR
#         )
#         logging.info("Files successfully stored in Chroma DB.")
#         return vectorstore
#     else:
#         logging.error("No documents to store in Chroma DB.")
#         return None

# # `initialize_directories()`를 적절한 위치에서 호출해 디렉토리를 초기화합니다.
# initialize_directories()

# # `store_files_in_chroma()`에서 벡터 스토어를 반환하고 이를 활용합니다.
# vectorstore = store_files_in_chroma()
# if vectorstore:
#     print("Vectorstore created and persisted.")
# else:
#     print("Error: No files were processed.")
