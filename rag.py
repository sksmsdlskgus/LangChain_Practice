import os
from langchain_chroma import Chroma  
from langchain.schema import Document
import uuid 
import logging
import datetime
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from chat import llm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chroma 벡터 DB 설정
CHROMA_DB_DIR = "vectorstore"
PDF_DIR = "pdfs"
embeddings = OllamaEmbeddings(model="llama3.1-instruct-8b:latest")
    

# 벡터 DB 초기화 함수 (한 번만 실행)
def get_vector_db():
    global vector_db
    if vector_db is None:
        vector_db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
        logger.info("벡터 DB 연결 완료")
    return vector_db

# 글로벌 변수로 벡터 DB 객체 선언 (싱글턴 패턴으로 관리)
vector_db = None

# 벡터 DB를 가져오고 retriever 및 qa_chain 설정
vector_db = get_vector_db()
retriever = vector_db.as_retriever()

# 검색 기반 QA 체인 (벡터 DB 활용)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="map_reduce", 
    retriever=retriever,
    return_source_documents=True
)

# 벡터 DB 메시지 저장 함수
def save_to_vector_db(messages, document_type, conversation_id, vector_db):
    try:
        # 벡터 DB 객체 가져오기 (get_vector_db 호출)
        vector_db = get_vector_db()
        
        # 각 메시지별로 벡터화 및 저장
        for message in messages:
            vectors = embeddings.embed_documents([message])

            # 고유 ID 생성 (UUID 사용)
            doc_id = str(uuid.uuid4())  # UUID를 사용하여 고유 ID 생성

            # 메타데이터 추가
            metadata = {
                "id": doc_id,
                "type": document_type,  # 'message', 'pdf', 'web'
                "source": "user_input",
                "conversation_id": conversation_id,  # 대화 ID 추가
                "timestamp": datetime.datetime.now().isoformat()
            }

            # Document 객체 생성
            document = Document(page_content=message, metadata=metadata)

            # 벡터 DB에 문서 추가
            vector_db.add_documents(
                documents=[document],  # Document 객체 전달
                embeddings=vectors,
                ids=[doc_id]  # 고유 ID 전달
            )
            
            logger.info(f"Message saved: {doc_id}, Message: {message}") # 실행된 고유id와 메시지 출력
            
            documents = vector_db.get()
            logger.info(f"전체 조회: {documents}") # 벡터 db에 저장된 전체 정보 출력

        logger.info("벡터 DB에 메시지 저장 성공")
    except Exception as e:
        logger.error(f"벡터 DB 저장 중 오류 발생: {e}")


####################################### PDF 파일에서 텍스트 추출 

def extract_text_from_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)  # PyPDFLoader로 PDF 로드
        pages = loader.load()  # 페이지 단위로 로드된 문서
        text = ""
        for page in pages:
            text += page.page_content  # 각 페이지의 내용 합치기
    except Exception as e:
        logger.error(f"PDF 텍스트 추출 실패: {pdf_path}, 오류: {e}")
        text = ""
    return text

# 엑셀 파일에서 텍스트 추출 (모든 시트 포함)
def extract_text_from_excel(excel_path):
    text = ""
    try:
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)  # 모든 시트를 문자열로 변환
            text += df.to_string(index=False, header=False) + "\n"
    except Exception as e:
        logger.error(f"엑셀 텍스트 추출 실패: {excel_path}, 오류: {e}")
    return text

# PDF/Excel 파일을 벡터 DB에 저장 
def save_files_to_vector_db():
    try:
        vector_db = get_vector_db()  # 벡터 DB 인스턴스 가져오기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)  # 청크 분할 설정

        for file_name in os.listdir(PDF_DIR):
            file_path = os.path.join(PDF_DIR, file_name)
            text = ""
            document_type = None  # 문서 타입 초기화

            # 파일 확장자 확인 후 텍스트 추출 및 문서 타입 설정
            if file_name.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
                document_type = "pdf"
            elif file_name.endswith(".xls") or file_name.endswith(".xlsx"):
                text = extract_text_from_excel(file_path)
                document_type = "excel"
            else:
                logger.warning(f"지원하지 않는 파일 형식: {file_name}")
                continue

            # 파일에 타입이 설정되지 않으면 skip
            if not document_type:
                logger.warning(f"파일 타입을 알 수 없습니다: {file_name}")
                continue

            # 청크 단위로 분할
            text_chunks = text_splitter.split_text(text)

            # 각 청크를 벡터화 및 저장
            for chunk in text_chunks:
                vectors = embeddings.embed_documents([chunk])  # 임베딩 생성
                doc_id = str(uuid.uuid4())  # 고유 ID 생성

                metadata = {
                    "id": doc_id,
                    "type": document_type,  # 'pdf', 'excel'
                    "source": file_name,  # 파일명 저장
                    "timestamp": datetime.datetime.now().isoformat()
                }

                document = Document(page_content=chunk, metadata=metadata)

                # 벡터 DB에 저장
                vector_db.add_documents(documents=[document], embeddings=vectors, ids=[doc_id])

                logger.info(f"Document saved: {doc_id}, Source: {file_name}")
                
                # documents = vector_db.get()
                # logger.info(f"전체 조회: {documents}")  # 벡터 db에 저장된 전체 정보 출력

        logger.info("벡터 DB에 문서 저장 완료")
    except Exception as e:
        logger.error(f"벡터 DB 저장 중 오류 발생: {e}") 