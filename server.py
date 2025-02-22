from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from langserve import add_routes
from chat import chain as chat_chain
from chat import llm
from dotenv import load_dotenv
from typing import Optional
from PIL import Image
import json
import pytesseract
import io
import os
from langchain_chroma import Chroma  # 최신 패키지로 임포트
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import uuid 
import logging
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from data import fetch_data_prec,fetch_data_law,fetch_data_ordin


# 환경 설정 파일 로딩
load_dotenv()

# FastAPI 애플리케이션 객체 초기화
app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("서버가 시작되었습니다!")

# 현재 시간 가져오기
timestamp = datetime.datetime.now().isoformat()  # ISO 형식으로 날짜 및 시간 반환

# Tesseract OCR 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chroma 벡터 DB 설정
CHROMA_DB_DIR = "vectorstore"
PDF_DIR = "pdfs"
embeddings = OllamaEmbeddings(model="llama3.1-instruct-8b:latest")

# 기존 DB 디렉토리 삭제
#if os.path.exists(CHROMA_DB_DIR):
#    import shutil
#    shutil.rmtree(CHROMA_DB_DIR)  # 디렉토리 및 그 안의 내용 모두 삭제
    
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

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
        

##################################### 30분 마다 스케줄러 생성 
scheduler = BackgroundScheduler()

def fetch_all_data():
    fetch_data_prec()
    fetch_data_law()
    fetch_data_ordin()
    
     # fetch_all_data가 완료된 후에 save_files_to_vector_db 호출
    save_files_to_vector_db()

# 국가법령정보 엑셀 업데이트
scheduler.add_job(fetch_all_data, trigger='interval', hours=24)


# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포시 도메인
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 기본 경로("/")에 대한 리다이렉션 처리
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/prompt/playground")

@app.get("/prompt/playground")
async def playground():
    return {"message": "Welcome to the Playground!"}

########### 대화형 인터페이스 ###########
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

# 입력 데이터 모델
class InputChat(BaseModel):
    messages: list[str]

# 대화형 API 엔드포인트
@app.post("/chat")
async def chat(input: str = Form(...), file: Optional[UploadFile] = File(None),  document_type: str = "message", vector_db: Chroma = Depends(get_vector_db)):
    # vector_db는 이제 Chroma 객체로 자동 주입됨
    try:
        input_data = InputChat(**json.loads(input))
        result = {}

        # 대화 ID 생성
        conversation_id = str(uuid.uuid4())

        # 이미지가 업로드된 경우 OCR 처리
        if file:
            image = Image.open(io.BytesIO(await file.read()))
            ocr_text = pytesseract.image_to_string(image, lang="kor+eng")
            input_data.messages.append(ocr_text)
            result["ocr_text"] = ocr_text
            
        # # 벡터 DB에서 관련 문서 검색
        # query = input_data.messages[-1]  # 최신 메시지 사용
        # docs = retriever.invoke(query)

        # # 검색된 문서를 LLM 입력에 추가
        # context = "\n\n".join([doc.page_content for doc in docs]) if docs else "관련 정보 없음"
        # input_data.messages.append(f"🔍 참고 정보:\n{context}")

        # # 챗봇 응답 생성
        # result["chatbot_response"] = chat_chain.invoke(input_data.messages)    
        
        # 챗봇 응답만 생성 프롬프팅 Test
        result["chatbot_response"] = chat_chain.invoke(input_data.messages)

        # 벡터 DB에 메시지 저장
        save_to_vector_db(input_data.messages, document_type, conversation_id, vector_db)

        return JSONResponse(content={"message": "Chat response", "type": document_type, "result": result, "input_messages": input_data.messages})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# 대화형 채팅 엔드포인트 설정
add_routes(
    app,
    chat_chain.with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

# 서버 실행 설정
if __name__ == "__main__":
    scheduler.start()

    # 데이터 가져오는 함수 실행
    # fetch_all_data()

    # FastAPI 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
