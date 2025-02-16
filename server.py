from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from langserve import add_routes
from chat import chain as chat_chain
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("서버가 시작되었습니다!")

# 환경 설정 파일 로딩
load_dotenv()

# FastAPI 애플리케이션 객체 초기화
app = FastAPI()

# Tesseract OCR 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chroma 벡터 DB 설정
CHROMA_DB_DIR = "vectorstore"
embeddings = OllamaEmbeddings(model="llama3.1-instruct-8b:latest")

# 기존 DB 디렉토리 삭제
if os.path.exists(CHROMA_DB_DIR):
    import shutil
    shutil.rmtree(CHROMA_DB_DIR)  # 디렉토리 및 그 안의 내용 모두 삭제
    
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# 벡터 DB 저장 함수
def save_to_vector_db(messages, document_type):
    try:
        # Chroma DB 클라이언트 연결 (한 번만 생성)
        vector_db = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
        
        # 메시지에 고유 ID 추가 및 벡터화 처리
        combined_text = " ".join(messages)
        vectors = embeddings.embed_documents([combined_text])

        # 고유 ID 생성 (UUID 사용)
        doc_id = str(uuid.uuid4())  # UUID를 사용하여 고유 ID 생성
        
        # 메타데이터 추가 (문서 타입과 출처 등)
        metadata = {
            "id": doc_id,
            "type": document_type,  # 'message', 'pdf', 'web'
            "source": "user_input"  # 예시로, source를 사용자 입력으로 설정
        }
        
        # Document 객체 생성
        document = Document(page_content=combined_text, metadata=metadata)
        
        # 벡터 DB에 문서 추가
        vector_db.add_documents(
            documents=[document],  # Document 객체 전달
            embeddings=vectors,
            ids=[doc_id]  # 고유 ID 전달
        )
        
        logger.info("벡터 DB에 메시지 저장 성공")
    except Exception as e:
        logger.error(f"벡터 DB 저장 중 오류 발생: {e}")

# 입력 데이터 모델
class InputChat(BaseModel):
    messages: list[str]

# 대화형 API 엔드포인트
@app.post("/chat")
async def chat(input: str = Form(...), file: Optional[UploadFile] = File(None), document_type: str = "message"):
    try:
        # JSON 문자열을 InputChat 모델로 파싱
        input_data = InputChat(**json.loads(input))
        result = {}

        # 이미지가 업로드된 경우 OCR 수행
        if file:
            image = Image.open(io.BytesIO(await file.read()))
            ocr_text = pytesseract.image_to_string(image, lang="kor+eng")
            input_data.messages.append(ocr_text)
            result["ocr_text"] = ocr_text  # OCR 텍스트 결과 추가
        
        # 챗봇 응답 생성
        result["chatbot_response"] = chat_chain.invoke(input_data.messages)
        
        # 메시지와 OCR 텍스트를 벡터 DB에 저장
        save_to_vector_db(input_data.messages, document_type)  # document_type을 함수에 전달
        
        # 최종 응답 반환
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
