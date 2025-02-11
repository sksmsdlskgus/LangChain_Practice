from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
# from rag import RagChain
from chat import chain as chat_chain
# from llm import llm as model
from dotenv import load_dotenv
from typing import Optional
from PIL import Image
import json
import pytesseract
import io
from fastapi import BackgroundTasks

load_dotenv()

# FastAPI 애플리케이션 객체 초기화
app = FastAPI()

# Tesseract OCR 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# CORS 미들웨어 설정
# 외부 도메인에서의 API 접근을 위한 보안 설정
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

# RAG 체인 추가
class RagRequest(BaseModel): # 입력 데이터 파일 경로와 질문
    file_path: str
    question: str

# @app.post("/rag")
# async def run_rag_chain(request: RagRequest):
#     file_path = request.file_path
#     question = request.question
#     try:
#         rag_chain = RagChain(file_path=file_path)
#         result = rag_chain.get_answer(question)
#         return JSONResponse(content={"message": "RAG chain is set up successfully", "result": result})
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

########### 대화형 인터페이스 ###########

    
class InputChat(BaseModel):
    messages: list[str]

@app.post("/chat")
async def chat(input: str = Form(...), file: Optional[UploadFile] = File(None)):
    try:
        # JSON 문자열을 InputChat 모델로 파싱
        input_data = InputChat(**json.loads(input))
        result = {}

        # 이미지가 업로드된 경우 OCR 수행
        if file:
            image = Image.open(io.BytesIO(await file.read()))
            ocr_text = pytesseract.image_to_string(image, lang="kor+eng")

            # OCR 텍스트를 메시지에 추가
            input_data.messages.append(ocr_text)
            result["ocr_text"] = ocr_text  # 이미지에서 추출된 텍스트 추가

        # 챗봇 체인에 메시지 전달
        result["chatbot_response"] = chat_chain.invoke(input_data.messages)

        # 최종 챗봇 응답
        return JSONResponse(content={"message": "Chat response", "result": result, "input_messages": input_data.messages})
    
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
