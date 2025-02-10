from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes
from rag import RagChain
from chat import chain as chat_chain
from llm import llm as model
from dotenv import load_dotenv

load_dotenv()

# FastAPI 애플리케이션 객체 초기화
app = FastAPI()

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

@app.post("/rag")
async def run_rag_chain(request: RagRequest):
    file_path = request.file_path
    question = request.question
    try:
        rag_chain = RagChain(file_path=file_path)
        result = rag_chain.get_answer(question)
        return JSONResponse(content={"message": "RAG chain is set up successfully", "result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

########### 대화형 인터페이스 ###########

class InputChat(BaseModel):
    """채팅 입력을 위한 기본 모델 정의"""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )
    
@app.post("/chat")
async def chat(input: InputChat):
    try:
        # chat_chain.invoke() 메서드를 사용하기 전에 chain 타입과 입력이 올바른지 확인
        result = chat_chain.invoke(input.messages)
        return JSONResponse(content={"message": "Chat response", "result": result})
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
