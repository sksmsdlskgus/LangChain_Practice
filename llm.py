from langchain_ollama import ChatOllama

# LLM 모델 초기화
llm = ChatOllama(model="llama3.1-instruct-8b:latest")

# 모델을 통해 응답을 받는 함수
def get_response(user_message: str):
    response = llm.generate([user_message])
    return response
