from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM 모델 초기화
llm = ChatOllama(model="llama3.1-instruct-8b:latest")

# ChatPromptTemplate 생성 (리스트 형태로 수정)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Snail, an AI assistant that helps foreigners settle, travel, and live in Korea. "
            "Your default language is Korean, and you should respond in Korean unless the user speaks another language—then reply in that language. "
            "Your main goal is to introduce Korean culture and provide helpful information about living in Korea."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 체인 생성
chain = prompt | llm | StrOutputParser()
