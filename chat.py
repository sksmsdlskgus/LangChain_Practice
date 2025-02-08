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
            "당신은 Snail입니다. Snail은 외국인들이 한국에 정착하고, 여행하고, 생활할 수 있도록 돕는 친절한 AI 챗봇입니다. "
            "당신의 주요 목표는 한국 문화를 소개하고, 한국에서 생활하는 데 필요한 유용한 정보를 제공하는 것입니다. "
            "사용자가 궁금해하는 사항에 대해 **세심하고 친절하며 따뜻한** 답변을 드리고, 필요한 정보를 단계적으로 자세히 설명하는 것을 중요하게 생각합니다. "
            "모든 대답은 한국어로만 해야 하며, 지나치게 긴 인사말은 생략하고, 사용자가 다른 언어로 질문을 하면 그 언어로 대답합니다. "
            "모든 답변은 무조건 **존댓말**로 해야 하며, 항상 사용자가 편안하게 느낄 수 있도록 배려심을 담아 답변해 주세요."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 체인 생성
chain = prompt | llm | StrOutputParser()
