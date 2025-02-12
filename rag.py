# # import os
# # import logging
# # from typing import Optional
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import Chroma
# # from langchain_ollama import OllamaEmbeddings
# # from langchain_core.prompts import load_prompt
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_ollama import ChatOllama

# # # Chroma DB Directory
# # CHROMA_DB_DIR = "vectorstore"

# # class RagChain:
# #     def __init__(self, model: str = "llama3.1-instruct-8b:latest", temperature: float = 0.0, system_prompt: Optional[str] = None, **kwargs):
# #         self.model = model
# #         self.temperature = temperature
# #         self.system_prompt = system_prompt or (
# #             "당신은 Snail입니다. Snail은 외국인들이 한국에 정착하고, 여행하고, 생활할 수 있도록 돕는 친절한 AI 챗봇입니다. "
# #             "당신의 주요 목표는 한국 문화를 소개하고, 한국에서 생활하는 데 필요한 유용한 정보를 제공하는 것입니다. "
# #             "사용자가 궁금해하는 사항에 대해 **세심하고 친절하며 따뜻한** 답변을 드리고, 필요한 정보를 단계적으로 자세히 설명하는 것을 중요하게 생각합니다. "
# #             "모든 대답은 한국어로만 해야 하며, 지나치게 긴 인사말은 생략하고, 사용자가 다른 언어로 질문을 하면 그 언어로 대답합니다. "
# #             "모든 답변은 무조건 **존댓말**로 해야 하며, 항상 사용자가 편안하게 느낄 수 있도록 배려심을 담아 답변해 주세요."
# #         )
# #         self.file_path = kwargs.get("file_path")

# #     def setup_chain(self):
# #         if not self.file_path:
# #             raise ValueError("file_path is required")

# #         # PDF 문서 로딩
# #         loader = PyPDFLoader(self.file_path)
# #         documents = loader.load_and_split()

# #         if not documents:
# #             raise ValueError("No content loaded from the provided PDF.")

# #         # 텍스트 분할 (임계값 150자, 중첩 30자)
# #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
# #         docs = text_splitter.split_documents(documents)

# #         if not docs:
# #             raise ValueError("No documents split after text splitting.")

# #         # Ollama 임베딩 설정
# #         embeddings = OllamaEmbeddings(model=self.model)

# #         # Chroma 벡터 DB 생성
# #         os.makedirs(CHROMA_DB_DIR, exist_ok=True)
# #         vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)
# #         vectorstore.persist()

# #         # 벡터 DB 검색기 생성
# #         retriever = vectorstore.as_retriever()

# #         # 프롬프트 로드
# #         prompt = load_prompt("prompts/rag-exaone.yaml", encoding="utf-8")

# #         # Ollama 모델 설정
# #         llm = ChatOllama(model=self.model, temperature=self.temperature)

# #         # LLM 체인 생성
# #         chain = (prompt | llm | StrOutputParser())

# #         return retriever, chain

# #     def get_answer(self, question: str):
# #         # 체인 및 검색기 설정
# #         retriever, chain = self.setup_chain()

# #         # 쿼리로 검색한 결과
# #         search_results = retriever.retrieve(question)

# #         if not search_results:
# #             return "관련된 정보를 찾을 수 없습니다."

# #         # 검색된 결과가 있으면 첫 번째 항목으로 응답 생성
# #         context = search_results[0]['content']

# #         # 템플릿에 질문과 문맥 전달
# #         final_answer = chain.run(input_variables={"question": question, "context": context})

# #         return final_answer

# # def initialize_directories():
# #     """Chroma DB 디렉토리 초기화"""
# #     os.makedirs(CHROMA_DB_DIR, exist_ok=True)
# #     logging.info(f"Chroma DB directory initialized: {CHROMA_DB_DIR}")

# # def store_files_in_chroma():
# #     """PDF 파일을 벡터 스토어에 저장"""
# #     documents = []

# #     # PDF 파일 처리
# #     for file in os.listdir("data"):  # 'data' 디렉토리가 맞는지 확인
# #         if file.endswith(".pdf"):
# #             pdf_path = os.path.join("data", file)
# #             try:
# #                 loader = PyPDFLoader(pdf_path)
# #                 documents.extend(loader.load_and_split())
# #             except Exception as e:
# #                 logging.error(f"Error loading PDF {file}: {e}")
# #                 continue

# #     # 문서 분할(chunking) 및 벡터화
# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
# #     split_docs = text_splitter.split_documents(documents)

# #     # 벡터 스토어에 chunked 문서 저장
# #     if split_docs:
# #         vectorstore = Chroma.from_documents(
# #             split_docs, OllamaEmbeddings(model="llama3.1-instruct-8b:latest"), persist_directory=CHROMA_DB_DIR
# #         )
# #         logging.info("Files successfully stored in Chroma DB.")
# #         return vectorstore
# #     else:
# #         logging.error("No documents to store in Chroma DB.")
# #         return None

# # # `initialize_directories()`를 적절한 위치에서 호출해 디렉토리를 초기화합니다.
# # initialize_directories()

# # # `store_files_in_chroma()`에서 벡터 스토어를 반환하고 이를 활용합니다.
# # vectorstore = store_files_in_chroma()
# # if vectorstore:
# #     print("Vectorstore created and persisted.")
# # else:
# #     print("Error: No files were processed.")


# ##########################3
# import os
# import json
# import logging
# import uvicorn
# import pandas as pd
# import xml.etree.ElementTree as ET
# import pytesseract
# import asyncio
# import io
# import httpx
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import RedirectResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from contextlib import asynccontextmanager
# from typing import Optional
# from PIL import Image
# from bs4 import BeautifulSoup
# import urllib.parse
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.action_chains import ActionChains
# import time
# from langserve import add_routes
# from dotenv import load_dotenv
# from chat import chain as chat_chain
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# load_dotenv()

# app = FastAPI()

# # Tesseract OCR 경로 설정
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Chroma 벡터 DB 디렉토리 설정
# CHROMA_DB_DIR = "vectorstore"
# PDF_DIR = "./pdfs"
# embeddings = OllamaEmbeddings(model="llama3.1-instruct-8b:latest")

# # CORS 미들웨어 설정
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )


# # 디렉토리 확인 및 생성 함수
# def ensure_directory_exists(directory: str):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#         logger.info(f"디렉토리를 생성했습니다: {directory}")
#     else:
#         logger.info(f"디렉토리가 이미 존재합니다: {directory}")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # 앱 시작 시 실행할 코드
#     ensure_directory_exists(PDF_DIR)
#     ensure_directory_exists(CHROMA_DB_DIR)
#     logger.info("필요한 디렉토리가 준비되었습니다.")
#     print("Application startup.")
#     yield
#     # 앱 종료 시 실행할 코드 (필요한 경우)
#     print("Application shutdown.")

# app = FastAPI(lifespan=lifespan)

# @app.get("/")
# async def redirect_root_to_docs():
#     return RedirectResponse("/prompt/playground")

# @app.get("/prompt/playground")
# async def playground():
#     return {"message": "Welcome to the Playground!"}

# async def download_file(link, save_path):
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.get(link)
#             if response.status_code == 200:
#                 with open(save_path, 'wb') as file:
#                     file.write(response.content)
#                 logger.info(f"파일 다운로드 완료: {save_path}")
#             else:
#                 logger.error(f"파일 다운로드 실패: {link} (상태 코드: {response.status_code})")
#     except Exception as e:
#         logger.error(f"파일 다운로드 중 오류 발생: {e}")


# async def wait_for_download(file_path, timeout=30):
#     """다운로드 완료 대기 함수"""
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         if os.path.exists(file_path):
#             print(f"다운로드 완료: {file_path}")
#             return True
#         time.sleep(1)
#     print(f"다운로드 시간 초과: {file_path}")
#     return False


# async def process_files_in_dir():
#     tasks = []
#     for file_name in os.listdir(PDF_DIR):
#         file_path = os.path.join(PDF_DIR, file_name)
#         file_extension = file_name.split('.')[-1].lower()

#         if file_extension == 'pdf':
#             tasks.append(process_pdf(file_path))
#         elif file_extension == 'csv':
#             tasks.append(process_csv(file_path))
#         elif file_extension == 'json':
#             tasks.append(process_json(file_path))
#         elif file_extension == 'xml':
#             tasks.append(process_xml(file_path))

#     if tasks:
#         await asyncio.gather(*tasks)
#         logger.info("모든 파일 처리 완료")


# async def process_pdf(file_path):
#     try:
#         loader = PyPDFLoader(file_path)
#         documents = loader.load()
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_documents(documents)
#         vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
#         vectorstore.persist()
#         logger.info(f"PDF 청크 및 벡터 저장 완료: {file_path}")
#     except Exception as e:
#         logger.error(f"PDF 처리 중 오류 발생: {e}")
        
# # CSV 처리 및 벡터화 저장
# async def process_csv(file_path: str):
#     try:
#         df = pd.read_csv(file_path)
#         text_data = df['column1'].astype(str) + ' ' + df['column2'].astype(str)
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_text("\n".join(text_data))
#         vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
#         vectorstore.persist()
#         logger.info(f"CSV 데이터 및 벡터 저장 완료: {file_path}")
#     except Exception as e:
#         logger.error(f"CSV 처리 중 오류 발생: {e}")

# # JSON 처리 및 벡터화 저장
# async def process_json(file_path: str):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         text_data = [str(item['key']) for item in data if item.get('key')]
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_text("\n".join(text_data))
#         vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
#         vectorstore.persist()
#         logger.info(f"JSON 데이터 및 벡터 저장 완료: {file_path}")
#     except Exception as e:
#         logger.error(f"JSON 처리 중 오류 발생: {e}")

# # XML 처리 및 벡터화 저장
# async def process_xml(file_path: str):
#     try:
#         tree = ET.parse(file_path)
#         root = tree.getroot()
#         text_data = [elem.text for elem in root.iter('tag_name') if elem.text]
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         chunks = splitter.split_text("\n".join(text_data))
#         vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
#         vectorstore.persist()
#         logger.info(f"XML 데이터 및 벡터 저장 완료: {file_path}")
#     except Exception as e:
#         logger.error(f"XML 처리 중 오류 발생: {e}")


# async def crawl_and_process_files():
#     chrome_options = Options()
#     chrome_options.add_experimental_option("prefs", {
#         "download.default_directory": os.path.abspath(PDF_DIR),  # PDF_DIR의 절대 경로로 설정
#         "download.prompt_for_download": False,  # 다운로드 시 저장 경로 묻지 않음
#         "download.directory_upgrade": True,     # 디렉토리 업그레이드 허용
#         "safebrowsing.enabled": True,           # 안전 브라우징 활성화
#         "plugins.always_open_pdf_externally": True  # PDF를 새 탭이 아닌 외부로 다운로드
#     })

#     print("다운로드 경로:", os.path.abspath(PDF_DIR))
#     # chrome_options.add_argument("--headless")  # 헤드리스 모드로 실행
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument(f"--download-default-directory={PDF_DIR}")

#     driver = webdriver.Chrome(options=chrome_options)
#     driver.implicitly_wait(10)

#     try:
#         url = "https://www.data.go.kr/tcs/opd/ndm/view.do"
#         driver.get(url)
#         keywords = ["다문화", "외국인", "한국"]

#         for keyword in keywords:
#             search_box = driver.find_element(By.ID, "input-keyword")
#             search_box.clear()
#             search_box.send_keys(keyword)
#             search_box.send_keys(Keys.RETURN)
#             time.sleep(2)

#             while True:
#                 try:
#                     tab_xpath = "//a[@data-tooltip-content='#tooltip_cont_4' and text()='상세목록']"
#                     tab = WebDriverWait(driver, 10).until(
#                         EC.element_to_be_clickable((By.XPATH, tab_xpath))
#                     )
#                     tab.click()

#                     result_list = WebDriverWait(driver, 60).until(
#                         EC.presence_of_all_elements_located(
#                             (By.XPATH, "//div[@class='graph-section']//div[@class='padding' and @id='table-Info-area']//div[@class='search-result']//div[@class='cont-area']//div[@class='result-list']//ul[@id='srchUl']")
#                         )
#                     )

#                     for item in result_list:
#                         try:
#                             btn_open_details = item.find_elements(By.CLASS_NAME, "btn-open-detail-section")

#                             for btn_open_detail in btn_open_details:
#                                 doc_id = btn_open_detail.get_attribute("data-docid")
#                                 doc_id = doc_id.replace("FILE_", "")
#                                 detail_url = f"https://www.data.go.kr/data/{doc_id}/fileData.do"
#                                 driver.get(detail_url)

#                                 try:
#                                     WebDriverWait(driver, 20).until(
#                                         EC.presence_of_element_located((By.XPATH, "//*[@id='tab-layer-file']/div[2]/div[2]/a"))
#                                     )

#                                     download_button_xpath = "//*[@id='tab-layer-file']/div[2]/div[2]/a"
#                                     download_button = driver.find_element(By.XPATH, download_button_xpath)
#                                     download_button.click()

#                                     # # 다운로드 완료 대기
#                                     # download_success = await wait_for_download(PDF_DIR) 

#                                     # if download_success:
#                                     #     await process_files_in_dir()

#                                 except Exception as e:
#                                     logger.warning(f"다운로드 버튼을 찾을 수 없습니다. doc_id: {doc_id}, 오류: {e}")
#                                     driver.back()
#                                     continue

#                         except Exception as e:
#                             logger.error(f"항목 처리 중 오류 발생: {e}")
#                 except Exception as e:
#                     logger.error(f"처리 중 오류 발생: {e}")
#                     break
#     finally:
#         driver.quit()
#         logger.info("크롤링 및 파일 처리 종료")


# # 메인 함수 호출 (비동기 실행)
# asyncio.run(crawl_and_process_files())



# class InputChat(BaseModel):
#     messages: list[str]

# @app.post("/chat")
# async def chat(input: str = Form(...), file: Optional[UploadFile] = File(None)):
#     try:
#         input_data = InputChat(**json.loads(input))
#         result = {}

#         if file:
#             image = Image.open(io.BytesIO(await file.read()))
#             ocr_text = pytesseract.image_to_string(image, lang="kor+eng")
#             input_data.messages.append(ocr_text)
#             result["ocr_text"] = ocr_text

#         result["chatbot_response"] = chat_chain.invoke(input_data.messages)
#         return JSONResponse(content={"message": "Chat response", "result": result, "input_messages": input_data.messages})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    
# @app.get("/crawl")
# async def crawl_files():
#     try:
#         await crawl_and_process_files()
#         return JSONResponse(content={"message": "Files are being crawled and processed!"}, status_code=200)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# add_routes(
#     app,
#     chat_chain.with_types(input_type=InputChat),
#     path="/chat",
#     enable_feedback_endpoint=True,
#     enable_public_trace_link_endpoint=True,
#     playground_type="chat",
# )

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
