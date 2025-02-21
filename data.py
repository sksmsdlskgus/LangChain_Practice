import os
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PDF_DIR = "pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

###################################################### 국가법령정보 조례,법령,판례 엑셀 저장

def fetch_data_generic(target_type, keywords):
    print(f"{target_type} 데이터 수집 시작...")

    id = "lnh28862331"
    rows = []

    for keyword in keywords:
        base_url = (
            f"http://www.law.go.kr/DRF/lawSearch.do?OC={id}"
            f"&target={target_type}&type=XML"
            f"&query={keyword}"
            "&display=100"
            "&prncYd20000101~20250131"
            "&search=2"
        )

        # 요청 및 데이터 처리
        res = requests.get(base_url)
        xtree = ET.fromstring(res.text)
        totalCnt = int(xtree.find('totalCnt').text)
        logger.info(f"{keyword}의 {target_type} 갯수： {totalCnt}")
        logger.info(f"XML 데이터:\n{ET.tostring(xtree, encoding='unicode')}")

        for page in range(1, (totalCnt // 100) + 2):
            url = f"{base_url}&page={page}"
            response = requests.get(url)
            xtree = ET.fromstring(response.text)
            # target_type에 따라 인덱스 다르게 설정 - 데이터 추출 부분
            if target_type == "prec":
                items = xtree[5:]  # 판례는 5
            else:
                items = xtree[8:] 

            for node in items:
                if target_type == "prec":  # 판례
                    판례일련번호 = node.find("판례일련번호").text
                    사건명 = node.find("사건명").text
                    사건번호 = node.find("사건번호").text
                    선고일자 = node.find("선고일자").text
                    법원명 = node.find("법원명").text
                    사건종류명 = node.find("사건종류명").text
                    사건종류코드 = node.find("사건종류코드").text
                    판결유형 = node.find("판결유형").text
                    선고 = node.find("선고").text
                    판례상세링크 = node.find("판례상세링크").text

                    rows.append({
                        "판례일련번호": 판례일련번호,
                        "사건명": 사건명,
                        "사건번호": 사건번호,
                        "선고일자": 선고일자,
                        "법원명": 법원명,
                        "사건종류명": 사건종류명,
                        "사건종류코드": 사건종류코드,
                        "판결유형": 판결유형,
                        "선고": 선고,
                        "판례상세링크": 판례상세링크,
                    })

                elif target_type == "law":  # 법령
                    법령일련번호 = node.find("법령일련번호").text 
                    법령명한글 = node.find("법령명한글").text
                    법령ID = node.find("법령ID").text
                    공포일자 = node.find("공포일자").text
                    공포번호 = node.find("공포번호").text
                    제개정구분명 = node.find("제개정구분명").text
                    소관부처코드 = node.find("소관부처코드").text
                    법령구분명 = node.find("법령구분명").text
                    시행일자 = node.find("시행일자").text
                    법령상세링크 = node.find("법령상세링크").text

                    rows.append({
                        "법령일련번호": 법령일련번호,
                        "법령명한글": 법령명한글,
                        "법령ID": 법령ID,
                        "공포일자": 공포일자,
                        "공포번호": 공포번호,
                        "제개정구분명": 제개정구분명,
                        "소관부처코드": 소관부처코드,
                        "법령구분명": 법령구분명,
                        "시행일자": 시행일자,
                        "법령상세링크": 법령상세링크,
                    })

                elif target_type == "ordin":  # 조례
                    자치법규일련번호 = node.find("자치법규일련번호").text
                    자치법규명 = node.find("자치법규명").text
                    자치법규ID = node.find("자치법규ID").text
                    공포일자 = node.find("공포일자").text
                    공포번호 = node.find("공포번호").text
                    제개정구분명 = node.find("제개정구분명").text
                    지자체기관명 = node.find("지자체기관명").text
                    자치법규종류 = node.find("자치법규종류").text
                    시행일자 = node.find("시행일자").text
                    자치법규상세링크 = node.find("자치법규상세링크").text
                    자치법규분야명 = node.find("자치법규분야명").text
                    참조데이터구분 = node.find("참조데이터구분").text

                    rows.append({
                        "자치법규일련번호": 자치법규일련번호,
                        "자치법규명": 자치법규명,
                        "자치법규ID": 자치법규ID,
                        "공포일자": 공포일자,
                        "공포번호": 공포번호,
                        "제개정구분명": 제개정구분명,
                        "지자체기관명": 지자체기관명,
                        "자치법규종류": 자치법규종류,
                        "시행일자": 시행일자,
                        "자치법규상세링크": 자치법규상세링크,
                        "자치법규분야명": 자치법규분야명,
                        "참조데이터구분": 참조데이터구분,
                    })

    return rows

# 판례, 법령, 조례 데이터 처리 함수들
def fetch_data_prec():
    rows = fetch_data_generic("prec", ["외국인", "다문화"])
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(PDF_DIR, "판례.xlsx"), index=False)

def fetch_data_law():
    rows = fetch_data_generic("law", ["외국인", "다문화"])
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(PDF_DIR, "법령.xlsx"), index=False)

def fetch_data_ordin():
    rows = fetch_data_generic("ordin", ["외국인", "다문화"])
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(PDF_DIR, "조례,규칙.xlsx"), index=False)