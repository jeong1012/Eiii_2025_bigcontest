import os
import time
import json
import requests
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# .env 파일에서 환경 변수 로드
load_dotenv()


def generate_keywords_with_gemini(industry):
    """Gemini API를 사용하여 업종별 트렌드 키워드를 생성하는 함수"""
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    너는 한국 소상공인 전문 마케팅 분석가야.
    '{industry}' 업종과 관련해서, 요즘 대한민국 전체에서 사람들이 많이 검색할만한 트렌드 키워드를 10개 생성해줘.
    결과는 오직 JSON 형식의 문자열 리스트(string array)여야 해. 다른 설명은 절대 추가하지 마.
    """
    for attempt in range(3):
        try:
            print(f"    - '{industry}' 키워드 생성 요청... (시도 {attempt + 1}/3)")
            response = model.generate_content(prompt, request_options={"timeout": 60})
            print(f"    - '{industry}' 키워드 생성 완료.")
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except Exception as e:
            if "429" in str(e) and "Quota" in str(e):
                print(f"    - Gemini API 사용량 초과. 60초 후 재시도... ({attempt + 1}/3)")
                time.sleep(61)
                continue
            else:
                print(f"    - Gemini 오류 발생 (재시도 안함): {e}")
                return []
    print(f"    - '{industry}' 키워드 생성 3번 재시도 실패.")
    return []

def get_naver_trend_data(keywords, start_date, end_date):
    """Naver Datalab API를 호출하여 검색어 트렌드 데이터를 가져오는 함수"""
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    url = "https://openapi.naver.com/v1/datalab/search"
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "week",
        "keywordGroups": [{"groupName": keyword, "keywords": [keyword]} for keyword in keywords]
    }
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"     - Naver API 오류: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        print("     - Naver API 요청 시간 초과")
        return None
    except Exception as e:
        print(f"     - Naver API 요청 중 오류 발생: {e}")
        return None

# --- 2. 메인 실행 로직 ---

def main():
    """전체 프로세스를 실행하는 메인 함수"""
    
    # --- 1단계: Gemini API로 업종별 키워드 생성 및 저장 ---
    print("[1단계] Gemini API로 업종별 인기 키워드를 생성합니다...")
    try:
        df = pd.read_csv('data/merged_data.csv')
    except FileNotFoundError:
        print("[오류] 'big_data_set1_f.csv' 파일을 찾을 수 없습니다. 스크립트와 같은 경로에 파일이 있는지 확인하세요.")
        return
        
    unique_industries = df['업종명_통합'].dropna().unique()
    
    industry_keywords = {}
    for industry in tqdm(unique_industries, desc="업종별 키워드 생성"):
        keywords = generate_keywords_with_gemini(industry)
        if keywords:
            industry_keywords[industry] = keywords
        time.sleep(2)  # API 과호출 방지를 위한 딜레이

    with open('keywords.json', 'w', encoding='utf-8') as f:
        json.dump(industry_keywords, f, ensure_ascii=False, indent=4)
    print("\n✅ [1단계 완료] 생성된 키워드를 'keywords.json' 파일로 저장했습니다.")

    # --- 2단계: Naver Datalab으로 트렌드 데이터 조회 및 저장 ---
    print("\n[2단계] Naver Datalab으로 전체 기간의 트렌드 데이터를 조회합니다...")
    
    # Naver API는 최대 1년까지 조회 가능하므로 기간을 나눔
    date_ranges = [
        ("2023-01-01", "2024-12-31"),  # 첫 12개월
        ("2024-01-01", "2024-12-31")   # 나머지 기간
    ]
    
    trend_results = []
    
    # 1단계에서 생성된 industry_keywords 딕셔너리를 바로 사용
    for industry, keywords in tqdm(industry_keywords.items(), desc="업종별 트렌드 조회"):
        for start_date, end_date in date_ranges:
            # 키워드를 5개씩 묶어서 API 호출 (API 제약사항)
            for i in range(0, len(keywords), 5):
                keyword_batch = keywords[i:i+5]
                trend_data = get_naver_trend_data(keyword_batch, start_date, end_date)
                
                if trend_data and trend_data.get('results'):
                    for result in trend_data['results']:
                        keyword_title = result['title']
                        for weekly_data in result['data']:
                            trend_results.append({
                                "업종명_통합": industry,
                                "트렌드_키워드": keyword_title,
                                "기간": weekly_data['period'],
                                "검색량_비율": weekly_data['ratio']
                            })
                # API 과호출 방지를 위한 딜레이
                time.sleep(0.2) 
                
    # --- 3단계: 최종 결과 파일로 저장 ---
    if not trend_results:
        print("\n[오류] 조회된 트렌드 데이터가 없습니다. API 키 또는 네트워크 상태를 확인하세요.")
        return

    result_df = pd.DataFrame(trend_results)
    output_filename = 'industry_trend_timeseries2.csv'
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ [최종 완료] 전체 트렌드 분석 결과를 '{output_filename}' 파일로 저장했습니다.")

if __name__ == "__main__":
    main()