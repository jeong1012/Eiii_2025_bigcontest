import pandas as pd
import requests
import time
import re
import os
from dotenv import load_dotenv

# --- ⚙️ 설정 ---
# .env 파일에서 환경 변수 로드
load_dotenv()

KAKAO_API_KEY = os.getenv('KAKAO_API_KEY')

INPUT_CSV_PATH = 'data/20251010144853_중심 관광지.csv'

FINAL_OUTPUT_CSV_PATH = 'data/final_dong.csv'


# --- 📞 API 요청 함수들 ---

def get_address_simple(query):
    """(1단계용) 카카오맵 API를 호출하여 장소의 주소를 반환합니다."""
    url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={query}"
    headers = {'Authorization': f'KakaoAK {KAKAO_API_KEY}'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('documents'):
            return data['documents'][0].get('address_name', '주소 없음')
        else:
            return '검색 결과 없음'
    except requests.exceptions.RequestException:
        return '요청 오류'
    except (KeyError, IndexError):
        return '응답 오류'

def get_address_advanced(place_name):
    """(2단계용) 3단계 검색 전략을 사용하여 실패한 주소를 다시 찾습니다."""
    # 1단계 검색어: 특수문자, 괄호, 불필요 단어 제거
    query1 = re.sub(r'[/점]|(\(.*\))|휴업중|지점', ' ', place_name).strip()
    
    # 2단계 검색어: '/' 뒤 또는 마지막 단어(지점명 추정)를 제거
    base_name = query1.split('/')[0].strip()
    query2 = ' '.join(base_name.split()[:-1]) if len(base_name.split()) > 1 else base_name

    # 순서대로 검색 시도
    for i, query in enumerate([query1, query2]):
        if not query or (i > 0 and query == [query1, query2][i-1]):
            continue
        print(f"    - 고급 검색 시도 ({i+1}/2): '{query}'")
        url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={query}"
        headers = {'Authorization': f'KakaoAK {KAKAO_API_KEY}'}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get('documents'):
                address = data['documents'][0].get('address_name', '주소 없음')
                print(f"    >> 재검색 성공: {address}")
                return address
        except Exception:
            continue
        time.sleep(0.1)
            
    print("    >> 최종 검색 실패")
    return '최종 검색 실패'

# --- 🗺️ 행정동 변환 함수 ---
def map_address_to_dong(address):
    """주소 문자열을 성동구 행정동으로 매핑합니다."""
    if not isinstance(address, str): return '주소없음'
    mapping_rules = [
        ('성수1가1동', '성수1가1동'), ('성수1가2동', '성수1가2동'), ('성수2가1동', '성수2가1동'), 
        ('성수2가3동', '성수2가3동'), ('왕십리도선동', '왕십리도선동'), ('왕십리2동', '왕십리2동'), 
        ('금호2·3가동', '금호2·3가동'), ('금호1가동', '금호1가동'), ('금호4가동', '금호4가동'), 
        ('행당1동', '행당1동'), ('행당2동', '행당2동'), ('성수동1가', '성수1가1동'), 
        ('성수동2가', '성수2가1동'), ('금호동1가', '금호1가동'), ('금호동2가', '금호2·3가동'), 
        ('금호동3가', '금호2·3가동'), ('금호동4가', '금호4가동'), ('도선동', '왕십리도선동'), 
        ('상왕십리동', '왕십리2동'), ('하왕십리동', '왕십리2동'), ('행당동', '행당1동'), 
        ('홍익동', '마장동'), ('마장동', '마장동'), ('용답동', '용답동'), 
        ('응봉동', '응봉동'), ('사근동', '사근동'), ('송정동', '송정동'), ('옥수동', '옥수동')
    ]
    for keyword, result_dong in mapping_rules:
        if keyword in address: return result_dong
    return '매칭실패(타지역)'

# --- 🚀 메인 실행 로직 ---
def main():
    # --- 파일 불러오기 ---
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"✅ 원본 파일 '{INPUT_CSV_PATH}'을(를) 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: '{INPUT_CSV_PATH}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return

    # --- 1단계: 전체 관광지 주소 1차 크롤링 ---
    print("\n--- 1단계: 카카오맵 API로 전체 관광지 주소 1차 검색을 시작합니다 ---")
    addresses = []
    for index, row in df.iterrows():
        tour_name = row['관광지명']
        addr = get_address_simple(tour_name)
        addresses.append(addr)
        print(f"  [{index + 1}/{len(df)}] '{tour_name}' -> {addr}")
        time.sleep(0.1)
    df['주소'] = addresses

    # --- 2단계: 실패 데이터 대상 고급 재검색 ---
    fail_conditions = ['검색 결과 없음', '요청 오류', '응답 오류']
    failed_df = df[df['주소'].isin(fail_conditions)]
    
    if not failed_df.empty:
        print(f"\n--- 2단계: 총 {len(failed_df)}개의 실패 데이터를 대상으로 고급 재검색을 시작합니다 ---")
        for index, row in failed_df.iterrows():
            name = row['관광지명']
            print(f"\n  재검색 대상: '{name}'")
            new_address = get_address_advanced(name)
            df.loc[index, '주소'] = new_address
    else:
        print("\n--- 2단계: 1차 검색에 모두 성공하여 재검색을 건너뜁니다 ---")

    # --- 3단계: 전체 주소에 대해 행정동 매핑 ---
    print("\n--- 3단계: 전체 주소에 대해 행정동 변환을 시작합니다 ---")
    df['행정동'] = df['주소'].apply(map_address_to_dong)
    print("✅ 행정동 변환 완료.")

    # --- 4단계: 최종 데이터 필터링 ---
    print("\n--- 4단계: 최종 데이터 필터링을 시작합니다 ---")
    
    # '분류' 컬럼 존재 여부 확인
    if '분류' not in df.columns:
        print("⚠️ 경고: '분류' 컬럼이 없어 숙박시설 필터링을 건너뜁니다.")
        final_df = df[df['행정동'] != '매칭실패(타지역)'].copy()
    else:
        original_rows = len(df)
        final_df = df[(df['행정동'] != '매칭실패(타지역)') & (df['분류'] != '숙박')].copy()
        removed_rows = original_rows - len(final_df)
        print(f"✅ '매칭실패(타지역)' 및 '숙박' 분류 데이터 {removed_rows}개를 제외했습니다.")

    # --- 5단계: 결과 파일 저장 ---
    final_df.to_csv(FINAL_OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n🎉 모든 작업 완료! 최종 결과가 '{FINAL_OUTPUT_CSV_PATH}' 파일에 저장되었습니다.")

if __name__ == '__main__':
    main()