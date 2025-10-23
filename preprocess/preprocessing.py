import re, time, math, requests, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from datetime import datetime

import pandas as pd
import numpy as np
import os

file_path   = '2025_빅콘테스트_데이터_레이아웃_20250902/1456289/big_data_set1_f.csv' 
file_path_1 = '2025_빅콘테스트_데이터_레이아웃_20250902/1456289/big_data_set2_f.csv' 
file_path_2 = '2025_빅콘테스트_데이터_레이아웃_20250902/1456289/big_data_set3_f.csv' 

def group_industries(df):
    """
    지정된 규칙에 따라 업종명을 통합하는 함수
    """
    
    # 통합 규칙 정의 (딕셔너리 형태)
    grouping_map = {
        # '음료 중심' 그룹
        '카페': '음료 중심', '커피전문점': '음료 중심', '테이크아웃커피': '음료 중심', 
        '주스': '음료 중심', '차': '음료 중심',
        # '베이커리' 그룹
        '베이커리': '베이커리', '와플/크로플': '베이커리', '마카롱': '베이커리', 
        '도너츠': '베이커리', '탕후루': '베이커리', '떡/한과': '베이커리',
        # '건강식품' 그룹
        '건강식품': '건강식품', '인삼제품': '건강식품', '건강원': '건강식품'
    }

    # .map() 함수를 사용하여 업종명을 변환합니다.
    # grouping_map에 없는 업종은 원래 이름(x)을 그대로 사용합니다.
    df['업종명_통합'] = df['HPSN_MCT_ZCD_NM'].map(lambda x: grouping_map.get(x, x))
    
    return df

def strip_bunji(addr: str) -> str:
    """주소 끝의 번지/호수 숫자부 제거"""
    if not isinstance(addr, str): return addr
    s = addr.strip()
    s = re.sub(r"\s*\.$", "", s)                 # 끝의 .
    s = re.sub(r"\s\d[\d\-]*\s*\.?$", "", s)     # 끝의 숫자/하이픈/점
    s = re.sub(r"\s+", " ", s).strip()
    return s

def geocode(addr: str, rest_api_key: str, timeout: int = 10):
    """카카오 주소→좌표"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {rest_api_key}"}
    try:
        r = requests.get(url, headers=headers, params={"query": addr}, timeout=timeout)
        if r.status_code != 200: return None, None
        docs = r.json().get("documents", [])
        if not docs: return None, None
        d0 = docs[0]
        return float(d0["x"]), float(d0["y"])
    except Exception:
        return None, None

def _load_shp(shp_path: str) -> gpd.GeoDataFrame:
    """국토부 행정동 SHP 로드 → WGS84로 변환"""
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

# ---------------------------
# 1차: 전체 처리 (주소→좌표→within)  **DF 입력 버전**
# ---------------------------
def classify_with_molit_df(df_in: pd.DataFrame,
                           addr_col: str,
                           shp_path: str,
                           rest_api_key: str,
                           progress_every: int = 50) -> pd.DataFrame:
    df = df_in.copy()
    if addr_col not in df.columns:
        raise KeyError(f"'{addr_col}' not in final_df.columns: {list(df.columns)}\n"
                       f"→ 위 병합 단계에서 master_cols_to_merge 에 '{addr_col}' 를 포함시켜 주세요.")
    df[addr_col] = df[addr_col].astype(str).apply(lambda x: re.sub(r"\s+", " ", x).strip())

    gdf = _load_shp(shp_path)

    # 주소→좌표
    lngs, lats, oks = [], [], []
    start, n = datetime.now(), len(df)
    for i, addr in enumerate(df[addr_col].astype(str), start=1):
        lng, lat = geocode(addr, rest_api_key)
        lngs.append(lng); lats.append(lat); oks.append(lng is not None and lat is not None)
        if (i % progress_every == 0) or (i == n):
            elapsed = (datetime.now() - start).total_seconds()
            rate = i/elapsed if elapsed else 0
            eta = (n-i)/rate if rate else math.nan
            print(f"[{i}/{n}] ok={sum(oks)}, fail={i-sum(oks)}, elapsed={elapsed:.1f}s, speed={rate:.2f}/s, ETA~{eta:.1f}s", flush=True)

    df["lng"], df["lat"] = lngs, lats

    # 좌표→Point→within
    pts = gpd.GeoDataFrame(
        df,
        geometry=[Point(x,y) if (x is not None and y is not None) else None for x,y in zip(df["lng"], df["lat"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, gdf[["ADM_CD","ADM_NM","geometry"]], how="left", predicate="within")
    df["dong"] = joined["ADM_NM"]
    return df

# ---------------------------
# NaN만 재시도(원주소→번지제거)
# ---------------------------
def _geocode_and_join(sub_df: pd.DataFrame,
                      shp_path: str,
                      addr_col_for_geocode: str,
                      key_col_for_merge: str,
                      rest_api_key: str,
                      progress_every: int = 25) -> pd.DataFrame:
    n = len(sub_df); lats, lngs = [], []
    start = datetime.now()
    for i, addr in enumerate(sub_df[addr_col_for_geocode].astype(str), start=1):
        x, y = geocode(addr, rest_api_key)
        lngs.append(x); lats.append(y)
        if (i % progress_every == 0) or (i == n):
            elapsed = (datetime.now() - start).total_seconds()
            ok_cnt = sum(v is not None for v in lngs)
            print(f"[{i}/{n}] ok={ok_cnt}, fail={i-ok_cnt}, elapsed={elapsed:.1f}s", flush=True)

    sub_geo = sub_df.copy()
    sub_geo["lng"] = lngs; sub_geo["lat"] = lats
    sub_geo["geometry"] = [Point(x,y) if (x is not None and y is not None) else None for x,y in zip(sub_geo["lng"], sub_geo["lat"])]
    sub_geo = gpd.GeoDataFrame(sub_geo, geometry="geometry", crs="EPSG:4326")

    gdf = _load_shp(shp_path)
    joined = gpd.sjoin(sub_geo[sub_geo["geometry"].notna()],
                       gdf[["ADM_CD","ADM_NM","geometry"]],
                       how="left", predicate="within")

    out = joined[[key_col_for_merge, "ADM_NM"]].rename(columns={"ADM_NM":"dong"})
    out = out.dropna(subset=["dong"]).drop_duplicates(subset=[key_col_for_merge], keep="first")
    return out

def rerun_only_nans_with_bunji_strip(df_with_dong: pd.DataFrame,
                                     shp_path: str,
                                     addr_col: str,
                                     rest_api_key: str,
                                     progress_every: int = 25) -> pd.DataFrame:
    base = df_with_dong.copy()

    # Step1: 원주소 재시도
    mask1 = base["dong"].isna()
    print(f"[step1] NaN {mask1.sum()} rows")
    if mask1.sum() > 0:
        sub1 = base.loc[mask1, [addr_col]].copy()
        res1 = _geocode_and_join(sub1,
                                 shp_path,
                                 addr_col_for_geocode=addr_col,
                                 key_col_for_merge=addr_col,
                                 rest_api_key=rest_api_key,
                                 progress_every=progress_every)
        base = base.merge(res1, on=addr_col, how="left", suffixes=("","_new"))
        base["dong"] = base["dong"].fillna(base["dong_new"])
        base = base.drop(columns=["dong_new"])

    # Step2: 번지 제거 후 재시도
    mask2 = base["dong"].isna()
    print(f"[step2] NaN after step1: {mask2.sum()} rows")
    if mask2.sum() > 0:
        sub2 = base.loc[mask2, [addr_col]].copy()
        sub2["addr_stripped"] = sub2[addr_col].map(strip_bunji)
        res2 = _geocode_and_join(sub2[[addr_col, "addr_stripped"]],
                                 shp_path,
                                 addr_col_for_geocode="addr_stripped",
                                 key_col_for_merge=addr_col,
                                 rest_api_key=rest_api_key,
                                 progress_every=progress_every)
        base = base.merge(res2, on=addr_col, how="left", suffixes=("","_new"))
        base["dong"] = base["dong"].fillna(base["dong_new"])
        base = base.drop(columns=["dong_new"])

    print(f"[done] NaN left: {base['dong'].isna().sum()}")
    return base

# ---------------------------
# 특수 예외 처리
# ---------------------------
def apply_exceptions(df: pd.DataFrame, addr_col: str) -> pd.DataFrame:
    """하드코딩 예외: '서울 성동구 천호대로 276'→'용답동', '논현2동/중앙동'→NaN"""
    out = df.copy()
    out.loc[out[addr_col].astype(str).str.strip() == "서울 성동구 천호대로 276", "dong"] = "용답동"
    out.loc[out["dong"].isin(["논현2동", "중앙동"]), "dong"] = pd.NA
    return out

# ---------------------------
# 파이프라인 실행 (DF 입력용 엔트리)
# ---------------------------
def run_pipeline_from_df(df: pd.DataFrame,
                         shp_path: str,
                         rest_api_key: str,
                         addr_col: str = "MCT_BSE_AR",
                         progress_every: int = 50,
                         save_path: str | None = None) -> pd.DataFrame:
    """
    final_df(=위 병합 결과)를 그대로 받아 동(행정동) 분류 전체 파이프라인 수행
    """
    # 1차: 주소→좌표→행정동
    df_out = classify_with_molit_df(df_in=df,
                                    addr_col=addr_col,
                                    shp_path=shp_path,
                                    rest_api_key=rest_api_key,
                                    progress_every=progress_every)
    # NaN 재시도(원주소/번지제거)
    df_out = rerun_only_nans_with_bunji_strip(df_with_dong=df_out,
                                              shp_path=shp_path,
                                              addr_col=addr_col,
                                              rest_api_key=rest_api_key,
                                              progress_every=25)
    # 예외 처리
    df_out = apply_exceptions(df_out, addr_col=addr_col)

    if save_path:
        df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"saved: {save_path}")
    return df_out

def grouped_df():
    # 1) set1 읽고 업종 통합
    data = pd.read_csv(file_path, encoding='cp949')
    df1_master = group_industries(data)

    # 동분류에 필요한 최소 컬럼만 추출 (가맹점 당 1회만 처리하도록 중복 제거)
    master_cols = ['ENCODED_MCT', 'MCT_NM', '업종명_통합', 'HPSN_MCT_BZN_CD_NM', 'MCT_BSE_AR']
    master_for_dong = (
        df1_master[master_cols]
        .dropna(subset=['MCT_BSE_AR'])             # 주소 없는 행 제외(선택)
        .drop_duplicates(subset=['ENCODED_MCT'])   # 가맹점 단위로 1회만 지오코딩
        .reset_index(drop=True)
    )

    # 2) set1(마스터)만으로 동분류 수행 (지오코딩 + within)
    REST_API_KEY = "33648d44f1f0c8e06517947b6e904bdd"  # 이미 아래 main에도 있지만, 함수 내부에서도 사용할 수 있게
    SHP_PATH = "BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp"
    master_with_dong = run_pipeline_from_df(
        df=master_for_dong,
        shp_path=SHP_PATH,
        rest_api_key=REST_API_KEY,
        addr_col="MCT_BSE_AR",
        progress_every=50,
        save_path=None  # 마스터 결과 중간저장은 생략
    )

    # 3) set2, set3 병합 (시간/매출 등 트랜잭션 성격)
    df2 = pd.read_csv(file_path_1, encoding='cp949')
    df3 = pd.read_csv(file_path_2, encoding='cp949')
    merged_tx = pd.merge(df2, df3, on=['ENCODED_MCT', 'TA_YM'], how='inner')

    # 4) 최종 병합: 트랜잭션(joined) + 마스터(동/좌표/업종 통합 등)
    #    master_with_dong에서 필요한 컬럼만 추려서 붙인다
    keep_cols = ['ENCODED_MCT', 'MCT_NM', '업종명_통합', 'HPSN_MCT_BZN_CD_NM', 'MCT_BSE_AR', 'dong']
    master_slim = master_with_dong[keep_cols].copy()

    final_df = pd.merge(
        merged_tx,
        master_slim,
        on='ENCODED_MCT',
        how='left'
    )

    # 5) 특수값 치환
    final_df.replace(-999999.9, np.nan, inplace=True)

    return final_df


if __name__ == "__main__":
    # 1) 너의 병합 함수로 최종 DF 만들기
    final_df = grouped_df() 

    # 2) 파일 저장
    save_path = "merged_data.csv"
    final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 최종 병합 데이터 저장 완료: {save_path}")