import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import MinMaxScaler


def load_data_raw(path: str) -> pd.DataFrame:
    """CSV 파일 로드 및 주요 지표 전처리"""
    try:
        df = pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")

    # 연월 숫자화
    df['TA_YM'] = pd.to_numeric(df['TA_YM'], errors='coerce')

    # -------------------- (1) 객단가 수준 매핑 --------------------
    if 'RC_M1_AV_NP_AT' in df.columns:
        mapping = {
            '1_10%이하': 5,
            '2_10-25%': 17,
            '3_25-50%': 37,
            '4_50-75%': 62,
            '5_75-90%': 82,
            '6_90%초과(하위 10% 이하)': 95
        }
        df['RC_M1_AV_NP_AT_NUM'] = df['RC_M1_AV_NP_AT'].map(mapping)

    # -------------------- (2) 매출 경쟁력 점수화 --------------------
    if 'M1_SME_RY_SAA_RAT' in df.columns:
        df['매출경쟁력_점수'] = df['M1_SME_RY_SAA_RAT']
    elif 'M12_SME_RY_SAA_PCE_RT' in df.columns:
        df['매출경쟁력_점수'] = 100 - df['M12_SME_RY_SAA_PCE_RT']

    # -------------------- (3) 연령대 통합 계산 --------------------
    age_cols = {
        '20대이하': ['M12_MAL_1020_RAT', 'M12_FME_1020_RAT'],
        '30대':     ['M12_MAL_30_RAT',   'M12_FME_30_RAT'],
        '40대':     ['M12_MAL_40_RAT',   'M12_FME_40_RAT'],
        '50대':     ['M12_MAL_50_RAT',   'M12_FME_50_RAT'],
        '60대이상': ['M12_MAL_60_RAT',   'M12_FME_60_RAT'],
    }
    for age_group, cols in age_cols.items():
        use_cols = [c for c in cols if c in df.columns]
        df[age_group] = df[use_cols].sum(axis=1)

    if 'M1_SME_RY_SAA_RAT' in df.columns:
        df['M1_SME_RY_SALE_CMP_RAT'] = df['M1_SME_RY_SAA_RAT']

    return df

def get_analysis_data(df: pd.DataFrame):
    """
    업종별/가게별 주요 지표 계산 및 최근/전년 데이터 비교
    RadarChart 및 성장 비교용 데이터 반환
    """
    analysis_col = '업종명_통합'
    age_group_cols = ['20대이하', '30대', '40대', '50대', '60대이상']

    # -------------------- (업종별 기본 통계) --------------------
    upjong_age_ratio = df.groupby(analysis_col)[age_group_cols].mean().fillna(0)

    delivery_col = 'M1_SME_RY_DLVR_SALE_RAT' if 'M1_SME_RY_DLVR_SALE_RAT' in df.columns else 'DLV_SAA_RAT'
    delivery_ratio_raw = df.groupby(analysis_col)[delivery_col].mean()
    delivery_ratio = delivery_ratio_raw.dropna().sort_values(ascending=False)
    no_delivery_data = delivery_ratio_raw[delivery_ratio_raw.isnull()].index.tolist()

    male_cols = [c for c in df.columns if 'M12_MAL' in c]
    female_cols = [c for c in df.columns if 'M12_FME' in c]
    df2 = df.copy()
    df2['남성비율'] = df2[male_cols].sum(axis=1)
    df2['여성비율'] = df2[female_cols].sum(axis=1)
    upjong_gender_ratio = df2.groupby(analysis_col)[['남성비율', '여성비율']].mean().fillna(0)

    visit_cols = ['MCT_UE_CLN_NEW_RAT', 'MCT_UE_CLN_REU_RAT']
    upjong_visit_ratio = df.groupby(analysis_col)[visit_cols].mean().fillna(0)

    # -------------------- (RadarChart용 핵심 지표 5개) --------------------
    if 'DLV_SAA_RAT' in df.columns and 'M1_SME_RY_DLVR_SALE_RAT' not in df.columns:
        df['M1_SME_RY_DLVR_SALE_RAT'] = df['DLV_SAA_RAT']
    if 'M1_SME_RY_SAA_RAT' not in df.columns and 'M1_SME_RY_SAA_PCE_RT' in df.columns:
        df['M1_SME_RY_SAA_RAT'] = 100 - df['M1_SME_RY_SAA_PCE_RT']

    radar_metrics = [
        'MCT_UE_CLN_REU_RAT',   # 재방문 고객
        'MCT_UE_CLN_NEW_RAT',   # 신규 고객
        'M1_SME_RY_DLVR_SALE_RAT',  # 배달 매출
        'M1_SME_RY_SAA_RAT',    # 매출 경쟁력
        'RC_M1_AV_NP_AT_NUM'    # 객단가 수준
    ]
    existing_radar_metrics = [c for c in radar_metrics if c in df.columns]

    # -------------------- (MinMax Scaling 업종별 전체기간 기준으로 적용) --------------------
    df_scaled = df.copy()
    scale_targets = [c for c in existing_radar_metrics if c != 'RC_M1_AV_NP_AT_NUM']

    if scale_targets:
        for upjong, subdf in df_scaled.groupby('업종명_통합'):
            sub = subdf.copy()

            # 1️⃣ 이상치 1~99% 구간으로 clip
            for col in scale_targets:
                low, high = sub[col].quantile([0.01, 0.99])
                sub[col] = sub[col].clip(lower=low, upper=high)

            # 2️⃣ 결측값은 업종 평균으로 대체 (NaN → 업종 평균)
            sub[scale_targets] = sub[scale_targets].apply(lambda x: x.fillna(x.mean()))

            # 3️⃣ 로그 변환 (값이 0 이하인 경우 대비 → +1 shift)
            sub_log = sub[scale_targets].apply(lambda x: np.log1p(x))

            # 4️⃣ 업종별 전체기간 기준 MinMaxScaling (fit 한 번)
            scaler = MinMaxScaler((0, 100))
            scaler.fit(sub_log)
            scaled_vals = scaler.transform(sub_log)

            # 5️⃣ 변환 결과 반영
            df_scaled.loc[sub.index, scale_targets] = scaled_vals

    # -------------------- (1) 업종 평균 대비 분석: 최근 3개월) --------------------
    recent_yms = df_scaled['TA_YM'].dropna().unique()
    recent_yms.sort()
    recent_yms_list = recent_yms[-3:].tolist()
    df_recent = df_scaled[df_scaled['TA_YM'].isin(recent_yms_list)]

    store_raw_recent, industry_raw_recent, store_scores_recent = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not df_recent.empty:
        # 매장별 최근 3개월 평균
        store_raw_recent = df_recent.groupby(['업종명_통합', 'ENCODED_MCT'])[existing_radar_metrics].mean()
        # 업종별 평균
        industry_raw_recent = store_raw_recent.groupby('업종명_통합')[existing_radar_metrics].mean()
        # 업종 내 순위 백분율 (0~100)
        store_scores_recent = store_raw_recent.groupby('업종명_통합')[existing_radar_metrics].rank(pct=True) * 100

        # 인덱스 정리
        store_raw_recent = store_raw_recent.reset_index().set_index('ENCODED_MCT')
        store_scores_recent = store_scores_recent.reset_index().set_index('ENCODED_MCT')

    # -------------------- (2) 자체 성장 분석: 전년 7~12 vs 올해 7~12) --------------------
    latest_ym = df_scaled['TA_YM'].max()
    latest_year = int(latest_ym // 100)

    df_current_year = df_scaled[df_scaled['TA_YM'].between(latest_year * 100 + 7, latest_year * 100 + 12)]
    df_previous_year = df_scaled[df_scaled['TA_YM'].between((latest_year - 1) * 100 + 7, (latest_year - 1) * 100 + 12)]

    store_raw_current_year = df_current_year.groupby(['업종명_통합', 'ENCODED_MCT'])[existing_radar_metrics].mean()
    store_raw_previous_year = df_previous_year.groupby(['업종명_통합', 'ENCODED_MCT'])[existing_radar_metrics].mean()

    # -------------------- (3) 성장률 계산 --------------------
    store_growth_scores, store_growth_raw = pd.DataFrame(), pd.DataFrame()
    if not store_raw_current_year.empty and not store_raw_previous_year.empty:
        merged_yoy = pd.merge(
            store_raw_current_year, store_raw_previous_year,
            on=['업종명_통합', 'ENCODED_MCT'], suffixes=('_cur', '_prev'), how='inner'
        ).reset_index()

        growth_cols = []
        for metric in existing_radar_metrics:
            cur_col, prev_col, growth_col = f'{metric}_cur', f'{metric}_prev', f'{metric}_growth'
            # 점수 차이 (포인트 기준)
            merged_yoy[growth_col] = merged_yoy[cur_col] - merged_yoy[prev_col]
            growth_cols.append(growth_col)

        # 업종 내 상대적 순위 백분율 (0~100)
        ranked_growth = merged_yoy.groupby('업종명_통합')[growth_cols].rank(pct=True) * 100

        store_growth_scores = pd.concat(
            [merged_yoy[['ENCODED_MCT']], ranked_growth], axis=1
        ).set_index('ENCODED_MCT')

        store_growth_raw = merged_yoy[['ENCODED_MCT'] + growth_cols].set_index('ENCODED_MCT')

    # -------------------- (결과 반환) --------------------
    return (
        upjong_age_ratio,
        delivery_ratio,
        no_delivery_data,
        upjong_gender_ratio,
        upjong_visit_ratio,
        store_raw_recent,         # ✅ radar용 (0~100)
        industry_raw_recent,      # ✅ radar용 (0~100)
        store_scores_recent,
        store_raw_current_year,   # ✅ YoY용
        store_raw_previous_year,  # ✅ YoY용
        store_growth_scores,
        store_growth_raw,
        existing_radar_metrics
    )


def make_store_summary(
    analysis_df: pd.DataFrame,
    store_category: str,
    age_ratio: pd.DataFrame,
    delivery_ratio: pd.Series,
    age_group_cols: List[str]
) -> str:
    store_age_avg = analysis_df[age_group_cols].mean()
    cat_age_avg   = age_ratio.loc[store_category]
    top_age = store_age_avg.idxmax()
    diff_age = store_age_avg[top_age] - cat_age_avg[top_age]
    diff_age_str = f"{diff_age:+.1f}p"

    male_cols   = [c for c in analysis_df.columns if 'M12_MAL' in c]
    female_cols = [c for c in analysis_df.columns if 'M12_FME' in c]
    male_sum   = float(analysis_df[male_cols].sum(axis=1).mean()) if male_cols else 0.0
    female_sum = float(analysis_df[female_cols].sum(axis=1).mean()) if female_cols else 0.0
    if male_sum + female_sum > 0:
        male_pct = 100 * male_sum / (male_sum + female_sum)
        female_pct = 100 - male_pct
        gender_text = f"남 {male_pct:.1f}%, 여 {female_pct:.1f}%"
    else:
        gender_text = "성별 데이터 없음"

    visit_cols = ['MCT_UE_CLN_NEW_RAT','MCT_UE_CLN_REU_RAT']
    has_visit = all(col in analysis_df.columns for col in visit_cols)
    if has_visit:
        new_avg = float(analysis_df[visit_cols[0]].mean())
        reu_avg = float(analysis_df[visit_cols[1]].mean())
        visit_text = f"신규 {new_avg:.1f}%, 재방문 {reu_avg:.1f}%"
    else:
        visit_text = "신규/재방문 데이터 없음"

    if store_category in delivery_ratio.index:
        rank = delivery_ratio.index.get_loc(store_category) + 1
        total = len(delivery_ratio)
        deli_val = delivery_ratio.loc[store_category]
        deli_text = f"배달 매출 비중 {deli_val:.1f}% (업종 {rank}/{total}위)"
    else:
        deli_text = "배달 매출 데이터 없음"

    return (
        f"이 가게는 **{top_age}** 비중이 가장 높으며(업종 대비 {diff_age_str}), "
        f"성별 구성은 {gender_text}입니다. "
        f"방문 유형은 {visit_text}이며, {deli_text}입니다."
    )

# ------------------------- LLM 컨텍스트 -------------------------
def build_evidence_context_from_store(
    analysis_df, age_group_cols, store_category, age_ratio, delivery_ratio
) -> str:
    s = analysis_df[age_group_cols].mean()
    c = age_ratio.loc[store_category]
    age_lines = [f"- {age}: 가게 {s[age]:.1f}%, 업종 {c[age]:.1f}%" for age in age_group_cols]

    male_cols   = [col for col in analysis_df.columns if 'M12_MAL' in col]
    female_cols = [col for col in analysis_df.columns if 'M12_FME' in col]
    if male_cols and female_cols:
        male = float(analysis_df[male_cols].sum(axis=1).mean())
        fem  = float(analysis_df[female_cols].sum(axis=1).mean())
        total = male + fem
        gender_line = f"- 성별: 남 {100*male/total:.1f}%, 여 {100*fem/total:.1f}%"
    else:
        gender_line = "- 성별: 데이터 없음"

    visit_cols = ['MCT_UE_CLN_NEW_RAT','MCT_UE_CLN_REU_RAT']
    if all(col in analysis_df.columns for col in visit_cols):
        new_avg = float(analysis_df[visit_cols[0]].mean())
        reu_avg = float(analysis_df[visit_cols[1]].mean())
        visit_line = f"- 방문: 신규 {new_avg:.1f}%, 재방문 {reu_avg:.1f}%"
    else:
        visit_line = "- 방문: 데이터 없음"

    if store_category in delivery_ratio.index:
        rank = delivery_ratio.index.get_loc(store_category)+1
        total_cnt = len(delivery_ratio)
        deli = float(delivery_ratio.loc[store_category])
        deli_line = f"- 배달: {deli:.1f}% (업종 {rank}/{total_cnt}위)"
    else:
        deli_line = "- 배달: 데이터 없음"

    return "\n".join(age_lines + [gender_line, visit_line, deli_line])


# ------------------------- 근거 표 유틸 -------------------------
def _clamp_pct(x):
    try:
        if pd.isna(x):
            return x
        x = float(x)
        if x < 0:   return 0.0
        if x > 100: return 100.0
        return x
    except Exception:
        return pd.NA


def evidence_table_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        lines = ["| 지표 | 가게(%) | 업종 평균(%) | 참고 |", "|---|---:|---:|---|"]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['지표']} | {'' if pd.isna(r['가게(%)']) else r['가게(%)']} | "
                f"{'' if pd.isna(r['업종 평균(%)']) else r['업종 평균(%)']} | {r.get('참고','')} |"
            )
        return "\n".join(lines)


# --- 메트릭 카탈로그: KEY -> row(dict) 형태로 반환 ---
def build_metric_catalog(
    analysis_df: pd.DataFrame,
    store_category: str,
    age_ratio: pd.DataFrame,
    delivery_ratio: pd.Series,
    age_group_cols: List[str],
    gender_ratio: pd.DataFrame | None = None,
    visit_ratio: pd.DataFrame | None = None,
) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}

    # 1) 연령
    s_age = analysis_df[age_group_cols].mean(numeric_only=True)
    c_age = age_ratio.loc[store_category] if store_category in age_ratio.index \
        else pd.Series({age: pd.NA for age in age_group_cols})
    for age in age_group_cols:
        s_val = _clamp_pct(s_age.get(age, pd.NA))
        c_val = _clamp_pct(c_age.get(age, pd.NA))
        catalog[f"AGE_{age}"] = {
            "지표": f"연령대 - {age}",
            "가게(%)": None if pd.isna(s_val) else float(round(s_val, 1)),
            "업종 평균(%)": None if pd.isna(c_val) else float(round(c_val, 1)),
            "참고": ""
        }

    # 2) 성별 (가게)
    male_cols = [c for c in analysis_df.columns if 'M12_MAL' in c]
    female_cols = [c for c in analysis_df.columns if 'M12_FME' in c]
    male_pct = fem_pct = pd.NA
    if male_cols and female_cols:
        male = float(analysis_df[male_cols].sum(axis=1, numeric_only=True).mean())
        fem  = float(analysis_df[female_cols].sum(axis=1, numeric_only=True).mean())
        tot = male + fem
        if tot:
            male_pct = 100 * male / tot
            fem_pct  = 100 - male_pct

    # 2) 성별 (업종 평균)
    cat_male = cat_fem = pd.NA
    if isinstance(gender_ratio, pd.DataFrame) and store_category in gender_ratio.index:
        gr = gender_ratio.loc[store_category]
        s = float(gr.get('남성비율', float('nan')))
        f = float(gr.get('여성비율', float('nan')))
        tot = s + f
        if tot > 0:
            cat_male = 100 * s / tot
            cat_fem  = 100 - cat_male

    catalog["GENDER_M"] = {
        "지표": "성별 - 남",
        "가게(%)": None if pd.isna(male_pct) else float(round(_clamp_pct(male_pct), 1)),
        "업종 평균(%)": None if pd.isna(cat_male) else float(round(_clamp_pct(cat_male), 1)),
        "참고": ""
    }
    catalog["GENDER_F"] = {
        "지표": "성별 - 여",
        "가게(%)": None if pd.isna(fem_pct) else float(round(_clamp_pct(fem_pct), 1)),
        "업종 평균(%)": None if pd.isna(cat_fem) else float(round(_clamp_pct(cat_fem), 1)),
        "참고": ""
    }

    # 3) 방문
    visit_cols = ['MCT_UE_CLN_NEW_RAT','MCT_UE_CLN_REU_RAT']
    new_avg = reu_avg = pd.NA
    if all(c in analysis_df.columns for c in visit_cols):
        new_avg = pd.to_numeric(analysis_df[visit_cols[0]], errors="coerce").mean()
        reu_avg = pd.to_numeric(analysis_df[visit_cols[1]], errors="coerce").mean()

    cat_new = cat_reu = pd.NA
    if isinstance(visit_ratio, pd.DataFrame) and store_category in visit_ratio.index:
        vr = visit_ratio.loc[store_category]
        cat_new = float(vr.get('MCT_UE_CLN_NEW_RAT', float('nan')))
        cat_reu = float(vr.get('MCT_UE_CLN_REU_RAT', float('nan')))

    catalog["VISIT_NEW"] = {
        "지표": "방문 - 신규",
        "가게(%)": None if pd.isna(new_avg) else float(round(new_avg, 1)),
        "업종 평균(%)": None if pd.isna(cat_new) else float(round(cat_new, 1)),
        "참고": ""
    }
    catalog["VISIT_REU"] = {
        "지표": "방문 - 재방문",
        "가게(%)": None if pd.isna(reu_avg) else float(round(reu_avg, 1)),
        "업종 평균(%)": None if pd.isna(cat_reu) else float(round(cat_reu, 1)),
        "참고": ""
    }

    # 4) 배달
    deli_val = pd.NA
    note = ""
    if store_category in delivery_ratio.index:
        deli_val = float(delivery_ratio.loc[store_category])
        rank = int(delivery_ratio.index.get_loc(store_category) + 1)
        total = int(len(delivery_ratio))
        note = f"업종 {rank}/{total}위"
    catalog["DELIVERY"] = {
        "지표": "배달 매출 비중",
        "가게(%)": None if pd.isna(deli_val) else float(round(deli_val, 1)),
        "업종 평균(%)": None,
        "참고": note
    }

    return catalog


def evidence_table_from_keys(catalog: Dict[str, Dict[str, Any]], keys: List[str]) -> pd.DataFrame:
    rows = [catalog[k] for k in keys if k in catalog]
    if not rows:
        return pd.DataFrame(columns=["지표","가게(%)","업종 평균(%)","참고"])
    df = pd.DataFrame(rows, columns=["지표","가게(%)","업종 평균(%)","참고"])
    df["가게(%)"] = pd.to_numeric(df["가게(%)"], errors="coerce")
    df["업종 평균(%)"] = pd.to_numeric(df["업종 평균(%)"], errors="coerce")
    return df


def metric_catalog_to_text(catalog: Dict[str, Dict[str, Any]]) -> str:
    """LLM 프롬프트로 넘길 key-value 텍스트"""
    lines = []
    for k, v in catalog.items():
        s = "" if v["가게(%)"] is None else v["가게(%)"]
        c = "" if v["업종 평균(%)"] is None else v["업종 평균(%)"]
        extra = v.get("참고","") or ""
        lines.append(f"{k} | {v['지표']} | store={s} | cat={c} | extra={extra}")
    return "\n".join(lines)

# services/analysis.py  (추가)

import pandas as pd

def load_tourism_df(path: str = "data/final_dong.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949", errors="replace")
        
    usecols = ["관광지명", "분류", "행정동"]
    df = pd.read_csv(path)
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(f"final_dong.csv에 필요한 컬럼이 없습니다: {missing}")
    # 최소 전처리: 공백/결측 정리
    df["행정동"] = df["행정동"].astype(str).str.strip()
    df["관광지명"] = df["관광지명"].astype(str).str.strip()
    df["분류"] = df["분류"].astype(str).str.strip()
    # 중복 제거(동일 명소가 중복일 수 있으니)
    df = df.drop_duplicates(subset=["행정동", "관광지명", "분류"]).reset_index(drop=True)
    return df[usecols]


def build_tourism_context_for_dong(tour_df: pd.DataFrame, dong: str, top_k: int = 5) -> str:
    """
    해당 동의 관광지 리스트 요약 텍스트 생성.
    """
    if not isinstance(dong, str) or not dong:
        return "해당 동 정보를 확인할 수 없습니다."
    sub = tour_df.loc[tour_df["행정동"] == dong]
    if sub.empty:
        return f"{dong}에 등록된 관광지 데이터가 없습니다."
    # 상위 K만 나열 (간단 정렬: 관광지명 가나다)
    sub = sub.sort_values(["분류", "관광지명"])
    items = [f"- {row['관광지명']} ({row['분류']})" for _, row in sub.head(top_k).iterrows()]
    more = "" if len(sub) <= top_k else f"\n- 외 {len(sub)-top_k}곳"
    return f"[관광지(행정동={dong})]\n" + "\n".join(items) + more


def build_zone_profile_text(df: pd.DataFrame, hpsn: str, age_group_cols: list[str]) -> str:
    """
    상권(HPSN_MCT_BZN_CD_NM) 단위 간단 프로파일.
    - 샘플 수, 주요 업종 TOP3
    - 가능하면 연령/성별/방문 비율 평균 (컬럼이 있을 때만)
    """
    if not isinstance(hpsn, str) or not hpsn:
        return "상권 정보가 없습니다."
    zdf = df.loc[df["HPSN_MCT_BZN_CD_NM"] == hpsn]
    if zdf.empty:
        return f"상권 '{hpsn}'에 대한 데이터가 없습니다."

    n = len(zdf)
    top_cats = (
        zdf["업종명_통합"].value_counts().head(3)
        if "업종명_통합" in zdf.columns else pd.Series(dtype=int)
    )
    top_cats_txt = ", ".join([f"{k}({v}개)" for k, v in top_cats.items()]) if not top_cats.empty else "데이터 없음"

    lines = [f"[상권 프로파일: {hpsn}]",
             f"- 표본 매장 수: {n}",
             f"- 주요 업종 TOP3: {top_cats_txt}"]

    # 연령 평균 (있을 때만)
    if set(age_group_cols).issubset(zdf.columns):
        age_mean = zdf[age_group_cols].mean().round(1).to_dict()
        age_txt = ", ".join([f"{k} {v:.1f}%" for k, v in age_mean.items()])
        lines.append(f"- 평균 연령 비중: {age_txt}")

    # 성별 평균 (있을 때만: 남성/여성 비율 컬럼명이 데이터에 존재해야 함)
    gcols = [c for c in ["남성", "여성"] if c in zdf.columns]
    if len(gcols) == 2:
        gmean = zdf[gcols].mean().round(1).to_dict()
        lines.append(f"- 평균 성별 비중: 남성 {gmean.get('남성', float('nan')):.1f}%, 여성 {gmean.get('여성', float('nan')):.1f}%")

    # 방문유형 평균 (있을 때만: 신규/재방문)
    vcols = [c for c in ["신규", "재방문"] if c in zdf.columns]
    if len(vcols) == 2:
        vmean = zdf[vcols].mean().round(1).to_dict()
        lines.append(f"- 평균 방문유형: 신규 {vmean.get('신규', float('nan')):.1f}%, 재방문 {vmean.get('재방문', float('nan')):.1f}%")

    return "\n".join(lines)