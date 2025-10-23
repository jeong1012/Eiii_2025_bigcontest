# services/population.py
import re
import pandas as pd
from pathlib import Path

# 동명 → 엑셀 동명(베이스) 매핑 (인구수/성비/구성비 접미사는 코드에서 붙임)
DONG_TO_BASE = {
    "왕십리2동": "왕십리제2동",
    "행당1동": "행당제1동",
    "행당2동": "행당제2동",
    "금호2, 3가동": "금호2.3가동",
    "성수1가1동": "성수1가제1동",
    "성수1가2동": "성수1가제2동",
    "성수2가1동": "성수2가제1동",
    "성수2가3동": "성수2가제3동",
}

def _norm_col(c: str) -> str:
    s = str(c).replace("\u00A0", " ").strip()
    return re.sub(r"\s+", " ", s)

def load_population(pop_path: str | Path = "data/성동_연령별인구현황_v3.xlsx") -> pd.DataFrame | None:
    pop_path = Path(pop_path)
    if not pop_path.exists():
        return None
    df = pd.read_excel(pop_path, header=0)  # 단일 헤더
    df.columns = [_norm_col(c) for c in df.columns]
    if df.columns[0] != "연령":
        df = df.rename(columns={df.columns[0]: "연령"})

    # 숫자 컬럼 변환
    targets = tuple(["인구수", "구성비", "성비"])
    num_cols = [c for c in df.columns if c.endswith(targets)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df

def _find_col(df: pd.DataFrame, base: str, suffix: str) -> str | None:
    """'성수2가제3동 인구수' 식의 실제 컬럼을 공백 정규화하여 안전 매칭"""
    want = _norm_col(f"{base} {suffix}")
    lut = {_norm_col(c): c for c in df.columns}
    return lut.get(want)

def generate_population_insight(df_pop: pd.DataFrame, dong_name: str) -> str:
    if df_pop is None:
        return "해당 행정동의 인구 데이터가 없습니다."

    base = DONG_TO_BASE.get(dong_name, dong_name)
    col = _find_col(df_pop, base, "인구수")
    if not col:
        return f"'{base} 인구수' 컬럼을 찾을 수 없습니다."

    # 행 0=계, 1=남, 2=여, 3~ = 연령대
    sub = df_pop.iloc[3:].copy()
    sub["연령"] = pd.to_numeric(sub["연령"].astype(str).str.replace("세", "", regex=False), errors="coerce")
    sub = sub.dropna(subset=["연령"]).astype({"연령": int})

    bins = [0, 19, 29, 39, 49, 200]
    labels = ["20대이하", "30대", "40대", "50대", "60대이상"]
    sub["연령대"] = pd.cut(sub["연령"], bins=bins, labels=labels, right=True, include_lowest=True)

    grouped = sub.groupby("연령대", observed=True)[col].sum(min_count=1)
    if grouped.isna().all() or grouped.sum(skipna=True) == 0:
        return "해당 동 인구 데이터에 유효한 값이 없습니다."

    ratio = grouped / grouped.sum(skipna=True) * 100
    male = pd.to_numeric(df_pop.iloc[1][col], errors="coerce")
    female = pd.to_numeric(df_pop.iloc[2][col], errors="coerce")
    total = (male or 0) + (female or 0)
    male_pct = (male / total * 100) if total else None
    female_pct = (female / total * 100) if total else None

    top_age = ratio.idxmax()
    lines = [f"📍[{dong_name}] 인구 기반 마케팅 인사이트"]
    for age in labels:
        if age in ratio.index and pd.notna(ratio.loc[age]):
            lines.append(f"- {age}: {ratio.loc[age]:.1f}%")
    if male_pct is not None:
        lines.append(f"- 남성 비율: {male_pct:.1f}%, 여성 비율: {female_pct:.1f}%")
    lines.append(f"→ {top_age}층 비중이 높아, 해당 연령층 타깃팅이 유리합니다.")
    return "\n".join(lines)
