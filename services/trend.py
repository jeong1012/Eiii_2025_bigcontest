# services/trend.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple

def prepare_trend_df(path: str) -> pd.DataFrame:
    """크롤링 결과 로드 + dtype 정리."""
    df = pd.read_csv(path)
    # 컬럼 기대: ['업종명_통합', '트렌드_키워드', '기간', '검색량_비율']
    df['업종명_통합'] = df['업종명_통합'].astype(str)
    # 기간을 날짜로 파싱하되, 추세 계산용이므로 일 단위 정렬만 보장
    df['기간'] = pd.to_datetime(df['기간'])
    # 안전: 검색량_비율 숫자화
    df['검색량_비율'] = pd.to_numeric(df['검색량_비율'], errors='coerce')
    df = df.dropna(subset=['검색량_비율'])
    return df


def _trend_metrics(s: pd.Series) -> Tuple[float, float, float, float]:
    """
    입력: 시간순 정렬된 검색량 시계열 (float)
    반환: (최근평균, 최근증가폭, 전주대비, 기울기)
      - 최근평균: 최근 4주 평균
      - 최근증가폭: 최근 3개 값의 평균 - 그 이전 3개 값 평균  (데이터가 짧으면 자동 축소)
      - 전주대비: 마지막 값 - 직전 값
      - 기울기: 선형회귀 slope (x=0..n-1)
    """
    if len(s) < 3:
        last = s.iloc[-1] if len(s) else np.nan
        return last, np.nan, np.nan, np.nan

    last4_mean = s.tail(4).mean() if len(s) >= 4 else s.mean()

    # 최근증가폭 계산 (보통 3주 평균 - 직전 3주 평균)
    k = min(3, len(s)//2)  # 최소 3개 확보 어려우면 자동 축소
    recent_k = s.tail(k).mean()
    prev_k = s.tail(2*k).head(k).mean() if len(s) >= 2*k else s.head(max(1, len(s)-k)).mean()
    momentum = recent_k - prev_k

    wow = s.iloc[-1] - s.iloc[-2] if len(s) >= 2 else np.nan

    # 선형회귀 slope (간단 OLS)
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean)**2).sum()
    slope = float(((x - x_mean)*(y - y_mean)).sum() / denom) if denom > 0 else np.nan

    return float(last4_mean), float(momentum), float(wow), float(slope)


def summarize_trend_for_category(
    trend_df: pd.DataFrame,
    category: str,
    weeks_window: int = 12,
    top_k_keywords: int = 3,
) -> str:
    """
    업종(category)의 최근 추세를 텍스트로 요약.
    - 최근 N주 데이터만 사용(weeks_window)
    - 키워드별로 시계열을 만들고 지표 계산 후 상위 top_k 선택
    """
    sub = trend_df[trend_df['업종명_통합'] == category].copy()
    if sub.empty:
        return "최근 업종 트렌드 데이터 없음."

    # 최근 N주만 사용
    sub = sub.sort_values('기간')
    cutoff = sub['기간'].max() - pd.Timedelta(weeks=weeks_window)
    sub = sub[sub['기간'] >= cutoff]

    lines = []
    # 키워드별 집계 → 같은 날짜에 같은 키워드가 중복이면 평균
    for kw, g in sub.groupby('트렌드_키워드'):
        g = g.sort_values('기간')

        # 같은 날짜가 여러 행이면 평균으로 집계하여 '중복 인덱스' 제거
        s = (
            g.groupby('기간', as_index=True)['검색량_비율']
             .mean()
             .sort_index()
        )

        # 주간 스텝으로 균일화 (7일 간격). 값이 비면 앞값으로 채움
        s = s.resample('7D').ffill()
        # 또는 보간을 원하면 다음 라인을 사용:
        # s = s.resample('7D').interpolate('linear').fillna(method='bfill').fillna(method='ffill')

        last4, momentum, wow, slope = _trend_metrics(s)

        lines.append({
            "키워드": kw,
            "최근4주평균": last4,
            "모멘텀": momentum,   # 내부 이름은 유지
            "WoW": wow,
            "기울기": slope
        })

    if not lines:
        return "최근 업종 트렌드 데이터 없음."

    dfm = pd.DataFrame(lines)
    # 정렬 기준: '최근 증가폭(모멘텀)' 우선 → 기울기 → 최근 4주 평균
    dfm = dfm.sort_values(["모멘텀", "기울기", "최근4주평균"], ascending=[False, False, False]).head(top_k_keywords)

    bullets = []
    for _, r in dfm.iterrows():
        trend_dir = "↗️상승" if r["기울기"] > 0 else ("↘️하락" if r["기울기"] < 0 else "→보합")
        bullets.append(
            # 표기: WoW → '전주 대비', 모멘텀 → '최근 증가폭', 최근평균 → '최근 4주 평균'
            f"- {r['키워드']}: 최근 4주 평균 {r['최근4주평균']:.1f}, 최근 증가폭 {r['모멘텀']:+.1f}, 전주 대비 {r['WoW']:+.1f}, 추세 {trend_dir}"
        )

    header = f"[최근 업종 트렌드 Top {len(bullets)}]"
    return header + "\n" + "\n".join(bullets)
