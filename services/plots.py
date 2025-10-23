# services/plots.py
from __future__ import annotations
import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import io
from typing import Optional, Tuple

# -------------------------------------
# 공통 유틸: Matplotlib → 이미지 버퍼 변환
# -------------------------------------
def fig_to_buffer(fig) -> io.BytesIO:
    """Matplotlib Figure를 PNG 이미지 버퍼로 변환합니다."""
    # 여백 자동 정리
    fig.tight_layout(pad=1.0)

    buf = io.BytesIO()
    # bbox_inches='tight' 옵션을 제거해 고정 크기 유지
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf


# -------------------------------------
# 1️⃣ 성별 비율 파이차트
# -------------------------------------
def plot_gender_pie(analysis_df: pd.DataFrame) -> Tuple[Optional[io.BytesIO], bool]:
    # --- 데이터 처리 ---
    male_cols = [c for c in analysis_df.columns if 'M12_MAL' in c]
    female_cols = [c for c in analysis_df.columns if 'M12_FME' in c]
    if not male_cols or not female_cols:
        return None, False

    male_sum = analysis_df[male_cols].sum(axis=1).mean()
    female_sum = analysis_df[female_cols].sum(axis=1).mean()
    if (male_sum + female_sum) <= 0:
        return None, False

    gender_data = pd.Series([male_sum, female_sum], index=['남성', '여성'])

    # --- 시각화 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        gender_data.values,
        labels=gender_data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#66b3ff', '#ff9999'],
        textprops={'fontsize': 10}
    )
    ax.axis('equal')
    ax.set_title('손님 성별 비율', fontsize=14, pad=10)

    return fig_to_buffer(fig), True


# -------------------------------------
# 2️⃣ 신규/재방문 파이차트
# -------------------------------------
def plot_visit_pie(analysis_df: pd.DataFrame) -> Tuple[Optional[io.BytesIO], bool]:
    # --- 데이터 처리 ---
    visit_cols = ['MCT_UE_CLN_NEW_RAT', 'MCT_UE_CLN_REU_RAT']
    if not all(c in analysis_df.columns for c in visit_cols):
        return None, False

    new_visit_avg = analysis_df[visit_cols[0]].mean()
    reu_visit_avg = analysis_df[visit_cols[1]].mean()
    if (new_visit_avg + reu_visit_avg) <= 0:
        return None, False

    visit_data = pd.Series([new_visit_avg, reu_visit_avg], index=['신규', '재방문'])

    # --- 시각화 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(
        visit_data.values,
        labels=visit_data.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ffcc99', '#99ff99'],
        textprops={'fontsize': 10}
    )
    ax.axis('equal')
    ax.set_title('신규/재방문 비율', fontsize=14, pad=10)

    return fig_to_buffer(fig), True


# -------------------------------------
# 3️⃣ 가게 vs 업종 연령대 비교 막대그래프
# -------------------------------------
def plot_age_compare_bar(
    analysis_df: pd.DataFrame,
    age_ratio: pd.DataFrame,
    store_category: str,
    age_group_cols: list[str]
) -> Optional[io.BytesIO]:
    # --- 데이터 처리 ---
    store_age_avg = analysis_df[age_group_cols].mean().rename('가게')
    if store_category not in age_ratio.index:
        return None
    category_age_avg = age_ratio.loc[store_category].rename('업종평균')

    comparison_df = pd.concat([store_age_avg, category_age_avg], axis=1).reset_index()
    comparison_df = comparison_df.rename(columns={'index': '연령대'})
    comparison_melted = comparison_df.melt(id_vars='연령대', var_name='구분', value_name='비율')

    # --- 시각화 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(
        data=comparison_melted,
        x='연령대',
        y='비율',
        hue='구분',
        palette='pastel',
        ax=ax
    )
    ax.set_title('가게 vs 업종 연령대 비교', fontsize=14, pad=10)
    ax.set_ylabel('고객 비율 (%)', fontsize=10)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(title='', fontsize=10, loc='upper right')

    return fig_to_buffer(fig)


# -------------------------------------
# 4️⃣ (예비용) 기타 차트 공간 확보 가능
# -------------------------------------


# -------------------------------------
# 5️⃣ 5대 지표 비교 레이더 차트 (Plotly)
# -------------------------------------
def plot_radar_chart(current_year_scores, previous_year_scores, labels,
                     current_label='최근 성과', previous_label='성장률',
                     current_color='#2563EB', previous_color='#F59E0B'):
    import plotly.graph_objects as go
    fig = go.Figure()

    # 🔸 성장률 (아래쪽)
    fig.add_trace(go.Scatterpolar(
        r=previous_year_scores,
        theta=labels,
        fill='toself',
        name=previous_label,
        line=dict(color=previous_color, width=2),
        marker=dict(size=6, color=previous_color),
        fillcolor='rgba(245,158,11,0.25)',
        opacity=0.4
    ))

    # 🔹 최근 성과 (위쪽)
    fig.add_trace(go.Scatterpolar(
        r=current_year_scores,
        theta=labels,
        fill='toself',
        name=current_label,
        line=dict(color=current_color, width=2),
        marker=dict(size=6, color=current_color),
        fillcolor='rgba(37,99,235,0.25)',
        opacity=0.7
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        font=dict(size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.2,
            xanchor="center", x=0.5
        ),
        margin=dict(l=40, r=60, t=60, b=60)
    )
    return fig

