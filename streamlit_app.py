# main.py
import re
import json
import platform
import textwrap
from pathlib import Path
import base64

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_searchbox import st_searchbox
import bisect

# --- Service Imports ---
from services.population import load_population, generate_population_insight
from services.analysis import (
    load_data_raw, get_analysis_data, make_store_summary, build_evidence_context_from_store,
    load_tourism_df, build_zone_profile_text, build_tourism_context_for_dong,
    # 근거 테이블/카탈로그 유틸 (사용 중)
    build_metric_catalog, metric_catalog_to_text,
    evidence_table_from_keys, evidence_table_to_markdown,
)
from services.plots import (
    plot_gender_pie, plot_visit_pie,
    plot_age_compare_bar,
    plot_radar_chart
)
from services.trend import prepare_trend_df, summarize_trend_for_category
from services.llm import (
    build_integrated_marketing_plan_prompt, build_marketing_prompt, generate_answer_with_model,
    build_two_promo_ideas_prompt)
from services.youtube import search_videos_by_query


# --- Utility Function ---
def _strip_md_strike(s: str) -> str:
    return re.sub(r'~~(.*?)~~', r'\1', s, flags=re.DOTALL)


# main.py의 _parse_promos_from_llm 함수 아래에 추가

# main.py 파일 상단에 추가해주세요.

def _parse_promos_from_llm(raw: str):
    """
    LLM 응답에서 JSON 블록과 EVIDENCE_KEYS를 안정적으로 파싱합니다.
    - JSON 문법 오류가 있어도 충돌하지 않고 빈 데이터를 반환합니다.
    """
    promos, used_keys = [], []
    json_text = None
    # 코드 블록(```json ... ```) 또는 일반 텍스트에서 JSON 부분을 찾습니다.
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
    if match:
        json_text = match.group(1)
    else:
        body = raw.split("EVIDENCE_KEYS:")[0]
        match = re.search(r"\{[\s\S]*\}", body)
        if match:
            json_text = match.group(0)

    # JSON 파싱
    if json_text:
        try:
            data = json.loads(json_text)
            promos = (data or {}).get("promos", [])
        except json.JSONDecodeError:
            promos = []
            print("⚠️ LLM이 잘못된 형식의 JSON을 반환하여 파싱에 실패했습니다.")
    
    # EVIDENCE_KEYS 파싱
    if "EVIDENCE_KEYS:" in raw:
        try:
            _, keys_part = raw.rsplit("EVIDENCE_KEYS:", 1)
            used_keys = [k.strip() for k in keys_part.strip().split(",") if k.strip()]
        except Exception:
            used_keys = []
            
    return promos, used_keys

# main.py 파일의 _parse_promos_from_llm 함수 아래에 이 코드를 추가하세요.

def _parse_plan_from_llm(raw: str):
    """
    LLM 응답에서 마케팅 플랜 JSON 블록과 EVIDENCE_KEYS를 안정적으로 파싱합니다.
    - JSON 문법 오류가 있어도 충돌하지 않고 빈 데이터를 반환합니다.
    """
    plan_data, used_keys = {}, []
    json_text = None
    
    # 코드 블록(```json ... ```) 또는 일반 텍스트에서 JSON 부분을 찾습니다.
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
    if match:
        json_text = match.group(1)
    else:
        body = raw.split("EVIDENCE_KEYS:")[0]
        match = re.search(r"\{[\s\S]*\}", body)
        if match:
            json_text = match.group(0)

    # JSON 파싱
    if json_text:
        try:
            data = json.loads(json_text)
            plan_data = (data or {}).get("marketing_plan", {})
        except json.JSONDecodeError:
            plan_data = {}
            print("⚠️ LLM이 잘못된 형식의 Plan JSON을 반환하여 파싱에 실패했습니다.")
    
    # EVIDENCE_KEYS 파싱
    if "EVIDENCE_KEYS:" in raw:
        try:
            _, keys_part = raw.rsplit("EVIDENCE_KEYS:", 1)
            used_keys = [k.strip() for k in keys_part.strip().split(",") if k.strip()]
        except Exception:
            used_keys = []
            
    return plan_data, used_keys

# main.py (수정 후)

def _md_to_html_min(text: str) -> str:
    """
    [수정됨] 간단한 MD -> HTML 변환기
    - 볼드/이탤릭 효과 제거
    - 채널 목록에서 불필요한 <br> 태그 제거
    """
    if not isinstance(text, str):
        return ""

    # HTML 이스케이프 (태그 주입 방지)
    text = (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

    # **bold** 나 *italic* 같은 마크다운 서식을 제거하고 텍스트만 남깁니다.
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # 추천 채널 목록('-'으로 시작) 뒤의 줄바꿈은 <br>로 변환하지 않습니다.
    # 대신 자연스러운 줄바꿈(\n)은 그대로 유지하여 HTML에서 렌더링되도록 합니다.
    lines = text.split('\n')
    processed_lines = []
    for i, line in enumerate(lines):
        # "- 채널이름<br>- 채널이름" 과 같은 현상을 방지
        if line.strip().startswith("-") and i > 0:
            # 이전 라인이 비어있지 않으면, <br> 대신 공백을 추가하거나 그냥 둡니다.
            if processed_lines[-1]:
                 processed_lines.append(line)
            else:
                 processed_lines.append(line)
        else:
            processed_lines.append(line)

    text = "<br>".join(processed_lines)

    # 남아있는 단독 * 는 보기 좋게 중점으로 치환
    text = text.replace("*", "•")

    return text


def make_promo_cards_html(promos: list[dict]) -> str:
    """
    1) '분석' 타입 카드는 '이유' 필드를 표시
    2) '혜택' 내용의 줄바꿈(\n)을 <br>로 변환
    """
    if not promos:
        return "<div style='color:#6b7280;'>생성된 프로모션이 없습니다.</div>"

    type_styles = {
        "상권": {"color": "#10b981", "label": "상권 기반"},
        "관광지": {"color": "#3b82f6", "label": "관광지 기반"},
        "분석": {"color": "#8b5cf6", "label": "데이터 분석"},
        "전략": {"color": "#10b981", "label": "실행 전략"},
        "default": {"color": "#6b7280", "label": "기타 제안"}
    }

    parts = ["<h3 style='margin:8px 0 12px 0;'>💡 맞춤형 프로모션 제안</h3>"]
    for p in promos:
        p_type = p.get("type", "default").strip()
        style = type_styles.get(p_type, type_styles["default"])
        color = style["color"]
        label = style["label"]

        if p_type == "분석":
            card_html = textwrap.dedent(f"""
            <div style="border-left:5px solid {color};
                        background:#f9fafb;padding:10px 14px;margin-bottom:12px;border-radius:8px;">
              <p style="margin:0;font-size:1.05rem;"><b>{p.get('title','(제목 없음)')}</b>
              <span style="font-size:0.85rem;color:gray;"> {label}</span></p>
              <p style="margin:4px 0;color:#4b5563;">🎯 {p.get('target','')}
              <br>💬 <i>{p.get('hook','')}</i></p>
              <p style="margin-top:8px;font-size:0.9rem;color:#374151;line-height:1.6;">
                <b>[이유]:</b> {p.get('reason','-')}
              </p>
            </div>
            """).strip()
        else:
            offer_html = p.get('offer', '-').replace('\n', '<br>')
            card_html = textwrap.dedent(f"""
            <div style="border-left:5px solid {color};
                        background:#f9fafb;padding:10px 14px;margin-bottom:12px;border-radius:8px;">
              <p style="margin:0;font-size:1.05rem;"><b>{p.get('title','(제목 없음)')}</b>
              <span style="font-size:0.85rem;color:gray;"> {label}</span></p>
              <p style="margin:4px 0;color:#4b5563;">🎯 {p.get('target','')}
              <br>💬 <i>{p.get('hook','')}</i></p>
              <p style="margin:0;font-size:0.9rem;color:#374151;line-height:1.6;">
                <b>• 혜택:</b><br><div style="padding-left:12px;">{offer_html}</div>
                <b>• 실행:</b> {p.get('channel','-')} · {p.get('timing','-')}<br>
                <b>• 지표:</b> {p.get('kpi','-')}
              </p>
              <p style="margin-top:6px;font-size:0.9rem;color:#6b7280;">
                🔎 {p.get('rationale','-')}
              </p>
            </div>
            """).strip()

        parts.append(card_html)

    return "\n".join(parts)


# ===== plan 전용 렌더 헬퍼 =====
def render_plan_block(markdown_text: str):
    """통합 마케팅 플랜 본문을 예쁜 카드 UI로 렌더"""
    html = f"""
    <div style="
        background-color:#fdfdfd;
        border:1px solid #e5e7eb;
        border-radius:12px;
        padding:20px 24px;
        box-shadow:0 2px 5px rgba(0,0,0,0.04);
        line-height:1.65;
        font-size:15px;
    ">
      <h3 style="margin-top:0; color:#111827; font-weight:800;">📣 SNS 채널 마케팅 추천 전략</h3>
      <div style="color:#1f2937;">{_md_to_html_min(markdown_text)}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_youtube_block(query: str, videos: list[tuple[str, str]]):
    """유튜브 검색 결과를 카드형 UI로 렌더링"""
    html = f"""
    <div style="background:#fff8e1;padding:18px 20px;margin-top:14px;
                border-left:6px solid #fbbf24;border-radius:8px;">
        <h4 style="margin:0 0 8px 0;font-weight:700;color:#374151;">
            📺 참고 유튜브 영상
        </h4>
        <p style="margin:4px 0 10px 0;color:#4b5563;font-size:0.9rem;">
            ('{query}' 검색 결과입니다.)
        </p>
        <ul style="margin:0;padding-left:18px;font-size:0.9rem;">
    """
    if videos:
        for title, link in videos:
            html += f"<li><a href='{link}' target='_blank' style='color:#2563eb;text-decoration:none;'>{title}</a></li>"
    else:
        html += "<li>관련된 영상을 찾지 못했습니다.</li>"

    html += """
        </ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_plan_block_html(body_md: str) -> str:
    """마케팅 플랜 본문을 카드형 HTML로 변환해서 문자열로 반환"""
    return f"""
    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;
                padding:22px 24px;margin:12px 0 10px;">
      <div style="font-weight:900;
                  font-size:1.5rem;  
                  color:#111827;
                  margin:0 0 16px 0;
                  display:flex;
                  align-items:center;">
        <span style="font-size:1.8rem;margin-right:8px;">💡</span> 맞춤형 프로모션 제안
      </div>
      <div style="line-height:1.75;font-size:0.96rem;color:#374151;">
        {body_md}
      </div>
    </div>
    """


def render_youtube_block_html(query: str, videos: list[tuple[str, str]]) -> str:
    """유튜브 검색 결과 카드형 HTML 문자열 반환 (링크 클릭 가능)"""
    items_html = ""
    if videos:
        for title, link in videos:
            items_html += (
                f"<li style='margin:6px 0;'>"
                f"<a href='{link}' target='_blank' style='color:#2563eb;text-decoration:none;'>{title}</a>"
                f"</li>"
            )
    else:
        items_html = "<li style='margin:6px 0;'>관련된 영상을 찾지 못했습니다.</li>"

    return f"""
    <div style="background:#fff8e1;border:1px solid #fde68a;border-left:6px solid #fbbf24;
                border-radius:12px;padding:16px 18px;margin-top:10px;">
      <div style="font-weight:700;color:#92400e;margin:0 0 8px 0;">📺 참고 유튜브 영상</div>
      <ul style="margin:0;padding-left:18px;font-size:0.95rem;color:#374151;">
        {items_html}
      </ul>
    </div>
    """


# ---------------- Base UI ----------------
st.set_page_config(layout="wide", page_title="가게 데이터 분석 대시보드")
load_dotenv()

# 스크롤 깃발 처리
if st.session_state.get('scroll_to_top', False):
    components.html(
        """
        <script>
            setTimeout(function() {
                window.parent.window.scrollTo({ top: 0, behavior: 'auto' });
            }, 0);
        </script>
        """,
        height=0
    )
    st.session_state.pop('scroll_to_top')

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] {
    align-items: center;
}
.metric-card {
    border-radius: 8px;
    padding: 10px;
    background-color: #ffffff;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: center;
}
.metric-card .label {
    text-align: center;
    font-size: 15px;
    font-weight: 700;
    color: #4f4f4f;
    margin-bottom: 6px;
}
.metric-card .value-delta-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    margin-top: 5px;
}
.metric-card .value { display: none; }
.metric-card .delta {
    font-size: 18px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 4px;
}
.metric-card .delta.positive { color: #ff4b4b; }
.metric-card .delta.negative { color: #007bff; }

h5, h6, .js-plotly-plot .plotly .main-svg .gtitle {
    font-size: 20px !important;
    font-weight: 800 !important;
    fill: #222 !important;
    text-anchor: middle !important;
}
.js-plotly-plot text { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# --- Font/Theme Setup ---
font_name = "Malgun Gothic" if platform.system() == 'Windows' else (
    "AppleGothic" if platform.system() == 'Darwin' else "NanumGothic"
)
sns.set_theme(style="whitegrid", font=font_name)
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Windows':
    plt.rcParams["font.family"] = ["Malgun Gothic", "Segoe UI Emoji"]
elif platform.system() == 'Darwin':
    plt.rcParams["font.family"] = ["AppleGothic", "Apple Color Emoji"]
else:
    plt.rcParams["font.family"] = ["NanumGothic", "Noto Color Emoji"]

# ---------------- Cached Data Loading ----------------
@st.cache_data
def load_trend():
    return prepare_trend_df('data/industry_trend_timeseries.csv')

def load_data():
    """
    기존 services.analysis.load_data_raw()를 우선 사용.
    - UnicodeDecodeError 등 발생 시 utf-8-sig -> cp949 순으로 직접 로딩 폴백
    """
    try:
        return load_data_raw('data/merged_data.csv')
    except UnicodeDecodeError:
        try:
            return pd.read_csv('data/merged_data.csv', encoding='utf-8-sig')
        except Exception:
            return pd.read_csv('data/merged_data.csv', encoding='cp949', errors='ignore')
    except Exception:
        try:
            return pd.read_csv('data/merged_data.csv', encoding='utf-8-sig')
        except Exception:
            return pd.read_csv('data/merged_data.csv', encoding='cp949', errors='ignore')

@st.cache_data
def get_analysis(df):
    return get_analysis_data(df)

@st.cache_data
def get_tourism():
    return load_tourism_df('data/final_dong.csv')

# ---------------- Session State Initialization ----------------
defaults = {'history': [], 'messages': [], 'store_name_input': "", 'selected_mct_id': None, 'current_store': ""}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Sidebar Rendering ----------------
LOGO_PATH = Path("images/logo.png")
sidebar_ph = st.sidebar.empty()

@st.cache_data
def get_image_as_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 검색기록 수정
def render_sidebar():
    with sidebar_ph.container():
        # 사이드바 버튼 및 가게 이름 스타일 수정
        st.markdown("""
        <style>
            div[data-testid="stSidebar"] button[kind="secondary"] {
                white-space: nowrap !important;
                font-size: 12.5px !important;
                padding: 4px 10px !important;
                margin-top: 6px !important;
            }
            /* ✨ 가게 이름 표시를 위한 스타일 수정 ✨ */
            .selected-store-pill {
                display: inline-block;
                background-color: #ffffff;      /* 1. 배경을 흰색으로 변경 */
                border: 1px solid #e5e7eb;      /* 2. 테두리를 추가해 구분감 부여 */
                box-shadow: 0 1px 3px rgba(0,0,0,0.06); /* 3. 입체감을 위한 그림자 효과 */
                border-radius: 999px;           /* 4. 완전히 둥근 '알약' 모양으로 변경 */
                padding: 8px 16px;
                font-weight: 600;
                color: #1f2937;
                margin: 10px 0;
            }
        </style>
        """, unsafe_allow_html=True)

        # ... (이하 promo-grid 스타일 코드는 그대로 유지) ...
        st.markdown("""
        <style>
            .promo-grid { display: grid; grid-template-columns: 1fr; gap: 12px; }
            @media (min-width: 900px) { .promo-grid { grid-template-columns: 1fr 1fr; } }
            .promo-card { border:1px solid #e6e8eb; border-radius:12px; padding:14px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.05); }
            .promo-badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.8rem; font-weight:600; }
            .badge-tour { background:#eef6ff; color:#1d4ed8; }
            .badge-zone { background:#ecfdf5; color:#047857; }
            .promo-title { font-weight:700; font-size:1.05rem; margin:0 0 6px 0; }
            .promo-meta { color:#6b7280; font-size:.9rem; margin-bottom:8px; }
            .promo-li { margin:0; padding-left:16px; }
        </style>
        """, unsafe_allow_html=True)

        # 1. 로고 표시
        if LOGO_PATH.exists():
            img_base64 = get_image_as_base64(LOGO_PATH)
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 12px;">
                <a href="?reset=true" target="_self">
                    <img src="data:image/png;base64,{img_base64}" width="140">
                </a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='sb-center sb-muted'>팀 로고는 <code>images/logo.png</code>에 넣어주세요.</div>", unsafe_allow_html=True)

        st.write("")

        # 2. 가게 이름 표시 (위치 변경)
        current_store = st.session_state.get("current_store_name")
        if current_store:
            st.markdown(
                f"<div style='text-align: center;'><span class='selected-store-pill'>🏪&nbsp; {current_store}</span></div>",
                unsafe_allow_html=True
            )
        
        # 3. 수평선 표시
        st.divider()

        # 4. 검색 기록 제목 표시
        st.markdown("<h3 class='sb-center' style='margin-bottom: 10px;'>🔎 검색 기록</h3>", unsafe_allow_html=True)

        chats = [c for t, c in st.session_state.get('history', []) if t == "chat"]
        if chats:
            for c in chats:
                display_text = c if len(c) < 25 else c[:25] + "..."
                st.markdown(f"<div style='margin-bottom: 5px;'>💬 {display_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sb-muted' style='text-align: center;'>아직 질문이 없습니다.</div>", unsafe_allow_html=True)

        st.divider()
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            if st.button("기록 모두 지우기", key="btn_clear_history"):
                st.session_state.history = [] 
                st.query_params.clear()
                st.rerun()

render_sidebar()

# --- 로고 클릭 초기화 ---
if "reset" in st.query_params:
    st.session_state.store_name_input = ""
    st.session_state.selected_mct_id = None
    st.session_state.messages = []
    st.session_state.pop("search_results", None)
    st.query_params.clear()
    st.rerun()

# ---------------- Main Page Content ----------------
st.title('🤫 우리 가게 마케팅 파트너, Ei i!')
st.caption('가게 이름을 검색하면 가게의 종합 분석과 마케팅 아이디어를 얻을 수 있습니다.')

df = load_data()
if 'DLV_SAA_RAT' in df.columns and 'M1_SME_RY_DLVR_SALE_RAT' not in df.columns:
    df['M1_SME_RY_DLVR_SALE_RAT'] = df['DLV_SAA_RAT']

trend_df = load_trend()
tour_df = get_tourism()

if df is None:
    st.error("`data/merged_data.csv` 파일을 찾을 수 없습니다.")
    st.stop()

if 'ENCODED_MCT' in df.columns:
    df['ENCODED_MCT'] = df['ENCODED_MCT'].astype(str)

(age_ratio, delivery_ratio, no_delivery, gender_ratio, visit_ratio,
 store_radar_recent_raw, industry_radar_recent_raw, store_radar_recent_scores,
 store_radar_cur_year_raw, store_radar_prev_year_raw,
 store_radar_growth_scores,
 store_radar_growth_raw,
 RADAR_METRICS) = get_analysis(df)
age_group_cols = ['20대이하','30대','40대','50대','60대이상']

# ------------------ 검색박스 ------------------
all_store_names_list = df['MCT_NM'].unique().tolist()
all_store_names_set = set(all_store_names_list)

all_store_names_list = sorted(df['MCT_NM'].unique().tolist())
all_store_names_set = set(all_store_names_list)

def search_with_exact_priority(term: str) -> list[str]:
    if not term:
        return []

    start_index = bisect.bisect_left(all_store_names_list, term)

    end_index = bisect.bisect_right(all_store_names_list, term + '\uffff')

    startswith_suggestions = all_store_names_list[start_index:end_index]

    is_exact_match = term in all_store_names_set

    if is_exact_match:
        other_suggestions = [name for name in startswith_suggestions if name != term]
        return [term] + other_suggestions
    else:
        return startswith_suggestions

selected_store_name = st_searchbox(
    search_function=search_with_exact_priority,
    placeholder="분석할 가게 이름을 입력하세요...",
    label="가게 이름 검색",
    key="store_searchbox" 
)

# 검색기록 수정
if 'selected_store_from_search' not in st.session_state:
    st.session_state.selected_store_from_search = None

if selected_store_name != st.session_state.selected_store_from_search:
    st.session_state.selected_store_from_search = selected_store_name
    
    st.session_state.current_store_name = selected_store_name 
    st.session_state.selected_mct_id = None 
    st.session_state.messages = [] 
    st.session_state.history = [] 

    st.rerun()

hero_logo_path = Path("images/logo.png")
def _load_logo_as_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

show_hero_logo = (not selected_store_name) and (st.session_state.get("selected_mct_id") is None)
if show_hero_logo and hero_logo_path.exists():
    hero_b64 = _load_logo_as_base64(hero_logo_path)
    st.markdown(
        f"""
        <style>
        .hero-logo-bg {{
            position: fixed; inset: 0; display: flex; align-items: center;
            justify-content: center; padding-top: 250px; z-index: 1000; pointer-events: none;
        }}
        .hero-logo-bg img {{
            width: 700px; max-width: 70vw; opacity: 0.04; filter: grayscale(20%);
        }}
        @media (max-width: 900px) {{ .hero-logo-bg img {{ width: 70vw; }} }}
        </style>
        <div class="hero-logo-bg"><img src="data:image/png;base64,{hero_b64}" /></div>
        """,
        unsafe_allow_html=True,
    )

if not selected_store_name:
    st.stop()

result_df = df[df['MCT_NM'] == selected_store_name]

if result_df.empty:
    st.warning('해당하는 가게를 찾을 수 없습니다.')
    st.stop()

if selected_store_name and ('store', selected_store_name) not in st.session_state.history:
    st.session_state.history.insert(0, ('store', selected_store_name))

unique_stores = result_df[['ENCODED_MCT','MCT_BSE_AR']].drop_duplicates().copy()

if 'search_results' not in st.session_state or st.session_state.search_results is None:
    if not result_df.empty:
        st.session_state.search_results = unique_stores.copy()

if len(unique_stores) > 1 and st.session_state.selected_mct_id is None:
    st.info("동일한 이름의 가게가 여러 곳 있습니다. 분석할 가게의 주소를 선택해주세요.")
    selected_address = st.selectbox(
        "가게 주소 선택",
        options=unique_stores['MCT_BSE_AR'].tolist(),
        placeholder="분석할 가게의 주소를 선택해주세요.",
        index=None
    )
    if selected_address is None:
        st.stop()
    st.session_state.selected_mct_id = unique_stores.loc[
        unique_stores['MCT_BSE_AR'] == selected_address, 'ENCODED_MCT'
    ].iloc[0]
    st.rerun()
elif st.session_state.selected_mct_id is None:
    st.session_state.selected_mct_id = unique_stores['ENCODED_MCT'].iloc[0]
    st.rerun()

selected_id = str(st.session_state.selected_mct_id)
analysis_df = df[df['ENCODED_MCT'] == selected_id]
if analysis_df.empty:
    st.warning("선택된 가게 정보를 불러오는 데 실패했습니다. 다시 검색해주세요.")
    st.stop()

store_info = analysis_df.iloc[0]
store_name = store_info['MCT_NM']
store_category = store_info['업종명_통합']
st.session_state.current_store_name = store_name

if 'search_results' in st.session_state and len(st.session_state.search_results) > 1:
    other_stores = st.session_state.search_results
    current_address = store_info['MCT_BSE_AR']
    if current_address in other_stores['MCT_BSE_AR'].tolist():
        current_index = other_stores['MCT_BSE_AR'].tolist().index(current_address)
        selected_address = st.selectbox(
            "같은 이름의 다른 가게 보기:",
            options=other_stores['MCT_BSE_AR'],
            index=current_index,
            key="store_switcher"
        )
        if selected_address != current_address:
            new_mct_id = other_stores.loc[other_stores['MCT_BSE_AR'] == selected_address, 'ENCODED_MCT'].iloc[0]
            st.session_state.selected_mct_id = new_mct_id
            st.session_state.messages = []
            st.rerun()

st.divider()
st.header(f"'{store_name}' 분석 결과")
st.caption(f"주소: {store_info['MCT_BSE_AR']}")
st.divider()

st.subheader("📝 가게 분석 요약")
summary_text = make_store_summary(analysis_df, store_category, age_ratio, delivery_ratio, age_group_cols)
st.info(summary_text)

st.divider()
RADAR_LABELS = {
    'MCT_UE_CLN_REU_RAT': '재방문 고객',
    'MCT_UE_CLN_NEW_RAT': '신규 고객',
    'M1_SME_RY_DLVR_SALE_RAT': '배달 매출',
    'M1_SME_RY_SAA_RAT': '매출 경쟁력',
    'RC_M1_AV_NP_AT_NUM': '객단가 수준'
}

LOWER_IS_BETTER_METRICS = []
radar_labels_ordered = [RADAR_LABELS.get(m, m) for m in RADAR_METRICS]

has_recent_data = (selected_id in store_radar_recent_raw.index)
st.header("📊 종합 분석 : 성과와 성장성")
# st.caption("파란색: 업종 내 최근 성과 순위 | 주황색: 업종 내 YoY 성장률 순위")

tabs = st.tabs(["업종 대비 성과", "전년 대비 성과"])

# ===================================================================
# 1️⃣ 업종 평균 대비 성과
# ===================================================================
with tabs[0]:
    st.subheader("🏪 동일 업종 대비 우리 가게의 성과 분석 (최근 3개월 기준)")

    if selected_id in store_radar_recent_raw.index and store_category in industry_radar_recent_raw.index:
        # 1) 동일 순서로 정렬 + 숫자화 + 클리핑
        store_recent = (
            pd.to_numeric(store_radar_recent_raw.loc[selected_id], errors='coerce')
            .reindex(RADAR_METRICS).astype(float).clip(0, 100)
        )
        industry_recent = (
            pd.to_numeric(industry_radar_recent_raw.loc[store_category], errors='coerce')
            .reindex(RADAR_METRICS).astype(float).clip(0, 100)
        )
    else:
        st.info("최근 3개월 업종 비교용 데이터가 부족합니다.")
        st.stop()

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        radar_fig = plot_radar_chart(
            current_year_scores=store_recent,
            previous_year_scores=industry_recent,
            labels=[RADAR_LABELS.get(m, m) for m in RADAR_METRICS],
            current_label="우리 가게 ",
            previous_label="동일 업종",
            current_color="#2563EB",   # 파랑
            previous_color="#F59E0B"   # 주황
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div style='font-weight:700; font-size:18px; color:#111827; margin-bottom:2px;'>🎯 5대 핵심 지표 비교 분석</div>
        """, unsafe_allow_html=True)

        def metric_card(title, delta_html):
            return f"""
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;
                        padding:10px 0;text-align:center;line-height:1.3;
                        box-shadow:0 1px 3px rgba(0,0,0,0.05);
                        width:150px;height:70px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-weight:600;color:#111827;font-size:13px;margin-bottom:2px;">{title}</div>
                <div style="font-weight:700;font-size:16px;">{delta_html}</div>
            </div>
            """

        metric_htmls = []
        for metric in RADAR_METRICS:
            s_val = store_recent.get(metric, np.nan)
            i_val = industry_recent.get(metric, np.nan)
            if pd.isna(s_val) or pd.isna(i_val):
                delta_html = "<span>-</span>"
            else:
                delta_pts = float(s_val) - float(i_val)  # ✅ 같은 0~100 스케일
                arrow, color = ("▲", "#ff4b4b") if delta_pts >= 0 else ("▼", "#007bff")
                delta_html = f'<div style="color:{color};">{arrow} {abs(delta_pts):.1f}p</div>'
            metric_htmls.append(metric_card(RADAR_LABELS.get(metric, metric), delta_html))

        row_spacing, col_spacing = "16px", "16px"
        html_rows = f"""
        <div style="display:flex;flex-direction:column;gap:{row_spacing};align-items:flex-start;">
            <div style="display:flex;gap:{col_spacing};">{''.join(metric_htmls[:3])}</div>
            <div style="display:flex;gap:{col_spacing};">{''.join(metric_htmls[3:])}</div>
        </div>
        """
        components.html(html_rows, height=230, scrolling=False)
        st.caption("※ 점수(0~100) 기준, 업종 평균 대비 포인트 차이")


# ===================================================================
# 2️⃣ 자체 성장 분석 (과거 평균 vs 최근 3개월)
# ===================================================================
with tabs[1]:
    st.subheader("📆 우리 가게 자체 성장 분석 (전년 하반기 VS 올해 하반기)")

    # ✅ 정규화된 집계 결과 사용
    if (store_category, selected_id) in store_radar_cur_year_raw.index and \
       (store_category, selected_id) in store_radar_prev_year_raw.index:
        recent_store_raw = store_radar_cur_year_raw.loc[(store_category, selected_id)]
        past_store_raw   = store_radar_prev_year_raw.loc[(store_category, selected_id)]
    else:
        st.info("연간 비교용 데이터가 부족합니다.")
        st.stop()

    recent_store = pd.to_numeric(recent_store_raw, errors='coerce')
    past_store   = pd.to_numeric(past_store_raw, errors='coerce')

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        radar_fig = plot_radar_chart(
            current_year_scores=recent_store,
            previous_year_scores=past_store,
            labels=[RADAR_LABELS.get(m, m) for m in RADAR_METRICS],
            current_label="올해 하반기 ",
            previous_label="전년 하반기",
            current_color="#2563EB",   # 파랑
            previous_color="#F59E0B"   # 주황
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    with col2:
        # -------------------------------
        # 제목
        # -------------------------------
        st.markdown("""
        <div style='font-weight:700; font-size:18px; color:#111827; margin-bottom:2px;'>🎯 5대 핵심 지표 비교 분석</div>
        """, unsafe_allow_html=True)

        # -------------------------------
        # 카드 HTML 생성
        # -------------------------------
        def metric_card(title, delta_html):
            return f"""
            <div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;
                        padding:10px 0;text-align:center;line-height:1.3;
                        box-shadow:0 1px 3px rgba(0,0,0,0.05);
                        width:150px;height:70px;display:flex;flex-direction:column;justify-content:center;">
                <div style="font-weight:600;color:#111827;font-size:13px;margin-bottom:2px;">{title}</div>
                <div style="font-weight:700;font-size:16px;">{delta_html}</div>
            </div>
            """

        metric_htmls = []
        for metric in RADAR_METRICS:
            cur_val  = recent_store_raw.get(metric, np.nan)
            prev_val = past_store_raw.get(metric, np.nan)

            if pd.isna(cur_val) or pd.isna(prev_val):
                delta_html = "<span>-</span>"
            else:
                # ✅ 전년 대비 점수 차이 (0~100 동일 스케일)
                delta_pts = float(cur_val) - float(prev_val)
                arrow, color = ("▲", "#ff4b4b") if delta_pts >= 0 else ("▼", "#007bff")
                delta_html = f'<div style="color:{color};">{arrow} {abs(delta_pts):.1f}p</div>'

            metric_htmls.append(metric_card(RADAR_LABELS.get(metric, metric), delta_html))

        # -------------------------------
        # 카드 정렬 (3 + 2)
        # -------------------------------
        row_spacing = "16px"
        col_spacing = "16px"

        html_rows = f"""
        <div style="display:flex;flex-direction:column;gap:{row_spacing};align-items:flex-start;">
            <div style="display:flex;gap:{col_spacing};">
                {''.join(metric_htmls[:3])}
            </div>
            <div style="display:flex;gap:{col_spacing};">
                {''.join(metric_htmls[3:])}
            </div>
        </div>
        """

        components.html(html_rows, height=230, scrolling=False)
        st.caption("※ 점수(0~100) 기준, 전년 동기 대비 포인트 차이")

st.divider()
st.subheader("📉 시각화 분석")
st.caption("세부 데이터를 확인하려면 ‘자세히 보기’를 눌러주세요.")

if "show_more_charts" not in st.session_state:
    st.session_state.show_more_charts = False

btn_label = "🔎 자세히 보기" if not st.session_state.show_more_charts else "🔽 접기"
if st.button(btn_label, key="btn_toggle_more_charts"):
    st.session_state.show_more_charts = not st.session_state.show_more_charts

if st.session_state.show_more_charts:
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            image_buffer, ok = plot_gender_pie(analysis_df)
            if ok and image_buffer:
                st.image(image_buffer)
            else:
                st.info("성별 데이터가 없습니다.")

    with col2:
        with st.container(border=True):
            image_buffer, ok = plot_visit_pie(analysis_df)
            if ok and image_buffer:
                st.image(image_buffer)
            else:
                st.info("신규/재방문 데이터가 없습니다.")

    with col3:
        with st.container(border=True):
            image_buffer = plot_age_compare_bar(analysis_df, age_ratio, store_category, age_group_cols)
            if image_buffer:
                st.image(image_buffer)
            else:
                st.info("연령대 비교 데이터가 없습니다.")

# ------------------------------------------------------------------
# 관광지 데이터 (캐시된 값)
tour_df = get_tourism()

# ---------------- Chatbot ----------------
st.divider()
st.header("🤖 마케팅 아이디어 챗봇")

# 스토어 바뀌면 대화 리셋
if st.session_state.current_store != store_name:
    st.session_state.current_store = store_name
    st.session_state.messages = []
    st.session_state["current_dong"] = store_info.get("dong", None)

# 입력/상태
pending_q = st.session_state.pop("pending_question", None)
hide_tables_now = st.session_state.pop("hide_tables_this_run", False)

new_q = st.chat_input("이 가게를 위한 마케팅 아이디어를 질문해보세요.")
if new_q:
    st.session_state.history.append(('chat', new_q))
    st.session_state.pending_question = new_q
    st.session_state.hide_tables_this_run = True
    st.rerun()

# 과거 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m.get("is_html"):
            st.markdown(m["content"], unsafe_allow_html=True)
        else:
            st.markdown(m["content"])
        if (not hide_tables_now) and m.get("table_md") and m["role"] == "assistant":
            st.caption("※ 아래 표는 **답변을 뒷받침한 근거**이며, **내부 데이터**에 기반합니다.")
            st.markdown(m["table_md"])

# 최초 안내
if not st.session_state.messages:
    init_msg = f"안녕하세요! '{store_name}' 가게의 마케팅 전략에 대해 궁금한 점을 물어보세요."
    st.session_state.messages.append({"role": "assistant", "content": init_msg})
    with st.chat_message("assistant"):
        st.markdown(init_msg)

# === 질문 처리 ===
if pending_q:
    question = pending_q
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    commercial_keywords = ["상권", "상권분석", "유동", "배후", "집객", "권역", "상업지구"]
    tour_keywords = ["관광", "관광지", "명소", "데이트", "여행", "포토", "핫플", "둘레길", "공원", "미술관", "박물관"]
    plan_keywords = ["마케팅 채널", "채널", "SNS", "sns", "소셜", "인스타", "유튜브", "YouTube", "영상"]

    want_commercial = any(k in question for k in commercial_keywords)
    want_tour = any(k in question for k in tour_keywords)

    try:
        # 1️⃣ 상권/관광 관련 질문
        trend_summary_text = summarize_trend_for_category(trend_df, store_category)

        commercial_keywords = ["상권", "상권분석", "유동", "배후", "집객", "권역", "상업지구"]
        tour_keywords = ["관광", "관광지", "명소", "데이트", "여행", "포토", "핫플", "둘레길", "공원", "미술관", "박물관"]

        want_commercial = any(k in question for k in commercial_keywords)
        want_tour = any(k in question for k in tour_keywords)

        # ✅ 현재 선택된 동/상권 정보
        dong_name = st.session_state.get("current_dong")
        hpsn = store_info.get("HPSN_MCT_BZN_CD_NM", None)

        # ---- 컨텍스트 주입 ----
        tourism_context, zone_context = "", ""

        # 상권 컨텍스트 (HPSN)
        if want_commercial and isinstance(hpsn, str) and pd.notna(hpsn) and str(hpsn).strip():
            zone_context = build_zone_profile_text(df, hpsn, age_group_cols)

        # 관광지 컨텍스트 (행정동 기반)
        if (want_commercial or want_tour) and isinstance(dong_name, str) and dong_name:
            tourism_context = build_tourism_context_for_dong(tour_df, dong_name, top_k=5)

        # 상권 정보 없음 → 관광 중심 안내
        if want_commercial and (not isinstance(hpsn, str) or pd.isna(hpsn) or not str(hpsn).strip()):
            zone_context = "선택한 매장의 상권 코드(HPSN_MCT_BZN_CD_NM)가 비어있어, 행정동 기준 관광지 정보를 중심으로 제안합니다."

        # ---- evidence_context 확장 ----
        evidence_context = build_evidence_context_from_store(
            analysis_df=analysis_df, age_group_cols=age_group_cols,
            store_category=store_category, age_ratio=age_ratio, delivery_ratio=delivery_ratio
        )
        if zone_context or tourism_context:
            evidence_context += "\n\n[상권/관광지 팩트]\n"
            if zone_context:
                evidence_context += zone_context + "\n"
            if tourism_context:
                evidence_context += tourism_context + "\n"

        # ====== 실제 프로모션 생성 로직 ======
        if want_commercial or want_tour:
            try:
                # 고객층 통계 요약
                store_age_avg = analysis_df[age_group_cols].mean()
                top_age = store_age_avg.idxmax()
                top_age_val = float(store_age_avg[top_age])
            except Exception:
                top_age, top_age_val = None, None

            new_rate, rep_rate = None, None
            for col in analysis_df.columns:
                if "신규" in str(col):
                    try: new_rate = float(analysis_df[col].mean())
                    except: pass
                if "재방문" in str(col):
                    try: rep_rate = float(analysis_df[col].mean())
                    except: pass

            # 관광지 3곳까지 추출
            tours = []
            if tourism_context:
                t_matches = re.findall(r"^-+\s*([^\n]+)", tourism_context, flags=re.M)
                t_matches = [t.strip() for t in t_matches if t.strip()]
                tours = [re.split(r"[,(]", t)[0].strip() for t in t_matches]
                tours = list(dict.fromkeys(tours))[:3]

            hpsn_name = (str(hpsn).strip() if isinstance(hpsn, str) and str(hpsn).strip() else None)

            intro_lines = [f"**{store_name}** ({store_category}) 데이터를 종합해보면,"]
            if hpsn_name:
                intro_lines.append(f"이 매장은 **{hpsn_name}** 상권의 수요에 직접적인 영향을 받습니다.")
            else:
                intro_lines.append("**행정동 내 관광 수요**를 중심으로 접근이 유효합니다.")
            if top_age:
                intro_lines.append(f"주요 고객층은 **{top_age}({top_age_val:.1f}% 내외)**이며")
            if new_rate is not None and rep_rate is not None:
                intro_lines.append(f"신규/재방문 비율은 **{new_rate:.1f}% / {rep_rate:.1f}%** 수준입니다.")
            elif new_rate is not None:
                intro_lines.append(f"신규 방문 비율은 **{new_rate:.1f}%** 수준입니다.")
            elif rep_rate is not None:
                intro_lines.append(f"재방문 비율은 **{rep_rate:.1f}%** 수준입니다.")
            if tours:
                intro_lines.append(f"인근 주요 명소로는 **{', '.join(tours)}** 등이 있습니다.")
            intro_lines.append("이 데이터를 바탕으로 실행 가능한 두 가지 프로모션 전략을 제안드립니다.")

            intro_html = _md_to_html_min(" ".join(intro_lines))

            # ---- LLM 프로모션 요청 ----
            catalog = build_metric_catalog(
                analysis_df=analysis_df, store_category=store_category,
                age_ratio=age_ratio, delivery_ratio=delivery_ratio,
                age_group_cols=age_group_cols, gender_ratio=gender_ratio,
                visit_ratio=visit_ratio
            )
            catalog_text = metric_catalog_to_text(catalog)

            with st.spinner("프로모션을 구성하고 있습니다..."):
                raw = generate_answer_with_model(
                    build_two_promo_ideas_prompt(
                        store_name=store_name,
                        store_category=store_category,
                        user_question=question,
                        evidence_context=evidence_context,
                        metric_catalog_text=catalog_text
                    ),
                    provider="gemini"
                )

            promos, used_keys = _parse_promos_from_llm(raw)

            html_parts = [textwrap.dedent(f"""
            <div style="background:#f8fafc;border:1px solid #e5e7eb;
                        border-radius:10px;padding:12px 14px;margin-bottom:10px;line-height:1.7;">
                {intro_html}
            </div>
            """)]

            if promos:
                html_parts.append("<h3 style='margin:8px 0 12px 0;'>💡 맞춤형 프로모션 제안</h3>")
                for p in promos:
                    t = (p.get("type") or "").strip()
                    color = "#10b981" if t == "상권" else "#3b82f6"
                    label = "상권 기반" if t == "상권" else "관광지 기반"
                    html_parts.append(textwrap.dedent(f"""
                    <div style="border-left:5px solid {color};background:#f9fafb;
                                padding:10px 14px;margin-bottom:12px;border-radius:8px;">
                        <p style="margin:0;font-size:1.05rem;"><b>{p.get('title','(제목 없음)')}</b>
                        <span style="font-size:0.85rem;color:gray;"> {label}</span></p>
                        <p style="margin:4px 0;color:#4b5563;">🎯 {p.get('target','')}
                        <br>💬 <i>{p.get('hook','')}</i></p>
                        <p style="margin:0;font-size:0.9rem;color:#374151;">
                            • <b>혜택:</b> {p.get('offer','-')}<br>
                            • <b>실행:</b> {p.get('channel','-')} · {p.get('timing','-')}<br>
                            • <b>지표:</b> {p.get('kpi','-')}
                        </p>
                        <p style="margin-top:6px;font-size:0.9rem;color:#6b7280;">
                            🔎 {p.get('rationale','-')}
                        </p>
                    </div>
                    """))

            evidence_df = evidence_table_from_keys(catalog, used_keys)
            evidence_md = evidence_table_to_markdown(evidence_df)
            promo_html = "\n".join(html_parts)

            # === 출력 및 세션 저장 ===
            with st.chat_message("assistant"):
                st.markdown(promo_html, unsafe_allow_html=True)
                if not evidence_df.empty:
                    st.caption("※ 아래 표는 **답변을 뒷받침한 근거**이며, 내부 데이터에 기반합니다.")
                    st.markdown(evidence_md)

            st.session_state.messages.append({
                "role": "assistant",
                "content": promo_html,
                "is_html": True,
                "table_md": evidence_md
            })
            st.session_state["hide_tables_this_run"] = False
            st.rerun()

        # main.py의 "=== 질문 처리 ===" 블록 내부를 수정

        # 2️⃣ 통합 마케팅 플랜
        elif any(k in question for k in plan_keywords):
            with st.spinner("데이터를 분석하여 마케팅 전략을 수립하는 중입니다..."):
                # ... (catalog, catalog_text, prompt 생성 부분은 기존과 동일) ...
                catalog = build_metric_catalog(
                    analysis_df=analysis_df, store_category=store_category, age_ratio=age_ratio,
                    delivery_ratio=delivery_ratio, age_group_cols=age_group_cols,
                    gender_ratio=gender_ratio, visit_ratio=visit_ratio
                )
                catalog_text = metric_catalog_to_text(catalog)

                prompt = build_integrated_marketing_plan_prompt(
                    store_name=store_name, store_category=store_category,
                    user_question=question, metric_catalog_text=catalog_text
                )
                plan_response = generate_answer_with_model(prompt, provider="gemini")
            
            # --- ⬇️ 여기가 핵심 수정 부분입니다 ⬇️ ---

            # 1. 새로 만든 JSON 파서로 응답을 파싱합니다.
            plan_data, used_keys = _parse_plan_from_llm(plan_response)

            # 2. 파싱된 데이터가 비어있을 경우에 대한 안전장치
            if not plan_data:
                st.error("마케팅 플랜을 생성하는 데 실패했습니다. 질문을 조금 더 구체적으로 해보세요.")
                st.stop()

            # 3. 파싱된 JSON 데이터를 UI에 맞게 마크다운 텍스트로 재구성합니다.
            marketing_plan_text = f"""
            🎯 타겟 고객 분석
            {plan_data.get('target_analysis', '분석 결과가 없습니다.')}

            📢 추천 마케팅 채널
            {", ".join(plan_data.get('recommended_channels', ['추천 채널이 없습니다.']))}

            🚀 홍보 실행안
            {plan_data.get('action_plan', '실행안이 없습니다.')}
            """
            
            youtube_query = plan_data.get('youtube_keyword', f"{store_category} SNS 마케팅 사례")

            # 4. 근거키 추출 및 근거 테이블 생성 (기존 로직과 유사)
            if not used_keys:
                fallback_keys = [k for k in ["VISIT_NEW","VISIT_REU","DELIVERY","AGE_30대"] if k in catalog]
                used_keys = fallback_keys[:5]
            
            evidence_df = evidence_table_from_keys(catalog, used_keys)
            evidence_md = evidence_table_to_markdown(evidence_df)

            # 5. 화면에 렌더링하고 상태 저장 (기존 로직 재사용)
            plan_html = render_plan_block_html(_md_to_html_min(marketing_plan_text)) # _md_to_html_min 추가
            with st.chat_message("assistant"):
                st.markdown(plan_html, unsafe_allow_html=True)
                if not evidence_df.empty:
                    st.caption("※ 아래 표는 **답변을 뒷받침한 근거**이며, **내부 데이터**에 기반합니다.")
                    st.markdown(evidence_md)

            st.session_state.messages.append({
                "role": "assistant", "content": plan_html,
                "is_html": True, "table_md": evidence_md
            })

            # 유튜브 카드 렌더링 (기존 로직 재사용)
            if youtube_query:
                with st.spinner(f"'{youtube_query}' 관련 유튜브 영상을 검색하는 중입니다..."):
                    log_msg, videos = search_videos_by_query(youtube_query)
                youtube_html = render_youtube_block_html(youtube_query, videos)
                with st.chat_message("assistant"):
                    st.markdown(youtube_html, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant", "content": youtube_html, "is_html": True
                })

            st.session_state["hide_tables_this_run"] = False
            st.rerun()

            # --- ⬆️ 수정 완료 ⬆️ ---

        # 3️⃣ 일반 Q&A
        else:
            with st.spinner("데이터를 분석하여 맞춤 전략을 생성하는 중입니다..."):
                evidence_context = build_evidence_context_from_store(
                    analysis_df=analysis_df, age_group_cols=age_group_cols,
                    store_category=store_category, age_ratio=age_ratio,
                    delivery_ratio=delivery_ratio
                )
                catalog = build_metric_catalog(
                    analysis_df=analysis_df, store_category=store_category, age_ratio=age_ratio,
                    delivery_ratio=delivery_ratio, age_group_cols=age_group_cols,
                    gender_ratio=gender_ratio, visit_ratio=visit_ratio
                )
                catalog_text = metric_catalog_to_text(catalog)
                trend_summary_text = summarize_trend_for_category(trend_df, store_category)

                sensitive_keywords = ["인구", "거주", "연령", "성비", "고객층", "주거"]
                proxy_keywords = ["미래 타겟", "미래타겟", "향후 고객", "예상 고객", "미래 고객층"]
                
                trigger_sensitive = any(k in question for k in sensitive_keywords)
                trigger_proxy = any(k in question for k in proxy_keywords)

                st.write("DEBUG:", question, trigger_sensitive, trigger_proxy)
                
                # --- Gemini 안전 프롬프트용 세탁 ---
                safe_question = question
                for bad in proxy_keywords:
                    # Gemini가 '인구예측'으로 오인하지 않게 표현만 바꿈
                    safe_question = safe_question.replace(bad, "향후 고객층 변화")
                for bad in sensitive_keywords:
                    # 인구/성비 등의 직접 단어 제거
                    safe_question = safe_question.replace(bad, "")
                
                # --- population 실행 (미래 타겟도 포함해서) ---
                if trigger_sensitive or trigger_proxy:
                    try:
                        df_pop = load_population()
                        dong_name_norm = st.session_state.get("current_dong")
                        if dong_name_norm:
                            population_insight = generate_population_insight(df_pop, dong_name_norm)
                            evidence_context += f"\n\n[행정동 인구 데이터 기반]\n{population_insight}"
                        else:
                            evidence_context += "\n\n[행정동 인구 데이터 기반]\n주소에서 행정동을 추출할 수 없습니다."
                    except Exception as e:
                        evidence_context += f"\n\n[행정동 인구 데이터 기반]\n인구 데이터를 불러오는 중 오류 발생: {e}"
                else:
                    evidence_context += "\n\n[행정동 인구 데이터 기반]\n주소에서 행정동을 추출할 수 없습니다."

                # ✅ 6️⃣ LLM 프롬프트 생성 및 호출
                context_prompt = build_marketing_prompt(
                    store_name=store_name, store_category=store_category,
                    age_comparison_text="", delivery_rank_str="",
                    user_question=safe_question,  # ← 정제된 질문 전달
                    trend_summary_text=trend_summary_text,
                    evidence_context=evidence_context,
                    metric_catalog_text=catalog_text
                )

                answer_text = generate_answer_with_model(context_prompt, provider="gemini")
                

                promos, used_keys = _parse_promos_from_llm(answer_text)
                final_html = make_promo_cards_html(promos)

                if not used_keys:
                    fallback_keys = [k for k in ["VISIT_NEW", "VISIT_REU", "DELIVERY", "AGE_30대"] if k in catalog]
                    used_keys = fallback_keys[:4]

                evidence_df = evidence_table_from_keys(catalog, used_keys)
                evidence_md = evidence_table_to_markdown(evidence_df)

                with st.chat_message("assistant"):
                    st.markdown(final_html, unsafe_allow_html=True)
                    if not evidence_df.empty:
                        st.caption("※ 아래 표는 **답변을 뒷받침한 근거**이며, **내부 데이터**에 기반합니다.")
                        st.markdown(evidence_md)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_html,
                    "table_md": evidence_md,
                    "is_html": True
                })
                st.session_state["hide_tables_this_run"] = False
                st.rerun()

    

    except Exception as e:
        print("❌ Chatbot block error:", e)
        with st.chat_message("assistant"):
            st.error("답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") 








