import os

# services/llm.py
def build_marketing_prompt(
    store_name: str,
    store_category: str,
    user_question: str,
    trend_summary_text: str,
    evidence_context: str,
    metric_catalog_text: str
) -> str:
    """
    [ ✨ 최종 수정본 ✨ ]
    불필요한 인자를 모두 제거하고, LLM이 가장 명확하게 이해하도록 수정한 최종 프롬프트
    """
    trend_block = f"\n- 업종 트렌드: {trend_summary_text if trend_summary_text else '데이터 없음'}\n"

    return f"""당신은 주어진 데이터를 분석하여 마케팅 전략을 요약하는 전문 컨설턴트입니다.
반드시 아래 [JSON 출력 형식]에 맞춰 응답을 생성해야 합니다.
다른 설명, 제목, 마크다운(```) 없이 순수한 JSON 객체와 그 다음 줄의 EVIDENCE_KEYS만 출력하세요.

[JSON 출력 형식]
{{
  "promos": [
    {{
      "title": "핵심 고객 및 트렌드 분석", "type": "분석",
      "target": "[데이터를 기반으로 정의된 핵심 타겟 고객에 대해 여기에 서술]",
      "hook": "[타겟을 선정한 가장 중요한 데이터 근거를 여기에 서술]",
      "offer": "[현재 업종의 주요 트렌드를 1~2문장으로 여기에 요약]",
      "channel": "-", "timing": "-", "kpi": "-",
      "rationale": "데이터 기반의 합리적인 의사결정을 위한 핵심 정보입니다."
    }},
    {{
      "title": "맞춤형 마케팅 실행 전략", "type": "전략",
      "target": "{store_name} 가게의 모든 잠재 고객 및 기존 고객",
      "hook": "[가게에 가장 시급하고 효과적인 마케팅 액션을 한 문장으로 여기에 제시]",
      "offer": "[위 액션을 실행하기 위한 구체적인 방법 2~3가지를 여기에 상세히 서술]",
      "channel": "온/오프라인 채널 활용", "timing": "즉시 실행 가능", "kpi": "신규 고객 유입 및 매출 증대",
      "rationale": "[이 전략이 데이터 관점에서 왜 성공할 것인지에 대한 이유를 여기에 서술]"
    }}
  ]
}}

[분석할 데이터]
- 가게 정보: {store_name} ({store_category})
{trend_block}
- 핵심 고객 데이터 (METRIC CATALOG):
{metric_catalog_text}
- 기타 참고 정보:
{evidence_context}

[사용자 질문]
{user_question}

[최종 출력 규칙]
- 첫 줄: 위에 명시된 [JSON 출력 형식]을 따른 순수 JSON 객체
- 둘째 줄: `EVIDENCE_KEYS: key1,key2,key3` 형식
""".strip()

import os
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_THRESHOLD"] = "ERROR"


def generate_answer_with_model(prompt: str, provider: str = "gemini") -> str:
    provider = provider.lower()
    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model.start_chat(history=[]).send_message(prompt).text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content
    else:
        raise ValueError(f"Unsupported provider: {provider}")



# llm.py
def build_integrated_marketing_plan_prompt(
    store_name: str,
    store_category: str,
    user_question: str,
    metric_catalog_text: str
) -> str:
    """
    [JSON 출력 방식으로 수정됨]
    고객 데이터 기반으로 타겟 분석, 채널 추천, 홍보안, 유튜브 검색어까지
    한 번에 생성하는 통합 프롬프트
    """
    return f"""
당신은 데이터 기반 마케팅 전략가입니다. 제공된 데이터를 바탕으로 마케팅 기획안을 JSON 형식으로 작성해야 합니다.

[분석 및 기획 프로세스]
1.  **타겟 고객 분석**: METRIC CATALOG 데이터를 분석하여 가게의 핵심 고객층(나이, 성별 등)을 2~3문장으로 정의합니다.
2.  **마케팅 채널 추천**: 분석된 타겟 고객이 가장 많이 사용할 만한 마케팅 채널을 2~3개 추천합니다. (예: 인스타그램, 당근마켓, 유튜브)
3.  **홍보 실행안 작성**: 추천 채널 중 1~2개를 골라, 타겟 고객의 관심을 끌 구체적인 홍보 문구나 이벤트 아이디어를 포함한 실행안을 작성합니다.
4.  **참고 영상 검색어 생성**: 위 '홍보 실행안' 실행 시 참고할 만한 유튜브 영상 검색어 1개를 생성합니다.

[JSON 출력 형식]
- 반드시 아래 형식에 맞춰 순수한 JSON 객체와 EVIDENCE_KEYS만 출력하세요.
- 다른 설명, 제목, 마크다운(```)은 절대 포함하지 마세요.
{{
  "marketing_plan": {{
    "target_analysis": "[1번 프로세스 결과인 타겟 고객 분석 내용을 여기에 작성]",
    "recommended_channels": "[2번 프로세스 결과인 추천 채널을 여기에 리스트 형태로 작성. 예: [\"인스타그램\", \"유튜브\"]]",
    "action_plan": "[3번 프로세스 결과인 구체적인 홍보 실행안을 여기에 작성]",
    "youtube_keyword": "[4번 프로세스 결과인 유튜브 검색어 1개를 여기에 작성]"
  }}
}}

[데이터 및 질문]
- 가게 이름: {store_name} ({store_category})
- 사용자 질문: {user_question}
- 고객 데이터:
{metric_catalog_text}

[최종 출력 규칙]
- 첫 줄: 위에 명시된 [JSON 출력 형식]을 따른 순수 JSON 객체
- 둘째 줄: `EVIDENCE_KEYS: key1,key2,key3` 형식
""".strip()


import os

# services/llm.py
def build_marketing_prompt(
    store_name: str,
    store_category: str,
    age_comparison_text: str, # main.py와의 호환성을 위해 인자를 받습니다.
    delivery_rank_str: str,   # main.py와의 호환성을 위해 인자를 받습니다.
    user_question: str,
    trend_summary_text: str = "",
    evidence_context: str = "",
    evidence_table_md: str = "",
    metric_catalog_text: str = ""
) -> str:
    """
    [ ✨ UI 수정 요청 반영 ✨ ]
    1. 분석 카드에 '이유' 필드 추가 및 불필요한 필드 제거
    2. 전략 카드 '혜택' 부분을 줄바꿈하여 생성하도록 지시
    """
    trend_block = f"\n- 업종 트렌드: {trend_summary_text if trend_summary_text else '데이터 없음'}\n"

    return f"""당신은 주어진 데이터를 분석하여 마케팅 전략을 요약하는 전문 컨설턴트입니다.
반드시 아래 [JSON 출력 형식]에 맞춰 응답을 생성해야 합니다. 다른 설명 없이 순수한 JSON과 EVIDENCE_KEYS만 출력하세요.

[JSON 출력 형식]
{{
  "promos": [
    {{
      "title": "핵심 고객 및 트렌드 분석", "type": "분석",
      "target": "[데이터 기반 핵심 타겟 고객 정의]",
      "hook": "[타겟을 뒷받침하는 핵심 데이터 근거 제시]",
      "reason": "[타겟을 그렇게 선정한 이유를 분석 지표로 간결하게 서술]"
    }},
    {{
      "title": "맞춤형 마케팅 실행 전략", "type": "전략",
      "target": "{store_name} 가게의 모든 잠재 고객 및 기존 고객",
      "hook": "[가장 효과적인 마케팅 액션을 한 문장으로 제시]",
      "offer": "[실행 방법 2~3가지를 '1. ...\\n2. ...' 형식으로 줄바꿈하여 상세히 서술]",
      "channel": "온/오프라인 채널 활용", "timing": "즉시 실행 가능", "kpi": "신규 고객 유입 및 매출 증대",
      "rationale": "[이 전략이 데이터 관점에서 왜 성공할 것인지에 대한 이유를 서술]"
    }}
  ]
}}

[분석할 데이터]
- 가게 정보: {store_name} ({store_category})
{trend_block}
- 핵심 고객 데이터 (METRIC CATALOG):
{metric_catalog_text}
- 기타 참고 정보:
{evidence_context}

[사용자 질문]
{user_question}

[최종 출력 규칙]
- 첫 줄: 위에 명시된 [JSON 출력 형식]을 따른 순수 JSON 객체
- 둘째 줄: `EVIDENCE_KEYS: key1,key2,key3` 형식
""".strip()




import os
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_THRESHOLD"] = "ERROR"


def generate_answer_with_model(prompt: str, provider: str = "gemini") -> str:
    """
    [ ✨ 안정성 최종 개선 ✨ ]
    API 호출 시 발생하는 모든 예외를 처리하여 앱이 충돌하지 않도록 합니다.
    """
    provider = provider.lower()
    try:
        if provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("❌ ERROR: GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
                return ""
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        else:
            print(f"❌ ERROR: 지원하지 않는 LLM provider입니다: {provider}")
            return ""
    except Exception as e:
        print(f"❌ API 호출 중 심각한 오류 발생: {e}")
        return ""


def build_two_promo_ideas_prompt(
    store_name: str,
    store_category: str,
    user_question: str,
    evidence_context: str,
    metric_catalog_text: str = "",
    allowed_types: list | None = None,   # ★ 추가: 예) ["관광지"] 또는 ["관광지","상권"]
) -> str:
    """
    상권/관광 관련 질문에 대해, 가게 데이터 + 상권/관광지 팩트를 근거로
    '정확히 2개'의 실행 가능한 프로모션 아이디어를 생성하는 프롬프트.
    각 아이디어는 관광지/상권 중 더 적합한 유형(type)을 선택.
    """
    allowed_types = allowed_types or ["관광지", "상권"]
    if allowed_types == ["관광지"]:
        type_rule = "- 이번 응답에서 `type` 값은 **반드시 '관광지'만** 사용하세요. '상권'을 사용하지 마세요."
        schema_hint = '"type": "관광지"'
    else:
        type_rule = "- `type` 값은 '관광지' 또는 '상권' 중 하나."
        schema_hint = '"type": "관광지" | "상권"'

    return f"""
당신은 **데이터 기반 마케팅 전략 전문가**입니다.
아래 CONTEXT를 분석하여, 질문에 대해 **실행 가능한 프로모션 2개**를 제안하세요.
- 각 아이디어는 {schema_hint} 로 지정합니다.
- 타겟·혜택·채널·지표까지 구체적으로 작성해야 합니다.
- CONTEXT 밖의 사실/추정 금지. 데이터 속에 없는 수치/사실 금지.
- 한 문장 요약(후크) + 세부 실행계획 + 기대효과 포함.
{type_rule}
- **중요:** 출력은 **순수 JSON 1개**와 다음 줄의 `EVIDENCE_KEYS: ...` 한 줄만 포함. 코드펜스/설명문/마크다운 금지.

스키마 예시:
{{
  "promos": [
    {{
      "title": "아이디어 제목 (최대 25자)",
      "type": "관광지",
      "target": "핵심 타겟 고객 또는 방문 상황",
      "hook": "짧고 강렬한 홍보 문구",
      "offer": "구체적 혜택(이벤트/쿠폰/세트 구성 등)",
      "channel": "노출 채널 (지도앱/SNS/현수막/제휴 등)",
      "timing": "적합한 시간대, 요일, 시즌",
      "kpi": "성과 지표 (예: 신규유입, 재방문율, 전환율 등)",
      "rationale": "데이터 기반 한 줄 근거(수치 포함)"
    }},
    {{ "... 두 번째 프로모션 ..." }}
  ]
}}

BEGIN CONTEXT
[STORE]
- 가게명: {store_name}
- 업종명: {store_category}

[METRIC CATALOG]
{metric_catalog_text}

[LOCAL EVIDENCE]
{evidence_context}
END CONTEXT

[USER QUESTION]
{user_question}

(출력 형식 — 아주 중요)
1) 첫 줄: 위 스키마에 맞는 **순수 JSON 객체 1개**만 출력
2) 다음 줄: `EVIDENCE_KEYS: key1,key2,key3`
""".strip()