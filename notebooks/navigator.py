import os
import sys
import json
import re
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from browser_use import Agent, Browser, ChatGoogle

# 초기 설정
load_dotenv(override=True)

# 작업 파일들이 모일 디렉토리
ARTIFACT_DIR = os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "code_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)



# ==========================================
# 동적 N계층 Blueprint 스키마
# ==========================================
class PageLayer(BaseModel):
    """하나의 탐색 계층을 표현하는 단위 블록"""
    layer_name: str = Field(
        description="이 계층의 역할 이름 (예: '기사 목록', '상품 상세')"
    )
    url_pattern: str = Field(
        description="이 계층의 URL 구조 예시 또는 진입점 URL (실제 시작 URL은 entry_urls 참조)"
    )
    selectors: dict[str, str] = Field(
        description="이 계층에서 수집할 데이터의 CSS 셀렉터 딕셔너리 (key: 필드명, value: CSS 셀렉터)"
    )
    navigate_to_next: Optional[str] = Field(
        default=None,
        description="다음 계층으로 이동하는 링크의 CSS 셀렉터. 마지막 계층이면 반드시 None."
    )
    pagination_method: Optional[str] = Field(
        default=None,
        description="페이지네이션 방식 (URL파라미터 / AJAX버튼 / 무한스크롤 / None)"
    )

    @field_validator("selectors", mode="before")
    @classmethod
    def parse_selectors(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v

    @field_validator("navigate_to_next", "pagination_method", mode="before")
    @classmethod
    def parse_none_string(cls, v):
        """LLM이 None을 문자열 "None"으로 반환하는 경우를 처리합니다."""
        if v in ("None", "null", "없음", "N/A", ""):
            return None
        return v

class NavigatorBlueprint(BaseModel):
    """Navigator가 Coder에게 전달하는 동적 N계층 크롤링 설계 도면 (단일 구조)"""
    entry_urls: list[str] = Field(
        description=(
            "크롤링을 시작할 URL 목록. "
            "구조(계층/셀렉터)가 동일하고 시작점만 다른 경우 여러 개 지정. "
            "예) 정치 섹션 URL + 사회 섹션 URL"
        )
    )
    total_layers: int = Field(
        description="탐색에 필요한 총 계층 수 (layers 리스트의 길이와 동일)"
    )
    layers: list[PageLayer] = Field(
        description="탐색 순서대로 정렬된 PageLayer 목록. layers[0]은 entry_urls 각각에 반복 적용됨."
    )
    rendering_type: str = Field(
        description="Static SSR 또는 Dynamic CSR/JS"
    )
    anti_bot_notes: str = Field(
        description="로그인 필요 여부, 팝업, 캡차, 우회 조언 등. 없으면 '없음'"
    )

class NavigatorBlueprintCollection(BaseModel):
    """Navigator가 반환하는 Blueprint 모음 (1개 이상)"""
    total_jobs: int = Field(
        description="총 Blueprint 수. 구조가 같으면 1개, 구조가 다른 사이트/섹션은 각각 1개."
    )
    blueprints: list[NavigatorBlueprint] = Field(
        description=(
            "수집 작업별 Blueprint 목록. "
            "- 구조 동일 + 시작 URL만 다름 → Blueprint 1개, entry_urls에 복수 URL "
            "- 구조가 근본적으로 다름 → Blueprint를 별도 생성하여 복수 반환"
        )
    )

@dataclass
class NavigatorContext:
    shared_browser: Any  # Browser 인스턴스를 Context로 주입


# ==========================================
# 유틸리티 함수
# ==========================================
def save_blueprints(collection: NavigatorBlueprintCollection, prefix: str):
    """NavigatorBlueprintCollection을 Blueprint 개수만큼 별도 파일로 저장"""
    saved_paths = []
    for i, bp in enumerate(collection.blueprints):
        filename = f"blueprint_{prefix}_{i+1}.json"
        filepath = os.path.join(ARTIFACT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(bp.model_dump(), f, ensure_ascii=False, indent=2)
        saved_paths.append(filepath)
        print(f"  💾 저장 완료: {filepath}")
    return saved_paths


# ==========================================
# 도구 1: get_page_structure
# ==========================================
@tool(parse_docstring=True)
async def get_page_structure(url: str, scraping_goal: str) -> str:
    """웹페이지 HTML을 내부 LLM이 직접 분석하여 CSS 셀렉터 결과만 반환합니다.
    Navigator는 HTML 원문을 볼 필요 없이 셀렉터 분석 결과만 받습니다.

    Args:
        url: 분석할 웹페이지 URL
        scraping_goal: 수집하려는 데이터 설명. 예) "기사 제목과 링크 URL", "상품명과 가격"
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage
    import re

    print(f"\n📐 [get_page_structure] {url}")
    print(f"   🎯 분석 목표: {scraping_goal}")

    browser_cfg = BrowserConfig(headless=True, java_script_enabled=True)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=15000,
        delay_before_return_html=3.0,
        wait_for_images=False,
    )

    try:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=run_cfg)
    except Exception as e:
        return f"[Error] HTML 수집 실패: {e}\n→ browse_web을 사용하세요."

    from bs4 import BeautifulSoup

    raw_html = result.html or ""
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # CSS 셀렉터를 찾는 데 전혀 필요 없는 태그들(스크립트, 스타일, SVG 아이콘 등) 싹 제거
    for tag in soup(["script", "style", "noscript", "svg", "path", "header", "footer"]):
        tag.decompose()
        
    # HTML 구조와 class, id는 그대로 살아있는 깨끗한 뼈대 추출
    structured_html = soup.prettify()

    if not structured_html.strip():
        return "[Warning] HTML이 비어 있습니다. JS 렌더링 실패 가능성.\n→ browse_web을 사용하세요."

    analysis_llm = init_chat_model("google_genai:gemini-flash-latest", temperature=0)

    analysis_prompt = f"""아래 HTML에서 "{scraping_goal}"에 해당하는 요소의 CSS 셀렉터를 찾고 JSON으로만 응답하세요.
    [분석할 HTML]
    {structured_html}
    [응답 형식 - JSON만, 다른 텍스트 없이]
    {{
    "selectors": {{
        "필드명": "CSS셀렉터"
    }},
    "samples": {{
        "필드명": ["실제 텍스트 예시 1", "실제 텍스트 예시 2", "실제 텍스트 예시 3"]
    }},
    "container": "목록 전체를 감싸는 컨테이너 셀렉터 (없으면 null)",
    "navigate_to_next": "다음 계층(상세 페이지)으로 이동하는 링크 셀렉터 (없으면 null)",
    "pagination": "페이지네이션 방식 (URL파라미터/AJAX버튼/무한스크롤/null)",
    "confidence": "high 또는 medium 또는 low",
    "note": "주의사항. 확인된 경우 '없음'"
    }}
    [셀렉터 작성 규칙]
    - tag + class/id 조합 필수: a.sa_text_title, div.article_box, #main_content
    - a, div, span 처럼 태그만 있는 셀렉터 절대 금지
    - HTML에 실제로 존재하는 class/id만 사용
    - 없는 값은 반드시 null로 표기하세요. 문자열 "None"은 사용 금지.
    - 텍스트와 URL을 모두 수집해야 하면 키를 분리하세요:
        예) "title": "a.sa_text_title"  (텍스트 추출용)
        예) "url":   "a.sa_text_title"  (href 추출용, 같은 셀렉터여도 키는 분리)
    - samples에는 HTML에서 실제로 찾은 텍스트를 기재하세요
    - confidence low면 note에 근거 명시
    """
    response = await analysis_llm.ainvoke([HumanMessage(analysis_prompt)])
    
    content = response.content
    if isinstance(content, list):
        raw = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content]).strip()
    else:
        raw = content.strip()

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            print(f"   ✅ 셀렉터 추출: {parsed.get('selectors')} / confidence={parsed.get('confidence')}")
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    print(f"   ⚠️ JSON 파싱 실패, 원문 반환")
    return raw


# ==========================================
# 도구 3: verify_selectors_with_samples
# ==========================================
@tool(parse_docstring=True)
async def verify_selectors_with_samples(url: str, selectors_json: str) -> str:
    """주어진 CSS 셀렉터들이 해당 URL의 웹페이지에서 실제로 어떤 데이터를 추출하는지 검증하고 (최대 5개 샘플 반환), 이를 통해 셀렉터의 정확성을 평가합니다. get_page_structure로 찾은 셀렉터 후보를 검증할 때 필수적으로 사용하세요.

    Args:
        url: 검증할 웹페이지 URL
        selectors_json: 검증할 셀렉터 딕셔너리를 포함하는 유효한 JSON 문자열. 예) '{"title": "a.sa_text_title", "link": "a.sa_text_title::attr(href)"}'
    """
    import json
    import re
    from playwright.async_api import async_playwright
    
    print(f"\n🔍 [verify_selectors] {url}")
    try:
        selectors_dict = json.loads(selectors_json)
    except json.JSONDecodeError:
        return "[Error] selectors_json 파라미터는 유효한 JSON 포맷이어야 합니다. 예: '{\"title\": \"a.title\"}'"
    
    print(f"   🎯 검증 대상 셀렉터: {selectors_dict}")
    results = {key: [] for key in selectors_dict.keys()}
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(2000) # JS 렌더링 대기
            
            for key, selector in selectors_dict.items():
                actual_selector = selector
                attr_name = ""
                is_attr = "::attr(" in selector
                if is_attr:
                    match = re.search(r'(.*?)::attr\((.*?)\)', selector)
                    if match:
                        actual_selector = match.group(1).strip()
                        attr_name = match.group(2).strip()
                
                elements = await page.query_selector_all(actual_selector)
                
                for el in elements[:5]: # 상위 5개 요소만
                    if is_attr and attr_name:
                        val = await el.get_attribute(attr_name)
                    else:
                        val = await el.text_content()
                    
                    if val:
                        results[key].append(val.strip())
            
            await browser.close()
            
            output = []
            for k, v in results.items():
                output.append(f"[{k}] 매칭 항목 수: {len(v)}개 | 추출된 샘플: {v}")
            return "\n".join(output)
            
    except Exception as e:
        return f"[Error] 브라우저 셀렉터 검증 중 오류 발생: {str(e)}"


# ==========================================
# 도구 2: browse_web
# ==========================================
@tool(parse_docstring=True)
async def browse_web(runtime: ToolRuntime[NavigatorContext], url: str, instruction: str) -> str:
    """실제 브라우저로 웹페이지를 방문하여 동적 인터랙션을 수행합니다.
    
    사용 시점:
    - 클릭, 스크롤, 검색어 입력, 로그인 등 인터랙션이 필요한 경우
    - get_page_structure가 "[Warning]" 또는 "[Error]"를 반환한 경우 (폴백)
    - 현재 페이지에서 이어서 대화형으로 작업하는 경우
    
    Args:
        url: 이동할 URL. 현재 페이지에서 이어서 작업하려면 빈 문자열("")을 전달하세요.
        instruction: 수행할 구체적인 작업. 원하는 결과물을 명확히 기술하세요.
    """
    print(f"\n🌐 [browse_web] {'→ ' + url if url else '현재 페이지 이어서'}")
    print(f"   📋 작업: {instruction}")
    bu_llm = ChatGoogle(model="gemini-flash-latest")
    
    if url:
        nav_prefix = (
            f"첫 번째 액션으로 반드시 navigate를 실행하여 아래 URL로 이동하세요.\n"
            f"현재 브라우저 상태와 관계없이 즉시 이동부터 시작합니다.\n\n"
            f"[이동할 URL]\n{url}\n\n"
            f"[이동 후 수행할 작업]"
        )
    else:
        nav_prefix = (
            f"현재 열려있는 페이지에서 바로 아래 작업을 수행하세요.\n"
            f"navigate 액션으로 다른 페이지로 이동하지 마세요.\n\n"
            f"[수행할 작업]"
        )
    task = f"""{nav_prefix}
    {instruction}
    [결과 보고 규칙]
    - 작업 결과를 구체적으로 보고하세요.
    - 요소를 find_elements로 찾기 어렵다면, screenshot으로 화면을 직접 확인한 뒤
    시각적으로 목표 요소를 파악하세요.
    - CSS 셀렉터가 필요한 작업이라면:
        tag + class/id 조합으로 작성하세요. 예) "a.sa_text_title", "#dic_area"
        "a", "div" 처럼 태그만 있는 셀렉터는 사용 금지.
    - 확인 불가능한 정보는 "확인 불가"로 명시하세요.
    - 작업 완료 후 현재 페이지 URL과 상태를 함께 보고하세요.
    """
    agent = Agent(task=task, llm=bu_llm, use_vision="auto", browser=runtime.context.shared_browser)
    history = await agent.run(max_steps=15)
    result = history.final_result() or "탐색 완료, 결과 반환 없음"
    print(f"\n✅ [browse_web 완료] {result[:200]}...")
    return result


# ==========================================
# Navigator 에이전트 생성
# ==========================================
NAVIGATOR_SYSTEM_PROMPT = """
당신은 웹 크롤링 파이프라인의 총괄 매니저이자 아키텍트인 'Navigator'입니다.
도구를 상황에 맞게 유연하게 사용하고,
최종적으로 Coder가 즉시 실행할 수 있는 안정적이고 정교한 크롤링 Blueprint를 설계합니다.

당신의 역할은 도구를 기계적으로 실행하는 것이 아니라,
웹 구조를 분석하고 판단하여 데이터 수집 전략(청사진)을 치밀하게 세우는 것입니다.

────────────────
[도구 역할 및 사용 전략]

■ get_page_structure(url, scraping_goal)
  - 가장 빠르고 토큰 비용이 저렴한 주력 분석 도구입니다.
  - 브라우저를 시각적으로 띄우지 않고 백그라운드에서 HTML 전체를 분석하여 CSS 셀렉터 후보를 찾아냅니다.

■ verify_selectors_with_samples(url, selectors_json)
  - [필수 사용] get_page_structure가 찾아낸 셀렉터 후보가 실제로 유효한지 검증하는 강력한 도구입니다.
  - 이 도구는 실제 브라우저를 띄워 입력받은 CSS 셀렉터를 즉시 적용해보고 최대 5개의 실제 추출된 리얼 데이터를 반환합니다.
  - 샘플 데이터 배열이 비어있거나([]), "None" 이거나, 잘못된 값이라면 그 셀렉터는 실패한 것입니다. 즉시 셀렉터를 수정하여 다시 검증하거나 다른 도구를 사용해야 합니다.

■ browse_web(runtime, url, instruction)
  - 시각적 검증과 동적 행동(클릭, 스크롤, 대기)이 필요할 때 사용하는 최후의 보루(Fallback) 도구입니다.
  - [주의] get_page_structure와 verify_selectors_with_samples를 여러 번 반복하면서 스스로 셀렉터 수정을 시도해보고, 최소 3번 이상 실패했을 때만 아주 제한적으로 이 도구를 호출하세요.
  - 언제 사용하는가?
    (1) 목표 데이터가 어느 URL에 숨어 있는지, 어느 버튼을 눌러야 나오는지 모를 때
    (2) 동적 페이지에서 특정 상호작용 후 데이터가 로드되는지 확인할 때
    (3) 팝업, 캡차, 로그인 창 등의 Anti-Bot 요소가 가로막고 있는지 검증할 때

────────────────
[Blueprint 핵심 판단 가이드: Coder에게 넘겨줄 필수 정보]
**아래 항목들은 Coder가 코드를 짜는 핵심 기준이 되므로 매우 정확하게 판단해야 합니다.**

1. rendering_type (렌더링 방식)
   - "Static SSR": URL 접속 즉시 HTML 원본에 데이터가 정적으로 포함된 경우. (BeautifulSoup 활용)
   - "Dynamic CSR/JS": 상호작용이나 대기 후에야 자바스크립트로 데이터가 채워지는 경우. (Playwright 활용)

2. pagination_method (페이지 이동 방식)
   - "URL파라미터": 2페이지 이동 시 ?page=2 처럼 URL이 변경됨
   - "AJAX버튼": URL 변경 없이 '더보기' 버튼 등으로 목록이 추가됨
   - "무한스크롤": 마우스 스크롤을 내리면 자동 로드됨
   - "None": 페이징 없음

3. 셀렉터 정밀 검증 (Crucial - 추가된 규칙!):
   - 도구(get_page_structure 또는 browse_web)가 셀렉터 후보를 알려주면, 그게 진짜 "요소 1개"를 뜻하는지, "반복되는 컨테이너"를 뜻하는지 구분하세요.
   - 단순히 `a.sa_text_title` 라고만 적으면 Coder가 이게 텍스트인지 링크인지 헷갈립니다. 
   - 반드시 텍스트 추출용 셀렉터(예: `title: "a.sa_text_title"`)와 링크 추출용 속성 셀렉터(예: `link: "a.sa_text_title::attr(href)"`)를 명확하게 분리해서 Blueprint에 적으세요.
   - 컨테이너가 존재한다면 (예: 루프를 돌아야 하는 `div.sa_text`), 부모 컨테이너 셀렉터를 별도로 명시하는 것이 가장 좋습니다.

4. anti_bot_notes (장애 요소 및 주의사항)
   - 로그인 창으로 튕기는지, 캡차가 뜨는지, 동적 팝업이 뜨는지 상세하게 적어줍니다.

────────────────
[매우 중요한 원칙]
- 당신 뒤에 있는 Coder 에이전트는 웹사이트의 HTML이나 화면을 전혀 볼 수 없는 상태입니다. 단지 당신의 Blueprint 정보에만 의존합니다.
- 애매한 값으로 대충 넘기면 파이프라인은 100% 실패합니다.
- 확신이 들 때까지 도구를 사용해 검증하고 꼼꼼하게 작성하세요.
"""

nav_model = init_chat_model("google_genai:gemini-flash-latest", temperature=0.1)
nav_checkpointer = InMemorySaver()

navigator_agent = create_agent(
    model=nav_model,
    system_prompt=NAVIGATOR_SYSTEM_PROMPT,
    context_schema=NavigatorContext,
    tools=[get_page_structure, verify_selectors_with_samples, browse_web],
    checkpointer=nav_checkpointer,
    response_format=ToolStrategy(NavigatorBlueprintCollection),
)
