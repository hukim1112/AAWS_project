import sys
import os
import json

from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Load environment
from dotenv import load_dotenv
load_dotenv(override=True)

# 워커 에이전트들 임포트
from notebooks.navigator import create_navigator, NavigatorContext
from notebooks.coder import create_senior_coder, SeniorCoderContext

# 추가 유틸리티 툴스
from app.tools.utility import read_image_and_analyze, web_search_custom_tool

from browser_use import Agent, Browser, ChatGoogle

# 각 툴 호출 시 독립된 에이전트 인스턴스를 생성하여 메모리/상태 오염을 방지합니다.

# =========================================================
# 1. 하위 에이전트 인스턴스 전역 생성 (상태 유지용)
# =========================================================
# 툴 호출 때마다 새로 만들지 않고 바깥에서 인스턴스화하여 메모리/맥락(Checkpointer)을 유지합니다.
GLOBAL_NAVIGATOR_AGENT = create_navigator()
GLOBAL_CODER_AGENT = create_senior_coder()

# =========================================================
# 2. 분리된(Context Isolated) Handoff 도구 (Agents as Tools 패턴)
# =========================================================

@tool(parse_docstring=True)
async def chat_to_navigator(request: str, runtime: ToolRuntime, config: RunnableConfig, url: str = "", mode: str = "blueprint") -> str:
    """웹사이트의 구조를 분석하여 데이터를 추출할 수 있는 Blueprint(설계도)를 만들기 위해 웹탐색 전문가인 네비게이터와 대화합니다.
    사용자가 특정 크롤링을 원하거나 질문/인사가 있을 때 가장 먼저 이 도구를 사용하여 네비게이터에게 지시하세요.
    
    Args:
        request: 네비게이터에게 전달할 지시사항, 목표, 질문, 인사말 등 (예: '정치 섹션 메인 기사 5개 제목과 링크', '무신사 사이트 크롤링 어떻게 해?', '안녕하세요')
        url: 분석할 웹페이지의 기본 URL (반드시 http/https 포함). 단순 질문/대화이거나 특정 URL이 필요하지 않은 경우 빈 문자열("")로 둡니다.
        mode: 실행 모드. 청사진 생성이면 'blueprint', 단순 자연어 대화/질문/탐색이면 'chat'
    """

    prompt = f"Request: {request}\nTarget URL: {url}\nMode: {mode}"
    print(f"\n👨‍💼 [Supervisor] Navigator와 대화 중...(Mode: {mode}, URL: {url or '없음'})")


    # Runtime Context용 공유 브라우저 인스턴스 생성
    browser_instance = Browser(
        headless=False,
        disable_security=True,
        keep_alive=True,
    )

    ctx = NavigatorContext(shared_browser=browser_instance, response_mode=mode)
    
    try:
        current_thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
        # 전역으로 생성된 하위 에이전트 재사용
        result = await GLOBAL_NAVIGATOR_AGENT.ainvoke(
            {"messages": [("user", prompt)]},
            context=ctx,
            config={"configurable": {"thread_id": current_thread_id}}
        )
        return result["messages"][-1].content
    finally:
        if browser_instance:
            await browser_instance.stop()
        

@tool(parse_docstring=True)
async def chat_to_coder(task_description: str, runtime: ToolRuntime, config: RunnableConfig, blueprint_info: str = "") -> str:
    """Coder에게 파이썬 코드 작성, 실행, 디버깅 등의 작업을 지시할 때 사용합니다.
    크롤링 스크립트 기반 코딩 작업을 지시할 때는 Navigator가 생성한 Blueprint를 함께 전달하고, 단순 코딩이나 질문을 할 때는 빈 문자열("")을 넘기고 자연어로 지시하세요.
    
    Args:
        task_description: 작성할 스크립트의 코드 구현 목표 및 구체적 요구사항, 또는 코딩 관련 질문
        blueprint_info: Navigator가 찾아낸 렌더링 방식 및 대상 사이트 구조 정보(Blueprint). 웹 스크래핑 관련 지시가 아니면 빈 문자열("")로 둡니다.
    """
    
    prompt = f"다음 [Task]를 수행하세요.\n\n[Task]\n{task_description}"
    if blueprint_info:
        prompt += f"\n\n[Blueprint]\n{blueprint_info}"
        
    print(f"\n👨‍💼 [Supervisor] Coder와 대화 중...")
    
    current_thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    # 전역으로 생성된 Coder 에이전트 재사용 
    result = await GLOBAL_CODER_AGENT.ainvoke(
        {"messages": [("user", prompt)]},
        context=SeniorCoderContext(),
        config={"configurable": {"thread_id": current_thread_id}}
    )
    return result["messages"][-1].content


# =========================================================
# 2. Supervisor Agent 구성
# =========================================================

SUPERVISOR_SYSTEM_PROMPT = """
당신은 '데이터 추출 멀티에이전트 워크플로우'를 총괄하는 시니어 매니저('Supervisor') 에이전트입니다.
유저와 대화를 하면서 유저의 요구사항을 파악하고 다양한 작업을 친절하게 보조합니다.

당신에게는 다용도의 도구들과 전문 워커 에이전트 도구가 있습니다:
1. chat_to_navigator: 웹 구조를 탐색, 분석하여 Blueprint(청사진/셀렉터정보 등)를 만들어오거나, 방향성에 대해 자연어로 논의(chat 모드)할 때 사용하는 역할.
2. chat_to_coder: 확보된 Blueprint를 전달받아 실제 Python 코드를 작성, 디버깅하고 결과를 돌려주는 역할.
3. read_image_and_analyze: 이미지 파일을 읽고 내용을 분석.
4. web_search_custom_tool: DB에 없는 최신 정보나 일반 상식 검색.

[업무 방식]
1. 사용자가 특정 사이트와 데이터 수집 요구사항을 말하면, `chat_to_navigator`를 호출하여 분석을 지시하세요.
   - 단, 실제 크롤링 구조 설계(Blueprint)가 필요하면 mode="blueprint"로 호출하세요.
   - 크롤링 가능 여부나 단순 질문, 웹사이트 접근 방향성만 물어볼 때는 mode="chat"으로 호출하세요.
2. Navigator가 작업에 성공하여 렌더링 방식과 CSS 셀렉터가 포함된 Blueprint 정보를 반환하면, 당신은 이를 읽어봅니다.
3. 그 다음 이 얻어낸 Blueprint 내용과 미션 설명을 묶어서 `chat_to_coder`에게 넘겨서 코딩 및 실행을 지시하세요.
4. 당신이 스스로 코딩하거나 브라우저 액션을 직접 시도하지 마세요. 가급적 전문가를 활용해 문제를 해결해야 합니다.
5. Coder까지 작업을 완료하면, Coder가 돌려준 최종 Output을 사용자에게 알기 쉽게 (마크다운 포맷) 요약정리하여 보고하세요.
6. 복합적인 질문이나 이미지 분석/표시 요구가 있을 경우, 사용 가능한 도구를 활용하여 단계를 나누어 유연하게 처리하세요.
7. 필요한 경우 유저에게 되묻거나, 하위 에이전트에게 적극적으로 피드백을 주어 작업을 개선하세요.

[UI 및 이미지 렌더링 가이드라인]
- "이미지를 보여줘" 또는 "이 이미지 설명해줘"와 같은 요청이 있으면 이미지 관련 도구를 사용하세요.
- 사용자에게 완성된 이미지나 차트를 보여주어야 할 때는 반드시 `<Render_Image>path/to/image.png</Render_Image>` 형식을 사용하여 화면에 시각적으로 표시하세요.
"""

supervisor_model = init_chat_model("google_genai:gemini-flash-latest", temperature=0.1)
supervisor_checkpointer = InMemorySaver()

supervisor_agent = create_agent(
    model=supervisor_model,
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    tools=[chat_to_navigator, chat_to_coder, read_image_and_analyze, web_search_custom_tool],
    checkpointer=supervisor_checkpointer,
    name="supervisor_agent"
)

# app/server.py에서 agent_executor로 접근할 수 있게 alias 지정
agent_executor = supervisor_agent 
