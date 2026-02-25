from datetime import date
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from app.tools import tools_self_query

# 오늘 날짜
today_date = date.today().strftime("%Y-%m-%d")

# System Prompt
system_prompt = f"""
당신은 한국은행 보고서를 기반으로 깊이 있는 정보를 제공하는 Advanced RAG 에이전트입니다.

### 핵심 기능: Self-Querying
- 당신은 사용자의 질문에서 "연도(Year)"와 "분기(Quarter)" 정보를 파악하여 스스로 필터링 조건을 생성할 수 있습니다.
- 예: "2024년 1분기 반도체 동향 알려줘" -> Metadata 필터(year=2024, quarter=1) 자동 적용.

### 행동 가이드라인
1. 질문에 특정 시점(연도, 분기)이 포함되어 있다면, 이를 반영하여 `search_bok_reports_self_query` 도구를 호출하세요.
2. 검색된 문서(Metadata 포함)를 바탕으로 정확하고 상세하게 답변하세요.
3. 문서의 Metadata(연도, 분기)가 질문의 조건과 일치하는지 확인하며 답변을 작성하세요.

오늘의 날짜 : {today_date}
"""

def get_agent_executor():
    llm = init_chat_model(model="gpt-4o", model_provider="openai")
    memory = MemorySaver()
    
    # Self-Query Agent
    agent = create_agent(
        model=llm.bind_tools(tools_self_query, parallel_tool_calls=False),
        tools=tools_self_query,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return agent

agent_executor = get_agent_executor()
