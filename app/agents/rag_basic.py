from datetime import date
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from app.tools import tools_basic

# 오늘 날짜
today_date = date.today().strftime("%Y-%m-%d")

# System Prompt
system_prompt = f"""
당신은 한국은행 보고서를 기반으로 정보를 제공하는 유능한 Basic RAG 에이전트입니다.

### 행동 가이드라인
1. 사용자의 질문에 대해 `search_bok_reports_basic` 도구를 사용하여 정보를 검색하세요.
2. 검색된 문서(Metadata 포함)를 바탕으로 정확하고 간결하게 답변하세요.
3. 문서의 내용을 인용할 때는 출처(문서 번호, 연도, 분기 등)를 명시하면 좋습니다.
4. 만약 검색 결과에 답이 없다면, 솔직하게 모르겠다고 답하거나 범위를 넓혀 다시 검색하라고 제안하세요.

오늘의 날짜 : {today_date}
"""

def get_agent_executor():
    # LLM & Memory
    llm = init_chat_model(model="gpt-4o", model_provider="openai")
    memory = MemorySaver()
    
    # Create Agent (LangGraph 기반 Standard Agent)
    rag_agent = create_agent(
        model=llm.bind_tools(tools_basic, parallel_tool_calls=False),
        tools=tools_basic,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    
    return rag_agent

agent_executor = get_agent_executor()
