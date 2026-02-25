# LangChain Human-in-the-Loop (HITL) 패턴 (v2)

LLM 에이전트가 민감하거나 중요한 도구(결제, 파일 쓰기, DB 수정 등)를 자동 실행하는 것은 큰 리스크를 수반합니다. 도구 실행을 잠시 대기(Interrupt)시키고 사용자의 **승인(Approve), 수정(Edit), 거절(Reject)**에 따라 워크플로우를 이어가는 HITL 구조의 구체적 활용법입니다.

## 1. HumanInTheLoopMiddleware 설정 구성

미들웨어를 주입하여 도구별로 사람의 개입 권한을 세밀하게 제어할 수 있습니다.

| 설정 값 종류 | 대기(Interrupt) 발생 | Approve (승인) | Edit (수정) | Reject (거절) | 적용 의미 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `False` | ❌ 없음 | 됨 (자동) | 됨 (자동) | 됨 (자동) | 사용자 개입 없이 시스템이 그냥 실행 |
| `True` | ✅ 발생 | ⭕ 허용 | ⭕ 허용 | ⭕ 허용 | 모든 개입 권한 부여 |
| `{"allowed_decisions": [...]}` | ✅ 발생 | 명시된 것만 | 명시된 것만 | 명시된 것만 | 권한 일부 제약 (예: 수정 금지) |

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "tavily_search": True,   # 모든 권한(승인/수정/거절) 개방
        "read_file": False,      # 안전하므로 개입 없이 자동 실행
        "write_file": {          # 민감하므로 수정은 막고 승인/거절만 허용
            "allowed_decisions": ["approve", "reject"]
        },
    },
    description_prefix="🛑 사용자 승인 대기:",
)

agent = create_agent(
    model="gpt-4o",
    tools=[tavily_search, read_file, write_file],
    middleware=[hitl],
    checkpointer=InMemorySaver(), # 상태 저장을 위해 체크포인터 필수
)
```

## 2. 상태 대기 확인 (`__interrupt__`)

에이전트를 실행하면 지정된 도구 직전에 멈추게 됩니다.
```python
config = {"configurable": {"thread_id": "hitl_demo_session"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "사내 문서를 수정해서 저장해 줘."}]},
    config=config
)

if "__interrupt__" in result:
    print("시스템이 도구 실행을 위해 대기 중입니다. 판단(Decision)을 입력해주세요.")
```

## 3. Resume Commands (결정 재개 방식 3가지)

에이전트를 재시작할 때는 `Command` 래퍼 내부에 결정을 담아서 `invoke` 합니다.

### A. Approve (그대로 실행 승인)
LLM이 생성한 파라미터를 문제가 없다고 판단할 때 사용합니다.
```python
from langgraph.types import Command

agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
```

### B. Edit (파라미터 강제 수정 후 승인)
LLM이 파라미터를 잘못 생성했거나 사람이 직접 변수를 조작하고 싶을 때 씁니다.
```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "tavily_search",
                        "args": {"query": "사용자가 직접 수정한 쿼리 내용"},
                    },
                }
            ]
        }
    ),
    config=config
)
```

### C. Reject (실행 거절과 피드백 전달)
작업이 불가능하거나 위험할 때 거절하고, LLM이 오류를 바로잡도록 이유 메시지를 전달합니다.
```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": "해당 경로는 쓰기 권한이 없습니다. 다른 경로를 탐색하세요."
                }
            ]
        }
    ),
    config=config
)
```
