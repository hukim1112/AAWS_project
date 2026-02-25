# LangChain Runtime Context & Memory (v2)

에이전트나 도구(Tool)에서 외부에 있는 정보나 지속적인 상태(State)가 필요할 때 사용하는 **Runtime, Context, Store, Checkpointer**의 개념과 차이점을 자세히 다룹니다.

## 1. 주요 컴포넌트의 역할 분리

LLM 앱을 효율적으로 구축하려면 다음 세 요소의 스코프(Scope)를 명확히 이해해야 합니다.

1. **`Context` (Ephemeral / Request Scope)**
   - **역할**: 매 요청(Request)이나 세션 구동 시 일시적으로 주입되는 외부 환경 값 (예: `user_id`, `api_token`, 시간 정보 등).
   - **특징**: 데이터베이스에 반영구적으로 남지 않으며 실시간 상태를 도구에 전달하는 데 쓰입니다.

2. **`Checkpointer` (Thread Scope)**
   - **역할**: 단일 대화(Thread) 내의 `messages`와 상태(State)를 로깅하고 이어가도록 만듭니다.
   - **특징**: 사용자가 "이전에 한 말 기억해?" 라고 물었을 때 대화를 이어가게 하는 단기/중기 기억 장치입니다.

3. **`Store` (Global / Cross-Thread Scope)**
   - **역할**: 여러 스레드나 세션을 넘나들며 독립적으로 저장 및 검색이 가능한 전역 데이터(사용자 프로필, 요약 문서 등).
   - **특징**: 장기 기억 장치(Long-term memory) 역할을 합니다.

4. **`ToolRuntime` / `Runtime`**
   - 위 컴포넌트들(`Context`와 `Store`)에 도구 내부에서 손쉽게 접근할 수 있도록 묶어주는 래퍼(Wrapper)입니다.

## 2. 예시: Context 객체 정의 및 Tool에서 활용

`ToolRuntime` 파라미터는 에이전트가 도구를 호출할 때 시스템에서 알아서 주입하므로, LLM의 입력 토큰을 낭비하지 않는 장점이 있습니다.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str
    user_name: str

@tool
def get_user_greeting(runtime: ToolRuntime[Context]) -> str:
    """사용자에게 맞춤형 인사를 제공합니다."""
    # Context에 안전하게 접근
    name = runtime.context.user_name
    uid = runtime.context.user_id
    
    return f"안녕하세요, {name}({uid})님!"
```

## 3. 에이전트 실행 시 Context 주입

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[get_user_greeting],
    context_schema=Context,  # 사용할 Context 스키마 등록
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [("human", "나에게 인사해줘.")]},
    # 현재 세션(스레드) 식별용
    config={"configurable": {"thread_id": "session-123"}},
    # Tools에 전달될 Context 주입
    context=Context(user_id="U100", user_name="말랑카우")
)

print(result["messages"][-1].content)
```

## 🎯 요약 (v2 핵심)
- **보안성 및 토큰 효율성 극대화**: 민감하거나 사이즈가 큰 사용자 메타데이터를 프롬프트에 하드코딩하지 않고 `Context`로 주입함으로써 보안을 강화하고 LLM 토큰 소모를 줄일 수 있습니다.
- **분리된 기억 장치 활용**: 단일 대화의 연속성은 `Checkpointer`에, 사용자 전역 환경은 `Store`에 맡기며 이 둘을 조화롭게 사용하도록 설계합니다.
