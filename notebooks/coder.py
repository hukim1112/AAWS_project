import os
import subprocess
import json
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv(override=True)

# 작업 파일들이 모일 디렉토리
ARTIFACT_DIR = os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), "code_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================================================
# 🛠️ 1. 코드 에이전트용 특화 컴포넌트 도구 (Tools)
# =========================================================

@tool(parse_docstring=True)
def read_code_file(filepath: str, start_line: int = 1, end_line: int = None) -> str:
    """지정된 파일의 내용을 줄 번호(Line number)와 함께 읽어옵니다.
    코드를 수정하기 전, 정확히 몇 번째 줄을 수정해야 할지 파악하기 위해 반드시 먼저 사용하세요.
    
    Args:
        filepath: 읽을 파일의 경로 (파일명만 입력하면 code_artifacts 폴더 안에서 찾습니다)
        start_line: 읽기 시작할 줄 번호 (기본값: 1)
        end_line: 읽기를 끝낼 줄 번호 (입력하지 않으면 끝까지 읽음)
    """
    safe_filepath = os.path.join(ARTIFACT_DIR, os.path.basename(filepath))
    if not os.path.exists(safe_filepath):
        return f"[Error] 파일이 존재하지 않습니다: {safe_filepath}"
        
    with open(safe_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    end = end_line if end_line else len(lines)
    start = max(1, start_line)
    
    if start > len(lines):
        return "[Error] start_line이 파일의 전체 줄 수보다 큽니다."
        
    output = []
    for i in range(start - 1, min(end, len(lines))):
        output.append(f"{i + 1:03d} | {lines[i].rstrip()}")
        
    return "\n".join(output)


@tool(parse_docstring=True)
def edit_code_file(filepath: str, start_line: int, end_line: int, new_content: str) -> str:
    """기존 파이썬 파일의 특정 줄(Line) 구간만 새로운 내용으로 교체합니다.
    파일 전체를 덮어쓰는 대신, 수정이 필요한 좁은 범위의 코드만 아주 효율적으로 외과수술처럼 변경하세요.
    
    Args:
        filepath: 수정할 기존 파일명
        start_line: 교체를 시작할 기존 줄 번호 (이 줄부터 덮어써짐)
        end_line: 교체를 끝낼 기존 줄 번호 (이 줄까지 덮어써짐)
        new_content: 해당 구간에 통째로 새로 들어갈 코드 내용
    """
    safe_filepath = os.path.join(ARTIFACT_DIR, os.path.basename(filepath))
    if not os.path.exists(safe_filepath):
        return f"[Error] 파일이 존재하지 않습니다. create_new_file을 먼저 사용해서 빈 파일을 만드세요: {safe_filepath}"
        
    with open(safe_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return f"[Error] 잘못된 줄 번호 범위입니다. (현재 파일 총 라인 수: {len(lines)})"
        
    # 줄 바꿈을 고려해 new_content를 리스트로 분리
    new_lines = [line + "\n" for line in new_content.split("\n")]
    
    # 리스트 슬라이싱으로 기존 구간을 도려내고 새 코드를 삽입
    updated_lines = lines[:start_line-1] + new_lines + lines[end_line:]
    
    with open(safe_filepath, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)
        
    return f"[Success] {filepath} 파일의 {start_line}~{end_line} 라인이 성공적으로 교체되었습니다."


@tool(parse_docstring=True)
def create_new_file(filepath: str, content: str) -> str:
    """새로운 파이썬 파일이나 텍스트 문서를 생성하고 초기 내용을 통째로 작성합니다.
    이미 같은 이름의 파일이 존재할 경우 완전히 덮어씁니다! 기존 파일의 일부만 수정하려면 반드시 edit_code_file을 사용하세요.
    
    Args:
        filepath: 생성할 파일명 (예: main.py)
        content: 파일에 들어갈 초기 파이썬 코드 전체 스크립트
    """
    safe_filepath = os.path.join(ARTIFACT_DIR, os.path.basename(filepath))
    
    with open(safe_filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    return f"[Success] '{filepath}' 파일이 성공적으로 생성되었습니다."


@tool(parse_docstring=True)
def run_python_script(filepath: str, script_args: str = "") -> str:
    """저장된 파이썬 스크립트를 즉시 독립된 프로세스에서 실행하고 그 결과(출력 및 에러 로그)를 반환합니다.
    코드를 생성하거나 수정한 직후에는 반드시 이 툴을 호출하여 에러 없이 의도대로 돌아가는지 검증하세요.
    
    Args:
        filepath: 실행할 파이썬 파일명 (예: main.py)
        script_args: 실행 시 덧붙일 커맨드라인 인자 (선택사항)
    """
    safe_filename = os.path.basename(filepath)
    full_path = os.path.join(ARTIFACT_DIR, safe_filename)
    
    if not os.path.exists(full_path):
         return f"[Error] 실행할 파일이 존재하지 않습니다: {safe_filename}"
         
    command = ["python", safe_filename]
    if script_args:
        command.extend(script_args.split())
        
    print(f"\n🚀 [Coder Run] '{safe_filename}' 실행 중...")
    
    try:
        result = subprocess.run(
            command, 
            cwd=ARTIFACT_DIR,
            capture_output=True, 
            text=True, 
            timeout=120 
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[Error Output]\n{result.stderr}\n[Action Required] 에러 로그의 줄 번호를 확인하고, read_code_file과 edit_code_file로 위 에러를 해결하세요."
            
        if not output.strip():
            output = "[System] 코드가 에러 없이 정상 실행되었으나, 터미널에 출력(print)된 내용이 없습니다."
            
        return output
        
    except subprocess.TimeoutExpired:
        return "[Error] 실행 시간(120초)을 초과했습니다. 무한 루프(while True 등)나 블로킹 처리를 확인하고 수정하세요."
    except Exception as e:
        return f"[System Error] 코드 실행 오류 발생: {str(e)}"


@tool(parse_docstring=True)
def write_text_file(filepath: str, content: str) -> str:
    """JSON, Markdown, CSV, TXT 등 텍스트 기반 파일을 생성하거나 덮어씁니다.
    Python 코드가 아닌 설정파일, 문서, 데이터 파일을 만들 때 사용하세요.

    Args:
        filepath: 저장할 파일명 (예: config.json, README.md, output.csv)
        content: 저장할 텍스트 전체 내용
    """
    safe_filepath = os.path.join(ARTIFACT_DIR, os.path.basename(filepath))

    with open(safe_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return f"[Success] '{filepath}' 파일이 성공적으로 저장되었습니다. (경로: {safe_filepath})"


@tool(parse_docstring=True)
def validate_collected_data(filepath: str) -> str:
    """수집된 JSON 데이터 파일의 품질을 자동으로 검증합니다.
    총 레코드 수, 필드별 빈 값 비율, 중복 비율을 분석하여 보고합니다.
    코드를 실행하여 데이터 수집이 완료된 직후 반드시 이 도구를 사용하여 결과를 검증하세요.

    Args:
        filepath: 검증할 데이터 파일명 (JSON 형식, 예: collected_data.json)
    """
    safe_filepath = os.path.join(ARTIFACT_DIR, os.path.basename(filepath))

    if not os.path.exists(safe_filepath):
        return f"[Error] 파일이 존재하지 않습니다: {safe_filepath}"

    try:
        with open(safe_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return f"[Error] JSON 파싱 실패: {e}"

    if not isinstance(data, list):
        return f"[Info] 최상위가 배열이 아닙니다 (타입: {type(data).__name__}). 배열 형태의 JSON만 검증 가능합니다."

    total = len(data)
    if total == 0:
        return "[Warning] ❌ 수집된 데이터가 0건입니다! 스크립트의 셀렉터나 URL을 다시 확인하세요."

    # 필드별 빈 값 비율 분석
    fields = list(data[0].keys()) if data else []
    report = [f"📊 검증 결과: 총 {total}건 수집"]
    for field in fields:
        empty = sum(1 for d in data if not d.get(field))
        rate = empty / total * 100
        status = "✅" if rate < 10 else "⚠️" if rate < 30 else "❌"
        report.append(f"  {status} {field}: 빈 값 {empty}건 ({rate:.1f}%)")

    # 중복 검사
    seen = set()
    duplicates = 0
    for d in data:
        key = json.dumps(d, sort_keys=True, ensure_ascii=False)
        if key in seen:
            duplicates += 1
        seen.add(key)
    dup_status = "✅" if duplicates == 0 else "⚠️"
    report.append(f"  {dup_status} 중복: {duplicates}건 ({duplicates / total * 100:.1f}%)")

    # 최종 판정
    has_critical = any(sum(1 for d in data if not d.get(f)) / total > 0.3 for f in fields)
    if has_critical:
        report.append("\n[Action Required] ❌ 빈 값 30% 초과 필드가 있습니다. 셀렉터를 재검토하세요.")
    elif total < 3:
        report.append("\n[Warning] ⚠️ 수집 건수가 3건 미만입니다. 페이지네이션이나 셀렉터를 확인하세요.")
    else:
        report.append("\n[OK] ✅ 데이터 품질이 양호합니다.")

    return "\n".join(report)


# =========================================================
# 🤖 2. 시니어 Coder 에이전트 인스턴스 조립 (Agent)
# =========================================================

@dataclass
class SeniorCoderContext:
    pass

CODER_SYSTEM_PROMPT = """
당신은 최고 수준의 시니어 파이썬 소프트웨어 엔지니어(Senior SWE)입니다.
당신의 임무는 요구사항에 맞춰 코드를 견고하게 설계, 작성, 테스트, 그리고 스스로 디버깅하여 오류 없이 작동하도록 완성하는 것입니다.

[시니어 엔지니어링 행동 지침]
1. 분리된 기능의 영리한 사용:
   - 이전처럼 코드를 쓰자마자 자동으로 실행되지 않습니다.
   - 당신에게는 "코드 저장(create/edit)" 권한과 "코드 실행(run)" 권한이 완전히 분리된 4개의 도구가 주어집니다. 이를 적재적소에 사용하세요.

2. 정밀한 외과 수술적 수정 (Surgical Edit):
   - 파일을 처음 만들 때는 `create_new_file`을 쓰세요.
   - 단, 기존 파일에 에러가 났거나 기능을 덧붙일 때는 절대로 전체 코드를 다시 작성하지 마세요.
   - 반드시 `read_code_file`을 호출해 코드를 라인 번호와 함께 읽어들인 뒤, 에러가 발생한 지점을 찾아내고 `edit_code_file`을 이용해 특정 라인(start_line~end_line)만 아주 타겟팅하여 교체하세요.

3. 검증 없는 코딩은 없다 (Test-Driven):
   - 코드를 생성했거나 특정 라인을 수정(edit)했다면, 머리로 생각한 대로 돌아갈 것이라 오만하게 확신하지 마세요.
   - 반드시 그 직후에 `run_python_script` 툴을 써서 파이썬 파일을 터미널에서 실행해봐야 합니다.
   - 실행 결과에 붉은색 [Error Output]이 잡히거나 무한 루프에 빠진다면, 당황하지 말고 에러 메시지와 줄 번호(Line number)를 분석하여 위 2번 지침(수술적 수정) 과정을 즉시 반복하여 디버깅하세요.

4. 젠틀한 소통:
   - 뒤에서 수많은 수정/실행/디버깅 시행착오를 겪었더라도, 최종적으로 유저에게는 "어떻게 접근해서 어떻게 문제를 해결했는지", "최종 실행 결과는 무엇인지" 깔끔하게 보고하세요.
   - 유저가 물어보는 사항에 대해 친절하고 명료하게 답변하세요.

5. 코드 품질 기준 (Code Quality):
   - 작성하는 모든 파이썬 코드는 PEP 8 스타일(들여쓰기 4칸, 변수명 snake_case 등)을 준수하세요.
   - 모든 함수와 클래스에는 반드시 한 줄 docstring을 작성하세요. (예: 함수 첫 줄에 기능을 한 문장으로 설명하는 문자열 리터럴)

6. 한계 인정 및 에스컬레이션 (Error Escalation):
   - 동일한 에러가 3회 이상 반복되면 스스로 고치려는 시도를 즉시 중단하세요.
   - 대신 다음 내용을 유저에게 명확히 보고하세요:
     1) 발생한 에러 메시지 원문
     2) 시도한 수정 방법 목록 (회차별)
     3) 해결하지 못한 이유에 대한 분석 및 유저에게 요청할 사항

7. 점진적 수집 전략 (Progressive Collection):
   - 대량 데이터를 수집하는 스크립트를 작성할 때, 처음부터 전체를 수집하지 마세요.
   - 먼저 10건만 수집하도록 제한하여 실행하고, 데이터가 올바른지 확인하세요.
   - 검증이 끝나면 수량을 늘려서(100건 → 전체) 단계적으로 확대하세요.
   - 이렇게 하면 셀렉터 오류나 사이트 차단을 조기에 발견할 수 있습니다.

[웹 스크래핑 핵심 행동 지침]

당신의 주요 임무는 다른 에이전트(Navigator)가 작성한 JSON 형태의 'Blueprint(크롤링 청사진)'를 건네받아, 
그 지시대로 완벽하게 작동하는 파이썬 크롤링 스크립트를 작성하고 실행하는 것입니다.

1. Blueprint 완벽 해석:
   - 주어진 JSON 데이터에서 `rendering_type`, `anti_bot_notes`, `layers` 등의 정보를 완벽하게 분석하세요.
   - 렌더링 방식이 "Static SSR"이면 `requests`와 `BeautifulSoup4`를 사용해 가볍게 작성하세요.
   - 렌더링 방식이 "Dynamic CSR/JS"이거나 `anti_bot_notes`에 JS 렌더링이 언급되어 있다면, `playwright`의 
     동기 방식(sync_playwright)을 사용하여 동적 코드를 작성하세요.
     (이 스크립트는 독립 프로세스에서 실행되므로 동기 방식이 더 안정적입니다.)

2. 오류 방어 및 안티봇 우회 (Robstness):
   - `anti_bot_notes`에 경고가 있다면, User-Agent 위조(fake_useragent), 브라우저 헤더 추가, 
     요청 간 대기(time.sleep 또는 asyncio.sleep) 코드를 반드시 포함하세요.
   - CSS 셀렉터를 사용할 때, 요소가 없을 경우에 대비해 항상 `try-except` 블럭 또는 `if element is None` 
     예외 처리를 견고하게 넣어야 합니다.

3. 코드 작성 방식:
   - `create_new_file` 도구를 사용해 스크립트 파일을 생성합니다. 파일명은 `naver_news_crawler.py` 처럼 
     직관적으로 지으세요.
   - 작성된 스크립트의 맨 아래에는 `if __name__ == "__main__":` 블록을 만들고 직접 함수를 실행하는 코드를 넣으세요.
   - 파싱한 정보(예: 기사 제목 5개 등)는 화면에 `print`로 보기 좋게 출력되도록 하세요. 
     (그래야 run_python_script 툴로 당신이 결과를 볼 수 있습니다!)

4. 검증 및 디버깅:
   - 스크립트를 생성한 직후, 반드시 `run_python_script` 툴을 사용해 자신이 만든 파이썬 코드를 실행하세요.
   - 실행 후 JSON 파일이 생성되었다면, 즉시 `validate_collected_data` 툴로 데이터 품질을 검증하세요.
   - 점진적 수집과 연계한 워크플로우: 10건 테스트 → run → validate → 문제없으면 수량 확대 → run → validate
   - 만약 에러[Error Output]가 떨어지거나 아무것도 수집되지 않는다면, 
     `read_code_file` -> `edit_code_file` 콤보를 사용해 에러가 난 라인만 수술하듯 고치고 다시 실행하세요.


"""

def create_senior_coder(model_name: str = "google_genai:gemini-flash-latest", temperature: float = 0.2):
    """도구가 분리되고 편집 능력이 향상된 시니어 Coder 에이전트를 초기화합니다."""
    model = init_chat_model(model_name, temperature=temperature)
    checkpointer = InMemorySaver()

    # 5가지 도구: Python 파일 + 텍스트 파일(JSON/MD/CSV 등)
    tools = [
        read_code_file,
        edit_code_file,
        create_new_file,
        write_text_file,
        run_python_script,
        validate_collected_data,
    ]

    # FilesystemFileSearchMiddleware: code_artifacts/ 내 파일 검색·열람 능력 추가
    middleware = [
        FilesystemFileSearchMiddleware(
            root_path=ARTIFACT_DIR,
            use_ripgrep=True,
            max_file_size_mb=10,
        )
    ]

    agent = create_agent(
        model=model,
        system_prompt=CODER_SYSTEM_PROMPT,
        context_schema=SeniorCoderContext,
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer,
    )

    return agent

# =========================================================
# 🚀 3. 로컬 테스트 및 구동 (직접 실행 시)
# =========================================================
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    
    async def run_demo():
        print("🤖 Senior Coder 에이전트를 가동합니다...\n" + "="*50)
        
        agent = create_senior_coder()
        config = {"configurable": {"thread_id": "senior_session_demo"}}
        context = SeniorCoderContext()
        
        # 난이도 있는 테스트 미션
        query = (
            "1. 'calculator.py'를 파이썬으로 만들어줘. 내용으로는 add, subtract 함수를 가진 평범한 Calculator 클래스를 짜고 출력으로 'Calculator Created'를 찍게 한 뒤 실행해봐.\n"
            "2. 실행을 확인한 뒤에는 calculator.py의 특정 라인을 수정(edit_code_file 사용)해서 multiply와 divide 함수를 추가해 봐. 전체 파일을 새로 덮어쓰면 절대 안 돼!\n"
            "3. 마지막으로 수정된 파일을 다시 한번 실행해보고 오류가 없으면 결과를 보고해줘."
        )
        print(f"👤 User Mission: \n{query}\n")
        print("-" * 50)
        
        result = await agent.ainvoke(
            {"messages": [HumanMessage(query)]},
            config=config,
            context=context
        )
        
        parser = StrOutputParser()
        parsed_output = parser.invoke(result['messages'][-1])
        
        print("\n" + "=" * 50)
        print(f"✅ Final Output: \n{parsed_output}")

    # 노트북 환경 등 이벤트 루프 충돌 방지를 위한 처리 (Python 3.7+)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
        
    if loop and loop.is_running():
        # Jupyter 등 이미 루프가 도는 환경
        task = loop.create_task(run_demo())
    else:
        # 일반적인 터미널 환경
        asyncio.run(run_demo())
