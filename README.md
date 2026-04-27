# 🕷️ AAWS — AI Agent Web Scraper

> **AI가 웹을 읽고, 설계하고, 수집한다.**  
> 브라우저 자동화부터 멀티에이전트 협업까지, 실전 AI 크롤링 파이프라인 구축 핸즈온

---

## 🔍 프로젝트 소개

**AAWS (AI Agent Web Scraper)** 는 LLM 기반 에이전트가 웹 탐색·분석·데이터 수집을 자율적으로 수행하는 시스템을 설계하고 구현하는 핸즈온 프로젝트입니다.

전통적인 크롤링은 개발자가 직접 HTML 구조를 분석하고, 셀렉터를 찾고, 코드를 작성해야 합니다.
사이트 구조가 바뀌면 처음부터 다시 해야 하죠.

AAWS는 이 과정을 AI 에이전트에게 맡깁니다.

> *"AI 에이전트가 스스로 사이트 구조를 파악하고, 코드를 짜고, 데이터를 수집할 수 있을까?"*

이 핸즈온은 그 질문에 **"Yes"** 라고 대답하기 위한 5단계 여정입니다.  
단일 에이전트 설계에서 출발해, 에이전트들이 서로 역할을 나눠 협업하는 **자율 수집 팀**을 구축하는 것이 최종 목표입니다.

---

## 🚀 시작하기 (환경 세팅)

Codespaces 환경을 처음 열었다면, 터미널에서 다음 명령어를 실행하여 필요한 모든 패키지를 한 번에 설치하세요.

```bash
# install 폴더로 이동하여 전체 설치 스크립트 실행
cd install
bash install_all.sh
```

설치가 완료되면 Python 패키지와 디스플레이 관련 프로그램, 한글 폰트 설정이 모두 자동으로 마무리되며 스크립트 실행 권한이 부여됩니다.

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 키를 설정하세요.

```env
OPENAI_API_KEY="your-api-key"
TAVILY_API_KEY="your-api-key"
GOOGLE_API_KEY="your-api-key"
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY="lyour-api-key"
AGENTQL_API_KEY="your-api-key"
```

```
DISPLAY=":1" # github codespaces에서 VNC 서버로 브라우저를 보는 경우 설정이 필요합니다.
```

### 🔍 LangSmith 트레이싱

**[LangSmith](https://smith.langchain.com)** 는 LangChain/LangGraph 에이전트의 실행 흐름을 시각적으로 추적하고 디버깅할 수 있는 공식 모니터링 플랫폼입니다.

에이전트가 어떤 도구를 어떤 순서로 호출했는지, LLM에 어떤 프롬프트가 들어갔는지, 응답 시간은 얼마나 걸렸는지를 웹 UI에서 한눈에 확인할 수 있습니다.
실습 중 에이전트가 의도치 않은 행동을 할 때 원인을 파악하는 데 매우 유용합니다.

**설정 방법:**

1. [smith.langchain.com](https://smith.langchain.com) 에서 무료 계정 생성
2. **Settings → API Keys** 에서 API 키 발급
3. `.env` 파일에 아래 값 설정

```env
LANGCHAIN_API_KEY=lsv2_...your_key...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=AAWS_project   # 프로젝트 이름 (자유롭게 설정)
```

설정 후 노트북을 실행하면 [LangSmith 대시보드](https://smith.langchain.com)에서 실행 내역이 자동으로 기록됩니다.  
각 에이전트의 Tool 호출 체인, 토큰 사용량, 에러 위치까지 추적할 수 있어 디버깅 시간을 크게 줄여줍니다.

---

## 🖥️ VNC 서버 실행 (브라우저 시각화)

browser-use 에이전트가 실제 브라우저를 조작하는 장면을 실시간으로 확인하려면 VNC 서버를 실행하세요.  
에이전트가 스스로 마우스를 클릭하고 스크롤하는 장면을 눈으로 볼 수 있습니다.

```bash
./start_vnc.sh
```

1. Codespaces **포트(Ports)** 탭에서 **`6080` 포트**의 지구본 아이콘(Open in browser)을 클릭합니다.
2. 열린 페이지의 목록에서 **`vnc.html`** 을 클릭합니다.
3. 파란색 noVNC 화면에서 **`Connect`** 버튼을 클릭합니다.
4. 가상 데스크톱 화면에서 에이전트가 브라우저를 자율적으로 조작하는 것을 실시간으로 관찰할 수 있습니다.

---

## 📂 프로젝트 구조

```
AAWS_project/
├── notebooks/              # 핸즈온 실습 노트북 (01~05)
├── code_artifacts/         # Coder 에이전트가 생성한 코드 및 수집 결과(JSON)
├── docs/                   # 📖 LangChain 멀티에이전트 아키텍처 규칙 매뉴얼
├── app/
│   ├── agents/             # 에이전트 로직
│   │   ├── chatbot.py            # 기본 챗봇 에이전트
│   │   ├── multimodal_agent.py   # 멀티모달 에이전트
│   │   └── supervisor.py         # 데이터 수집 자동화 슈퍼바이저 에이전트
│   ├── tools/              # 에이전트 도구 모음
│   ├── server.py           # FastAPI 백엔드 서버 (에이전트 API 엔드포인트)
│   ├── client.py           # 터미널용 테스트 CLI
│   └── ui.py               # Streamlit 채팅 웹 인터페이스
├── install/                # 환경 설치 스크립트 모음
└── start_vnc.sh            # VNC + noVNC 서버 실행 스크립트
```

## 🧭 커리큘럼 구조

```
notebooks/
├── 01_BrowserUse_Basics.ipynb          # 🌐 Browser-use로 브라우저 자동화 맛보기
├── 02_The_Navigator.ipynb              # 🗺️ 웹 탐색 에이전트 (Navigator) 설계
├── 03_The_Coder.ipynb                  # 💻 코드 생성 에이전트 (Coder) 설계
├── 04_MultiAgent_Workflow.ipynb        # 🔗 Navigator + Coder 파이프라인 연결
└── 05_Supervised_MultiAgentTeam.ipynb  # 🎯 감독형 멀티에이전트 팀 구축 (최종 미션)
```

| 단계 | 노트북 | 핵심 개념 |
|:---:|--------|-----------|
| 1 | `01_BrowserUse_Basics` | Browser-use, playwright, 브라우저 자동화 |
| 2 | `02_The_Navigator` | Agent as Tool, 멀티턴 대화, shared browser |
| 3 | `03_The_Coder` | 코드 생성, 실행 피드백 루프, Blueprint 해석 |
| 4 | `04_MultiAgent_Workflow` | Navigator→Coder 파이프라인, LangGraph StateGraph |
| 5 | `05_Supervised_MultiAgentTeam` | Supervisor 패턴, 팀 자율 수행, 실전 데이터 수집 |


### 📗 01 — Browser-use 브라우저 자동화 기초

LLM이 실제 브라우저를 직접 조작할 수 있는 [browser-use](https://browser-use.com) 라이브러리를 소개합니다.  
"네이버에서 오늘 날씨 알려줘"처럼 자연어 명령 하나만으로 에이전트가 브라우저를 열고, 검색하고, 결과를 읽어오는 경험을 합니다.  
이 단계에서 브라우저 자동화의 기본 원리(Playwright, Accessibility Tree, VNC 시각화)를 이해합니다.

### 📘 02 — The Navigator: 웹 탐색 에이전트 설계

browser-use를 그대로 쓰는 것의 한계를 느끼고, 이를 **"도구(Tool)"로 감싸서** 더 스마트한 상위 에이전트에게 쥐어주는 **Agent as Tool** 패턴을 구현합니다.  
Navigator 에이전트는 사용자와 멀티턴 대화를 하며 브라우저 세션(shared browser)을 유지한 채 맥락을 이어갑니다.  
이 단계의 핵심 질문: *"에이전트가 이전에 보던 페이지를 기억하고 이어서 탐색할 수 있을까?"*

### 📙 03 — The Coder: 코드 생성 에이전트 설계

Navigator가 분석한 사이트 구조를 **Blueprint(청사진)** 형태로 전달받아, 실제로 동작하는 크롤링 코드를 작성하고 실행까지 하는 Coder 에이전트를 설계합니다.  
Coder는 코드를 실행한 뒤 오류가 나면 로그를 분석해 스스로 수정 → Python REPL 실행 피드백 루프를 통해 코드가 성공할 때까지 스스로 디버깅하는 Self-Healing 과정을 경험합니다.
이 단계의 핵심 질문: *"Blueprint만 줘도 에이전트가 작동하는 코드를 만들 수 있을까?"*

### 📕 04 — Multi-Agent Workflow: 파이프라인 연결

02와 03에서 만든 Navigator와 Coder를 하나의 흐름으로 연결합니다.  
Navigator가 Blueprint를 완성하면 자동으로 Coder에게 전달되어 수집까지 이어지는 **N계층 크롤링 파이프라인**을 구축합니다.  
LangGraph `StateGraph`로 자동화하는 실습 과제가 포함됩니다.

### 📔 05 — Supervised Multi-Agent Team: 감독형 팀 (최종 미션)

Supervisor가 Navigator와 Coder를 **팀원으로 지휘**하는 구조로 전환합니다.  
사용자는 목표만 말하면 되고, Supervisor가 "Navigator에게 분석 맡기고, 결과를 Coder에게 전달"하는 판단을 자율적으로 수행합니다.  
실전 사이트([Quotes to Scrape](http://quotes.toscrape.com))에 대해 3가지 시나리오로 팀의 자율 수행 능력을 검증합니다.

### 📖 에이전트 설계 매뉴얼 (docs/)

본 프로젝트는 LangChain과 LangGraph 기반의 멀티에이전트 아키텍처를 따릅니다.
에이전트 구현 시 참고할 수 있도록 `docs/` 폴더 내에 규칙 문서(`rule_langchain.md` 및 `LangChain/` 하위 파일)를 제공합니다.
이 문서들에는 langchain 에이전트를 구축할 때 활용할 수 있는 핵심 맥락 정보가 담겨 있습니다.
실습 과제를 해결할 때, 여러분 뿐만 아니라 코드를 작성하는 AI 에이전트에게도 이 맥락을 참조하게 하여 올바른 패턴을 스스로 이해하고 구현하도록 유도할 수 있습니다!

---

## 🏗️ 시스템 아키텍처

### 에이전트 팀 구조 (05단계 최종 형태)

```
👤 사용자
    │ "이 사이트에서 데이터 수집해줘"
    ▼
🧠 Supervisor
    ├── 🗺️ Navigator     ← crawl4ai + browser-use로 HTML 분석, Blueprint 설계
    └── 💻 Coder         ← Blueprint 해석, Playwright 코드 작성 및 실행
```

### 핵심 도구 스택

| 역할 | 도구 |
|------|------|
| LLM / 에이전트 프레임워크 | LangChain `create_agent`, LangGraph |
| 브라우저 자동화 (인터랙션) | [browser-use](https://browser-use.com) |
| HTML 수집 / 렌더링 | [crawl4ai](https://crawl4ai.com) |
| 코드 실행 | Python `subprocess` (Playwright sync) |
| 상태 관리 | LangGraph `InMemorySaver`, `StateGraph` |

---


---

## ▶️ 앱 실행 가이드

노트북 실습 외에, 완성된 에이전트를 **실제 웹 서비스처럼** 배포하고 테스트할 수 있습니다.  
모든 명령어는 **프로젝트 루트 (`AAWS_project/`)** 에서 실행하세요.

### 1. 백엔드 서버 실행

에이전트를 API로 노출하는 FastAPI 서버를 실행합니다.

```bash
python app/server.py --port 8000
```

정상 실행 시 `🚀 Server starting on http://0.0.0.0:8000` 메시지가 출력됩니다.  
`http://localhost:8000/docs` 에서 Swagger UI로 API를 직접 테스트할 수 있습니다.

### 2. CLI 테스트

서버가 켜진 상태에서 별도 터미널로 CLI 클라이언트를 실행합니다.

```bash
python app/client.py
# /switch {agent_name} 으로 에이전트 변경 (예: /switch navigator_agent)
```

### 3. Streamlit UI 실행

채팅 웹 인터페이스로 에이전트와 대화합니다. 서버가 켜진 상태에서 별도 터미널에 실행하세요.

```bash
streamlit run app/ui.py
```

`http://localhost:8501` 에서 접속 후, 사이드바에서 에이전트를 선택하고 대화를 시작합니다.  
Navigator 에이전트를 선택하면 VNC 화면에서 브라우저가 자동으로 움직이는 것을 볼 수 있습니다.

---

## 🔒 향후 연구과제: Anti-Bot 대응

이번 핸즈온 커리큘럼에서는 **공개된 연습용 사이트**([Quotes to Scrape](http://quotes.toscrape.com), 네이버 뉴스 등 로그인 불필요 영역)를 대상으로 실습했기 때문에, **Anti-Bot 방어 우회 기법은 의도적으로 다루지 않았습니다.**

실제 서비스 환경에서는 다양한 봇 차단 기술이 적용되어 있으며, 이를 해결하지 않으면 크롤러가 즉시 차단됩니다.
자체 학습 또는 다음 단계 심화 과정을 위해 주요 도전 과제와 접근 방향을 소개합니다.

### 주요 Anti-Bot 기법과 대응 방향

| 방어 기법 | 증상 | 대응 방향 |
|-----------|------|-----------|
| **Cloudflare / Akamai 봇 감지** | 접속 시 빈 페이지 또는 CAPTCHA 페이지 반환 | Playwright stealth 플러그인, 지연 시간 무작위화 |
| **CAPTCHA (reCAPTCHA, hCaptcha)** | 로봇 확인 요청 팝업 | [2captcha](https://2captcha.com), [CapSolver](https://capsolver.com) 등 CAPTCHA 해결 서비스 연동 |
| **IP 차단 / Rate Limiting** | 일정 횟수 이후 403, 429 오류 | 프록시 로테이션 (Bright Data, Oxylabs 등), 요청 간 delay 적용 |
| **JS 렌더링 감지** | Headless 브라우저 탐지 (navigator.webdriver 등) | `playwright-stealth` 라이브러리로 headless 특성 숨기기 |
| **로그인 필요 페이지** | 세션 없이 접근 시 리다이렉트 | 쿠키/세션 파일 저장 후 재사용, browser-use의 `shared_browser` 활용 |
| **동적 토큰 / CSRF** | API 요청 시 토큰 검증 실패 | 브라우저로 토큰 먼저 획득 후 API 호출 |

---

### 주의사항

> ⚠️ Anti-Bot 우회 기법은 대상 서비스의 **이용약관(ToS)을 반드시 확인**한 후 적용해야 합니다.  
> 허가 없는 크롤링은 법적 책임이 따를 수 있습니다.  
> 연습은 항상 **공식 허용된 테스트 사이트**나 **본인이 운영하는 서버**에서 진행하세요.

### 참고 자료

- [Playwright Stealth](https://github.com/AtuboDad/playwright_stealth): Headless 탐지 우회
- [crawl4ai Anti-Detection 가이드](https://docs.crawl4ai.com): crawl4ai의 스텔스 옵션
- [CapSolver](https://capsolver.com): CAPTCHA 자동 해결 API
- [Bright Data 프록시](https://brightdata.com): IP 로테이션 상용 서비스
- [browser-use 로그인 세션 유지](https://browser-use.com): `keep_alive=True` 패턴

---

### 기법별 구체적 접근 방법

#### 1. Headless 브라우저 탐지 우회 — `playwright-stealth`

사이트는 `navigator.webdriver`, `chrome.runtime` 같은 JS 속성으로 Headless 브라우저를 감지합니다.  
`playwright-stealth`는 이 속성들을 일반 브라우저처럼 위장합니다.

```python
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    stealth_sync(page)                     # ← Headless 특성 숨기기
    page.goto("https://target-site.com")
```

crawl4ai를 사용할 경우 `BrowserConfig`에서 설정합니다:

```python
from crawl4ai import BrowserConfig

browser_cfg = BrowserConfig(
    headless=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",  # 실제 브라우저 UA
    headers={"Accept-Language": "ko-KR,ko;q=0.9"},
)
```

---

#### 2. IP 차단 방어 — 프록시 로테이션 + 요청 딜레이

동일 IP에서 짧은 시간에 대량 요청을 보내면 차단됩니다.  
두 가지 전략을 병행합니다.

**① 요청 간 무작위 딜레이 (가장 단순한 방법)**

```python
import random, time

for url in url_list:
    page.goto(url)
    time.sleep(random.uniform(1.5, 4.0))  # 1.5~4초 무작위 대기
```

**② 프록시 로테이션 (IP 분산)**

```python
from crawl4ai import BrowserConfig

proxy_list = [
    "http://user:pass@proxy1.example.com:8080",
    "http://user:pass@proxy2.example.com:8080",
]

browser_cfg = BrowserConfig(
    proxy=random.choice(proxy_list),   # 요청마다 다른 IP
)
```

---

#### 3. CAPTCHA 자동 해결 — CapSolver API 연동

CAPTCHA가 나타나면 수동 입력 없이 외부 서비스 API로 해결합니다.

```python
import capsolver

capsolver.api_key = "YOUR_CAPSOLVER_KEY"

solution = capsolver.solve({
    "type": "ReCaptchaV2Task",          # CAPTCHA 종류
    "websiteURL": "https://target.com",
    "websiteKey": "6Le-..."             # 사이트의 reCAPTCHA 키
})

token = solution["gRecaptchaResponse"]  # 해결된 토큰
page.evaluate(f'document.getElementById("g-recaptcha-response").value = "{token}"')
page.click("#submit-button")
```

---

#### 4. 로그인 세션 유지 — 쿠키 저장 & 재사용

로그인 상태를 쿠키 파일로 저장해두면 매번 로그인 없이 세션을 재사용할 수 있습니다.

```python
from playwright.sync_api import sync_playwright
import json

# ① 최초 로그인 후 쿠키 저장
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://target.com/login")
    page.fill("#username", "my_id")
    page.fill("#password", "my_pw")
    page.click("#login-btn")
    page.wait_for_load_state("networkidle")

    cookies = page.context.cookies()
    with open("session.json", "w") as f:
        json.dump(cookies, f)           # 세션 파일 저장

# ② 이후 실행 시 쿠키 불러와서 로그인 없이 접근
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    with open("session.json") as f:
        context.add_cookies(json.load(f))
    page = context.new_page()
    page.goto("https://target.com/my-page")   # 로그인 상태 유지
```

browser-use에서는 `shared_browser`로 같은 효과를 낼 수 있습니다. (02 노트북 참고)

---

#### 5. AI 에이전트를 활용한 CAPTCHA·팝업 자동 처리

browser-use의 멀티모달 에이전트(vision 활성화)는 CAPTCHA나 팝업을 **시각적으로 인식**하고 자율적으로 처리할 수 있습니다.

```python
agent = Agent(
    task="로그인 페이지에서 CAPTCHA가 나타나면 이미지를 분석해서 텍스트를 입력하고, 팝업은 닫아주세요.",
    llm=bu_llm,
    use_vision=True,   # 스크린샷으로 CAPTCHA 이미지 인식
)
```

다만 이 방법은 복잡한 CAPTCHA(reCAPTCHA v3 등)에는 한계가 있으며, 전용 서비스와 병행하는 것이 현실적입니다.

---