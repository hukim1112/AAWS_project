import os
import base64
import json
import mimetypes
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@tool
def read_image_and_analyze(image_path: str, query_hint: str = "이 이미지의 내용을 상세히 설명해줘.") -> str:
    """
    로컬 이미지 파일을 읽고, Vision AI를 사용하여 이미지의 내용을 텍스트로 상세히 분석하여 반환합니다.
    이미지 경로와 함께, 무엇을 중점적으로 봐야 할지 힌트(query_hint)를 줄 수 있습니다.

    Args:
        image_path (str): 분석할 이미지의 파일 경로
        query_hint (str): 이미지에서 중점적으로 파악해야 할 내용 (예: "차트의 수치 변화를 설명해줘")
    """

    if not os.path.exists(image_path):
        return f"Error: 파일을 찾을 수 없습니다. 경로: {image_path}"

    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type: mime_type = "image/png"

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": f"당신은 유능한 이미지 분석가입니다. 다음 요청에 맞춰 이미지를 분석하세요: {query_hint}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                    }
                ]
            )
        ]

        result = vision_llm.invoke(messages)
        return f"[이미지 분석 결과 - {os.path.basename(image_path)}]\n{result.content}"

    except Exception as e:
        return f"Error: 이미지 분석 중 오류 발생. {str(e)}"

# --- Web Search Tool ---
try:
    from langchain_tavily import TavilySearch
except ImportError:
    # langchain_tavily 라이브러리가 설치되어 있지 않은 경우에 대한 예외 처리
    TavilySearch = None

# 검색 쿼리를 받아 웹에서 관련 문서/페이지를 찾아주는 도구입니다.
if TavilySearch:
    tavily_search = TavilySearch(max_results=5,
                                 topic="general",
                                 search_depth="advanced",
                                 include_raw_content=True)
else:
    tavily_search = None

@tool
def web_search_custom_tool(query: str) -> str:
    """
    웹 검색을 수행하여 최신 정보를 수집합니다.
    단순 요약뿐만 아니라, 가능한 경우 원문(Raw Content)을 우선적으로 제공하여 심도 있는 답변을 돕습니다.
    """
    print(f"---웹 검색 도구 호출: {query}---")

    # 1. Tavily 호출
    try:
        response = tavily_search.invoke({"query": query})
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {e}"

    processed_results = []

    # 안전장치: 너무 긴 텍스트는 잘라냄 (약 4000자 권장)
    MAX_LENGTH = 4000

    if isinstance(response, list):
        # TavilySearchResults는 설정에 따라 list나 dict를 반환할 수 있으므로 list인 경우 바로 처리
        items = response
    elif isinstance(response, dict) and "results" in response:
        items = response["results"]
    else:
        return "검색 결과가 없습니다."

    for item in items:
        # 2. Raw Content(원문) 상태 확인
        raw_text = item.get("raw_content")
        snippet_text = item.get("content", "") # 항상 있는 짧은 요약

        # 3. 우선순위 결정: Raw가 유효하고(None이 아니고), 내용이 충분히(500자 이상) 있다면 채택
        if raw_text and len(raw_text) > 500:
            final_content = raw_text
            data_source = "raw_full_text"
        else:
            # Raw가 없거나 너무 짧으면(오류일 가능성) 스니펫 사용
            final_content = snippet_text
            data_source = "snippet_fallback"

        # 4. 길이 제한 (Truncation)
        if len(final_content) > MAX_LENGTH:
            final_content = final_content[:MAX_LENGTH] + "...(중략)"

        # 5. JSON Dump (메타데이터 포함)
        doc_data = {
            "title": item.get("title"),
            "url": item.get("url"),
            "content": final_content, # 선별된 텍스트
            "source_type": data_source
        }

        # JSON 형태로 변환하여 리스트에 추가
        # ensure_ascii=False로 한글 깨짐 방지
        processed_results.append(json.dumps(doc_data, ensure_ascii=False))

    # 에이전트가 읽을 수 있도록 하나의 긴 문자열로 합쳐서 반환
    return "\n\n".join(processed_results)
