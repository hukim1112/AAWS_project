from langchain_core.tools import tool
from app.utils.data_loader import get_basic_retriever, get_self_query_retriever, get_multimodal_retriever
import json

# --- Basic RAG Tool ---
@tool(parse_docstring=True)
def search_bok_reports_basic(query: str) -> str:
    """Search the Bank of Korea industry reports for relevant information.
    Use this tool for general questions about industry trends, forecasts, and economic data.

    Args:
        query: The search query to find relevant information in the reports.
    """
    print(f"[Tool Log: BOK Basic] Searching for '{query}'...")

    retriever = get_basic_retriever()
    docs = retriever.invoke(query)

    result_text = ""
    for i, doc in enumerate(docs):
        meta_str = str(doc.metadata)
        result_text += f"[Document {i+1}]\nMetadata: {meta_str}\nContent: {doc.page_content}\n\n"

    if not result_text:
        return "No relevant documents found."
        
    return result_text

# --- Self-Query Tool ---
@tool(parse_docstring=True)
def search_bok_reports_self_query(query: str) -> str:
    """Search the Bank of Korea industry reports for relevant information.
    Use this tool for general questions about industry trends, forecasts, and economic data.
    The retriever automatically infers metadata filters (year, quarter) from your query.
    
    Args:
        query: The search query. Can include specific years or quarters (e.g., "2024년 1분기 반도체").
    """
    print(f"[Tool Log: BOK Self Query] Searching for '{query}'...")

    retriever = get_self_query_retriever()
    
    try:
        docs = retriever.invoke(query)

        
        result_text = ""
        for i, doc in enumerate(docs):
            meta_str = str(doc.metadata)
            result_text += f"[Document {i+1}]\nMetadata: {meta_str}\nContent: {doc.page_content}\n\n"
            
        if not result_text:
            return "No relevant documents found."
            
        return result_text
    except Exception as e:
        return f"Error during search: {e}"

# --- Multimodal RAG Tool ---
@tool(parse_docstring=True)
def search_bok_reports_multimodal(query: str) -> str:
    """Search the Bank of Korea industry reports (Multimodal). for relevant information.
    Use this tool for general questions about industry trends, forecasts, and economic data.
    The retriever automatically infers metadata filters (year, quarter) from your query.
    this tool extracts image paths from the documents. Use this when the user asks about charts, graphs, or visual data in the reports.
    
    Args:
        query: The search query. Can include specific years or quarters (e.g., "2024년 1분기 반도체").
    """
    import re
    import os
    import json
    from pathlib import Path
    
    print(f"[Tool Log: BOK Multimodal] Searching for '{query}'...")
    
    # 1. Multimodal Retriever 호출 (Self-Query 기능 포함)
    retriever = get_multimodal_retriever()
    
    try:
        # 2. 문서 검색 실행
        docs = retriever.invoke(query)
        
        # 3. 이미지 파일 경로 설정
        # 프로젝트 구조에 맞춰 이미지 저장소 경로를 계산합니다.
        current_dir = Path(__file__).parent 
        project_root = current_dir.parent.parent
        DATA_DIR = project_root / "data" / "extracted_images"
        
        combined_text = []
        found_image_paths = []
        
        # 4. 마크다운 이미지 링크 패턴 컴파일: ![alt](path)
        img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        
        for i, doc in enumerate(docs):
            content = doc.page_content
            meta_str = str(doc.metadata)
            page_num = doc.metadata.get('page', 'Unknown')
            
            # 5. 문서 내 이미지 경로 추출 및 유효성 검사
            paths_in_doc = img_pattern.findall(content)
            
            valid_paths_for_this_doc = []
            for rel_path in paths_in_doc:
                filename = os.path.basename(rel_path)
                abs_path = DATA_DIR / filename # 절대 경로로 변환
                
                # 실제 파일이 존재할 경우에만 리스트에 추가
                if abs_path.exists():
                    path_str = str(abs_path)
                    
                    # 전체 결과 리스트(found_image_paths)에 중복 없이 추가
                    if path_str not in found_image_paths:
                        found_image_paths.append(path_str)
                        valid_paths_for_this_doc.append(path_str)
            
            # 6. 텍스트 컨텍스트 구성 (LLM이 읽을 내용)
            # 이미지 경로가 발견되면 해당 문서 아래에 "> Found Images: ..." 형태로 명시하여
            # LLM이 텍스트와 이미지를 연결해서 이해하도록 도움
            doc_text = f"[Document {i+1} | Page {page_num}]\nMetadata: {meta_str}\nContent: {content}\n"
            if valid_paths_for_this_doc:
                doc_text += f"> Found Images: {', '.join(valid_paths_for_this_doc)}\n"
            
            combined_text.append(doc_text)

        # 7. 최종 결과 반환 (JSON 포맷)
        # context: 검색된 텍스트 컨텍스트
        # images: 활용 가능한 이미지 경로 리스트 (VLM 분석 및 UI 렌더링용)
        output_data = {
            "context": "\n\n".join(combined_text) if combined_text else "No relevant text context found.",
            "images": found_image_paths,
            "source": "한국은행 8대 업종 모니터링 보고서"
        }
        
        return json.dumps(output_data, ensure_ascii=False)

    except Exception as e:
        return f"Error during multimdoal search: {e}"
