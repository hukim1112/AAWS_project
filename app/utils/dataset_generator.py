import os
import glob
import random
import io
import base64
import json
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

# Dependency Check
try:
    from pdf2image import convert_from_path
except ImportError:
    print("Warning: 'pdf2image' not installed. Please install it and 'poppler-utils'.")
    convert_from_path = None

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def encode_image(image):
    """PIL Image를 Base64 문자열로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_golden_dataset(
    data_dir: str, 
    num_samples: int = 5, 
    output_file: Optional[str] = None
) -> List[Dict]:
    """
    RAG Evaluation을 위한 Golden Dataset(QA Pair)을 생성합니다.
    (PDF -> Image 변환 -> GPT-4o Vision 분석 -> QA 생성)
    
    Args:
        data_dir: PDF 파일들이 위치한 디렉토리 경로
        num_samples: 생성할 샘플(질문)의 개수 (PDF 파일 개수보다 많아도 됨)
        output_file: 결과를 저장할 파일 경로 (.csv 또는 .json)
        
    Returns:
        생성된 데이터셋 (List of Dict)
    """
    
    if convert_from_path is None:
        raise ImportError("pdf2image library is missing.")
        
    print(f"🚀 Golden Dataset 생성 시작 (목표: {num_samples}건)...")
    
    # PDF 파일 찾기
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"❌ PDF 파일을 찾을 수 없습니다: {data_dir}")
        return []

    generated_examples = []
    # 강력한 성능을 위해 gpt-4o 사용
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # num_samples를 채울 때까지 반복 (한 번에 3개씩(Simple, Reasoning, Visual) 생성되므로 Loop 횟수 조정)
    # 넉넉하게 Loop를 돌고 나중에 자릅니다.
    target_loops = (num_samples // 3) + 2 
    
    for i in range(target_loops):
        if len(generated_examples) >= num_samples:
            break
            
        try:
            target_pdf = random.choice(pdf_files)
            print(f"  - Reading PDF... : {os.path.basename(target_pdf)}")

            images = convert_from_path(target_pdf)
            if not images: continue

            # 연속된 3페이지 선택 (범위 초과 방지)
            max_start = max(0, len(images) - 3)
            start_idx = random.randint(0, max_start)
            selected_images = images[start_idx : start_idx + 3]

            # 이미지 컨텐츠 준비
            image_contents = []
            for img in selected_images:
                img_b64 = encode_image(img)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })

            # 프롬프트: 3페이지를 보고 종합적인 문제를 내도록 지시
            prompt = """
            당신은 RAG 시스템 평가를 위한 'Golden Dataset' 생성 전문가입니다.
            제공된 **3장의 연속된 보고서 페이지(이미지)**를 종합적으로 분석하여, 다음 3가지 유형의 질문-답변(QA) 쌍을 생성하세요.

            [필수 요구사항]
            1. **명확한 시점 명시 (중요)**: 모든 질문에는 반드시 **[연도]와 [분기]**를 포함해야 합니다. (예: "2024년 1분기 기준...", "2023년 상반기 대비...")
               - 만약 텍스트에 특정 시점이 없다면, 보고서의 전체 맥락(파일명 등)을 추론하여 넣으세요.
            2. **상세한 답변(Answer)**: 답변은 단답형이 아니라, 충분한 맥락을 포함한 문제 설명식으로 작성하세요.
            3. **Ground Truth Context**: 답변의 근거가 된 문장을 이미지에서 **그대로 발췌**하여 적으세요.
            4. **포맷**: 반드시 아래 JSON 포맷을 준수하세요.

            [질문 유형]
            1. **Simple**: 텍스트에 명시된 사실 확인 (예: 2024년 1분기 반도체 수출 성장률은?)
            2. **Reasoning**: 인과관계를 묻는 추론형 (예: 2024년 2분기 자동차 수출이 감소한 주된 원인은?)
            3. **Visual**: 도표/그래프를 해석해야만 알 수 있는 정보 (예: [그림 2-1]의 2024년 1분기 막대그래프 수치는?)

            Output JSON Format:
            {
                "samples": [
                    {"type": "Simple", "question": "...", "answer": "...", "ground_truth_context": "..."},
                    {"type": "Reasoning", "question": "...", "answer": "...", "ground_truth_context": "..."},
                    {"type": "Visual", "question": "...", "answer": "...", "ground_truth_context": "..."}
                ]
            }
            """

            msg = HumanMessage(content=[{"type": "text", "text": prompt}] + image_contents)
            
            # invoke
            res = llm.invoke([msg])

            # JSON 파싱 핸들링
            content = res.content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)

            for sample in data.get("samples", []):
                # 메타데이터 추가
                sample["source"] = os.path.basename(target_pdf)
                sample["page_range"] = f"{start_idx+1}~{start_idx+3}"
                
                generated_examples.append(sample)
                print(f"    ✅ [{sample['type']}] Q: {sample['question']}")
                
                if len(generated_examples) >= num_samples:
                    break

        except Exception as e:
            print(f"    ⚠️ 오류 발생 (Skip): {e}")
            continue

    # 결과 자르기 (정확히 num_samples만큼)
    generated_examples = generated_examples[:num_samples]
    
    # 파일 저장 옵션
    if output_file:
        try:
            if output_file.endswith(".csv"):
                df = pd.DataFrame(generated_examples)
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
            elif output_file.endswith(".json"):
                with open(output_file, "w", encoding='utf-8') as f:
                    json.dump(generated_examples, f, ensure_ascii=False, indent=2)
            print(f"💾 데이터셋 저장 완료: {output_file}")
        except Exception as e:
            print(f"❌ 파일 저장 실패: {e}")

    return generated_examples
