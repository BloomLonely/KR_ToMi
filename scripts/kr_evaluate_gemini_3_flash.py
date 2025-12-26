import argparse
import os
import sys
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import json
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()


def load_stories(txt_path: str) -> List[List[str]]:
    stories = []
    current_story = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('1 '):
                if current_story:
                    stories.append(current_story)
                current_story = [line]
            else:
                current_story.append(line)

        if current_story:
            stories.append(current_story)

    return stories


def parse_story_qa(story_lines: List[str]) -> Dict[str, str]:
    context = []
    question = None
    answer = None

    for line in story_lines:
        if '\t' in line:
            parts = line.split('\t')
            question_line = parts[0]
            answer = parts[1]

            question = question_line.split(' ', 1)[1] if ' ' in question_line else question_line
        else:
            text = line.split(' ', 1)[1] if ' ' in line else line
            context.append(text)

    return {
        'context': '\n'.join(context),
        'question': question,
        'answer': answer
    }


def load_traces(trace_path: str) -> List[str]:
    with open(trace_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def create_prompt(context: str, question: str) -> str:

    prompt = f"""다음 이야기를 읽고 질문에 답하세요.

이야기:
{context}

질문: {question}
장소 이름만 답하세요. 예시: "욕조", "양동이", "상자"
답변:"""
    return prompt


def parse_trace_label(trace_line: str) -> Dict[str, str]:
    parts = trace_line.split(',')
    full_type = parts[-1].strip()
    category = parts[-2].strip() if len(parts) > 1 else ""
    
    return {
        "full_trace": trace_line,
        "type": full_type,
        "category": category
    }


def query_openrouter(
    client: OpenAI,
    model: str,
    context: str,
    question: str
) -> str:
    prompt = create_prompt(context, question)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=50
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print(f"Error querying OpenRouter: {e}")
        return ""


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower()

    # 한국어 조사 제거 (은/는, 이/가, 을/를, 에, 에서, 로/으로)
    korean_particles = [
        '에서는', '에서도', '에서', '에는', '에도', '에',
        '으로는', '으로도', '으로', '로는', '로도', '로',
        '이는', '이도', '이', '가는', '가도', '가',
        '을는', '을도', '을', '를는', '를도', '를',
        '은', '는', '와', '과', '의'
    ]

    for particle in korean_particles:
        if answer.endswith(particle):
            answer = answer[:-len(particle)]
            break

    # 구두점 제거
    answer = answer.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    answer = answer.replace('。', '').replace('、', '')

    return answer.strip()


def evaluate_joint_accuracy(
    client: OpenAI,
    model: str,
    txt_path: str,
    trace_path: str,
    output_path: str = None
) -> Dict:
    stories = load_stories(txt_path)
    traces = load_traces(trace_path)

    story_groups = []
    trace_groups = []
    for i in range(0, len(stories), 6):
        story_groups.append(stories[i:i+6])
        trace_groups.append(traces[i:i+6])

    print(f"총 {len(story_groups)}개 스토리 그룹 로드됨")
    print(f"총 {len(stories)}개 질문")

    expected_categories = [
        "memory",
        "reality",
        "first_order_0_tom",
        "first_order_1_tom",
        "second_order_0_tom",
        "second_order_1_tom",
        "first_order_0_no_tom",
        "first_order_1_no_tom",
        "second_order_0_no_tom",
        "second_order_1_no_tom"
    ]
    
    category_stats = {cat: {'correct': 0, 'total': 0} for cat in expected_categories}
    
    results = []
    correct_stories = 0
    total_correct_qa = 0

    for group_idx, (group, group_traces) in enumerate(tqdm(zip(story_groups, trace_groups), total=len(story_groups), desc="Evaluating")):
        group_results = []
        all_correct = True

        for story_lines, trace_line in zip(group, group_traces):
            qa = parse_story_qa(story_lines)
            trace_info = parse_trace_label(trace_line)
            
            predicted_answer = query_openrouter(
                client, model, qa['context'], qa['question']
            )
            is_correct = normalize_answer(predicted_answer) == normalize_answer(qa['answer'])

            if is_correct:
                total_correct_qa += 1
            else:
                all_correct = False

            cat = trace_info['category']
            if cat not in category_stats:
                category_stats[cat] = {'correct': 0, 'total': 0}
            
            category_stats[cat]['total'] += 1
            if is_correct:
                category_stats[cat]['correct'] += 1

            group_results.append({
                'question': qa['question'],
                'correct_answer': qa['answer'],
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'type': trace_info['type'],
                'category': trace_info['category'],
                'full_trace': trace_info['full_trace']
            })

        if all_correct:
            correct_stories += 1

        results.append({
            'group_id': group_idx,
            'all_correct': all_correct,
            'questions': group_results
        })

    total_stories = len(story_groups)
    total_questions = len(stories)

    joint_accuracy = (correct_stories / total_stories) * 100 if total_stories > 0 else 0
    average_accuracy = (total_correct_qa / total_questions) * 100 if total_questions > 0 else 0

    category_summary = {}
    for cat, stats in category_stats.items():
        ratio_str = f"{stats['correct']}/{stats['total']}"
        category_summary[cat] = ratio_str

    summary = {
        'model': model,
        'total_story_groups': total_stories,
        'correct_stories': correct_stories,
        'joint_accuracy': joint_accuracy,
        'total_questions': total_questions,
        'correct_questions': total_correct_qa,
        'average_accuracy': average_accuracy,
        'category_stats': category_summary,
        'results': results
    }

    print("\n" + "="*30)
    print("평가 결과")
    print("="*30)
    print(f"모델: {model}")
    print(f"총 스토리 그룹: {total_stories}")
    print(f"정답 스토리 그룹: {correct_stories}")
    print(f"Joint Accuracy: {joint_accuracy:.2f}%")
    print(f"\n총 질문: {total_questions}")
    print(f"정답 질문: {total_correct_qa}")
    print(f"Average Accuracy: {average_accuracy:.2f}%")
    print("-" * 30)
    print("카테고리별 정답률:")

    sorted_cats = sorted(category_stats.keys())
    for cat in sorted_cats:
        stats = category_stats[cat]
        print(f"{cat}: {stats['correct']}/{stats['total']}")
        
    print("="*30)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n결과가 {output_path}에 저장되었습니다.")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Korean ToMi 데이터셋 평가"
    )
    parser.add_argument(
        "--txt-path",
        type=str,
        default="data/korean/test.txt",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default="data/korean/test.trace",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY가 .env 파일에 설정되지 않았습니다.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    MODEL = "google/gemini-3-flash-preview"

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = MODEL.replace("/", "_").replace("-", "_")
        args.output = f"results/result_{model_name}_{timestamp}.json"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", exist_ok=True)

    print(f"평가 시작")
    print(f"모델: {MODEL}")
    print(f"데이터: {args.txt_path}\n")

    evaluate_joint_accuracy(
        client=client,
        model=MODEL,
        txt_path=args.txt_path,
        trace_path=args.trace_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()