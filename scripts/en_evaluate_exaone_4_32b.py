import argparse
import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load .env BEFORE importing other libraries
load_dotenv()

# FORCE set HF_HOME before importing transformers (override system defaults)
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
    prompt = f"""Read the following story and answer the question.

Story:
{context}

Question: {question}
Answer with only the location name. Examples: "box", "kitchen", "basket"
Answer:"""
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


def query_exaone(
    model,
    tokenizer,
    context: str,
    question: str
) -> str:
    prompt = create_prompt(context, question)

    try:
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]

        # Tokenize input
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Generate response (deterministic, same as Gemini API with temperature=0.0)
        with torch.no_grad():
            outputs = model.generate(
                input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode output (excluding input tokens)
        answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return answer.strip()

    except Exception as e:
        print(f"Error querying EXAONE: {e}")
        return ""


def normalize_answer(answer: str) -> str:
    answer = answer.strip().lower()

    # Remove English articles
    for article in ['the ', 'a ', 'an ']:
        if answer.startswith(article):
            answer = answer[len(article):]

    # Remove prepositions
    for prep in ['in the ', 'at the ', 'inside the ', 'in ', 'at ', 'inside ']:
        if answer.startswith(prep):
            answer = answer[len(prep):]

    # Remove punctuation
    answer = answer.replace('.', '').replace(',', '').replace('!', '').replace('?', '')

    return answer.strip()


def evaluate_joint_accuracy(
    model,
    tokenizer,
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

            predicted_answer = query_exaone(
                model, tokenizer, qa['context'], qa['question']
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
        'model': 'LGAI-EXAONE/EXAONE-4.0-32B',
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
    print(f"모델: EXAONE-4.0-32B")
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
        description="English ToMi 데이터셋 평가 - EXAONE-4.0-32B"
    )
    parser.add_argument(
        "--txt-path",
        type=str,
        default="data/english/test.txt",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default="data/english/test.trace",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-32B",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # DEBUG: Print environment variables
    print("="*50)
    print("DEBUG: Environment Variables")
    print("="*50)
    print(f"HF_HOME: {os.getenv('HF_HOME')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print(f"XDG_CACHE_HOME: {os.getenv('XDG_CACHE_HOME')}")

    # Check transformers cache location
    try:
        from transformers.utils import TRANSFORMERS_CACHE
        print(f"TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE}")
    except:
        pass

    try:
        import huggingface_hub
        print(f"HF Hub cache dir: {huggingface_hub.constants.HF_HUB_CACHE}")
    except:
        pass
    print("="*50)
    print()

    # Set GPU from environment variable
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"Using GPU: {cuda_visible_devices}")

    print("="*50)
    print("EXAONE-4.0-32B 모델 로딩 중...")
    print("="*50)
    print(f"모델 경로: {args.model_path}")
    print(f"디바이스: {args.device}")
    print("이 작업은 몇 분 정도 걸릴 수 있습니다...\n")

    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        print(f"\n✓ 모델 로딩 완료!")
        print(f"✓ 모델 디바이스: {next(model.parameters()).device}")
        print(f"✓ 모델 dtype: {next(model.parameters()).dtype}\n")

    except Exception as e:
        print(f"✗ 모델 로딩 실패: {e}")
        return

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/result_exaone_4_32b_{timestamp}.json"

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "results", exist_ok=True)

    print(f"평가 시작")
    print(f"모델: EXAONE-4.0-32B")
    print(f"데이터: {args.txt_path}\n")

    evaluate_joint_accuracy(
        model=model,
        tokenizer=tokenizer,
        txt_path=args.txt_path,
        trace_path=args.trace_path,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
