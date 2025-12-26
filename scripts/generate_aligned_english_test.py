#!/usr/bin/env python3
"""
한국어 test 데이터의 질문 분포에 맞춰 영어 test 데이터를 생성하는 스크립트

목표 분포:
  memory: 999
  reality: 999
  first_order_0_tom: 243
  first_order_1_tom: 203
  second_order_0_tom: 414
  second_order_1_tom: 414
  first_order_0_no_tom: 756
  first_order_1_no_tom: 796
  second_order_0_no_tom: 585
  second_order_1_no_tom: 585
"""

import argparse
import os
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.story import StoryType, generate_story
from src.world import World
from tqdm import tqdm
import numpy as np
import random


def get_target_distribution():
    """
    한국어 test 데이터의 목표 분포 반환

    Returns:
        dict: {question_category: target_count}
    """
    return {
        'memory': 999,
        'reality': 999,
        'first_order_0_tom': 243,
        'first_order_1_tom': 203,
        'second_order_0_tom': 414,
        'second_order_1_tom': 414,
        'first_order_0_no_tom': 756,
        'first_order_1_no_tom': 796,
        'second_order_0_no_tom': 585,
        'second_order_1_no_tom': 585,
    }


def main(opt):
    world = World()
    target_dist = get_target_distribution()

    # 현재 각 카테고리별로 생성된 개수 추적
    current_counts = defaultdict(int)
    total_needed = sum(target_dist.values())

    stories_path = os.path.join(opt.out_dir, "test.txt")
    trace_path = os.path.join(opt.out_dir, "test.trace")

    # 수집된 스토리들 (나중에 한번에 쓰기 위해 저장)
    all_stories = []
    all_traces = []
    all_story_types = []

    print("=== Target Distribution ===")
    for category in sorted(target_dist.keys()):
        print(f"{category:30s}: {target_dist[category]:4d}")
    print(f"\nTotal questions to generate: {total_needed}")
    print("\n=== Generating Stories ===")

    with tqdm(total=total_needed) as pbar:
        max_attempts = total_needed * 100  # 무한 루프 방지
        attempts = 0

        while sum(current_counts.values()) < total_needed and attempts < max_attempts:
            attempts += 1
            world.reset()
            stories, traces, story_type = generate_story(world)

            # 각 생성된 질문에 대해 필요한지 확인
            for story, trace in zip(stories, traces):
                # trace의 마지막이 질문 카테고리
                question_category = trace[-1]

                # 해당 카테고리가 아직 필요한지 확인
                if question_category in target_dist:
                    if current_counts[question_category] < target_dist[question_category]:
                        # 스토리 저장
                        all_stories.append(story)
                        all_traces.append(trace)
                        all_story_types.append(story_type.value)
                        current_counts[question_category] += 1
                        pbar.update(1)

                        # 모든 카테고리가 다 채워졌는지 확인
                        if sum(current_counts.values()) >= total_needed:
                            break

    # 결과 출력
    print("\n=== Generation Statistics ===")
    for category in sorted(target_dist.keys()):
        target = target_dist[category]
        actual = current_counts[category]
        status = "✓" if actual == target else "✗"
        print(f"{status} {category:30s}: {actual:4d} / {target:4d}")

    print(f"\nTotal generated: {len(all_stories)} / {total_needed}")
    print(f"Attempts needed: {attempts}")

    # 파일 쓰기
    print(f"\n=== Writing Output Files ===")
    with open(stories_path, "w") as f, open(trace_path, "w") as trace_f:
        for story, trace, story_type in zip(all_stories, all_traces, all_story_types):
            # 스토리 쓰기
            print(
                "\n".join(
                    [f"{i+1} {line.render()}" for i, line in enumerate(story)]
                ),
                file=f,
            )
            # trace 쓰기 (마지막에 story_type 추가 - 한국어 형식과 동일하게)
            print(",".join(trace + [story_type]), file=trace_f)
            f.flush()

    print(f"TXT:   {stories_path}")
    print(f"TRACE: {trace_path}")
    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed for rng")
    parser.add_argument("--out-dir", "-o", default="data/english", help="Output directory")
    opt = parser.parse_args()

    np.random.seed(opt.seed)
    random.seed(opt.seed)

    os.makedirs(opt.out_dir, exist_ok=True)
    main(opt)
