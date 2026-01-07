"""
Result:
(hypermoa) work@main1[hypermoa]:~/khm/DrugMechCounterfactuals/src$ CUDA_VISIBLE_DEVICES=3 python toy_multiagent.py debate "5 + 3 * 2 = ?"

==================================================
전략: DEBATE (토론)
==================================================
질문: 5 + 3 * 2 = ?

[Agent 1 - 직관] 

5 + 3 * 2 = 11 (먼저 곱셈을 계산한 후 덧셈을 수행합니다).

[Agent 2 - 분석] 

5 + 3 * 2 = 5 + (3 * 2) = 5 + 6 = 11.  
정답은 11입니다.

[중재자 - 최종] 

5 + 3 * 2 = 11.

"""

"""
Simple Toy Multi-Agent Batch Inference Example

쉬운 수학 문제와 일상생활 질문을 multi-agent 방식으로 풀어보는 예제입니다.
배치 처리를 지원하여 여러 질문을 효율적으로 병렬 처리할 수 있습니다.

Usage:
    # 단일 질문 처리
    python toy_multiagent_batch_inference.py "5 + 3 * 2 = ?"

    # 여러 질문 배치 처리
    python toy_multiagent_batch_inference.py "5 + 3 * 2 = ?" "사과 5개에서 2개 먹었다. 몇 개?"

    # 배치 모드로 강제 실행 (워커 수 지정 가능)
    python toy_multiagent_batch_inference.py --batch --workers 16 "질문1" "질문2" "질문3"

    # 예제 질문으로 테스트 (인자 없이 실행)
    python toy_multiagent_batch_inference.py
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# vLLM 서버에서 돌고 있는 Qwen 모델 이름
MODEL = "Qwen/Qwen3-4B-Thinking-2507-FP8"
# MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_client() -> OpenAI:
    """vLLM 서버에 연결하는 OpenAI 클라이언트 생성."""
    vllm_port = os.environ.get("VLLM_PORT", "8002")  # 필요시 export VLLM_PORT=8008 등으로 변경
    base_url = f"http://localhost:{vllm_port}/v1"
    return OpenAI(api_key="EMPTY", base_url=base_url)


def call_llm(prompt: str, client: OpenAI) -> str:
    """단일 프롬프트를 LLM에 전송하고 응답 문자열만 반환."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return resp.choices[0].message.content


def call_llm_batch(prompts: list[str], client: OpenAI, max_workers: int = 8) -> list[str]:
    """
    배치 프롬프트를 LLM에 병렬로 전송하고 응답 리스트를 반환.
    vLLM 서버가 내부적으로 continuous batching을 수행하므로,
    여러 요청을 동시에 보내면 효율적으로 처리됨.
    """
    def _single_call(prompt: str) -> str:
        return call_llm(prompt, client)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_single_call, prompts))
    return results


def solve_with_multiagent(question: str):
    """
    하나의 질문을 두 개의 에이전트가 병렬로 풀고,
    중재자가 두 답변을 종합하는 단일 파이프라인.
    """
    client = get_client()

    # Agent 1: 직관형
    prompt1 = f"""당신은 빠른 직관으로 답하는 Agent 1입니다.
질문: {question}
짧게 답하세요. (1-2문장)"""

    # Agent 2: 분석형
    prompt2 = f"""당신은 신중하게 분석하는 Agent 2입니다.
질문: {question}
단계별로 생각하고 짧게 답하세요. (1-2문장)"""

    def _call(p: str) -> str:
        return call_llm(p, client)

    # 두 에이전트를 병렬로 호출
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_call, prompt1)
        f2 = executor.submit(_call, prompt2)
        answer1 = f1.result()
        answer2 = f2.result()

    # 중재자: 최종 답
    prompt_mod = f"""당신은 중재자입니다. 두 답변을 보고 최종 답을 결정하세요.

질문: {question}

Agent 1의 답: {answer1}
Agent 2의 답: {answer2}

최종 답을 한 문장으로 말하세요."""
    final = call_llm(prompt_mod, client)

    return answer1, answer2, final


def solve_with_multiagent_batch(questions: list[str], batch_workers: int = 8):
    """
    여러 질문을 배치로 처리하는 multi-agent 파이프라인.
    각 에이전트가 질문 배치를 한 번에 처리하고,
    중재자도 배치로 최종 답을 생성함.

    Args:
        questions: 질문 리스트
        batch_workers: 배치 내 병렬 처리 워커 수

    Returns:
        list of tuples: [(answer1, answer2, final), ...] for each question
    """
    client = get_client()

    # Agent 1: 직관형 프롬프트 배치 생성
    prompts_agent1 = [
        f"""당신은 빠른 직관으로 답하는 Agent 1입니다.
질문: {q}
짧게 답하세요. (1-2문장)"""
        for q in questions
    ]

    # Agent 2: 분석형 프롬프트 배치 생성
    prompts_agent2 = [
        f"""당신은 신중하게 분석하는 Agent 2입니다.
질문: {q}
단계별로 생각하고 짧게 답하세요. (1-2문장)"""
        for q in questions
    ]

    # 두 에이전트의 배치를 병렬로 처리
    def _batch_call(prompts: list[str]) -> list[str]:
        return call_llm_batch(prompts, client, max_workers=batch_workers)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_batch_call, prompts_agent1)
        f2 = executor.submit(_batch_call, prompts_agent2)
        answers1 = f1.result()  # list of answers from agent 1
        answers2 = f2.result()  # list of answers from agent 2

    # 중재자: 최종 답 배치 생성
    prompts_moderator = [
        f"""당신은 중재자입니다. 두 답변을 보고 최종 답을 결정하세요.

질문: {q}

Agent 1의 답: {a1}
Agent 2의 답: {a2}

최종 답을 한 문장으로 말하세요."""
        for q, a1, a2 in zip(questions, answers1, answers2)
    ]
    finals = call_llm_batch(prompts_moderator, client, max_workers=batch_workers)

    # 결과를 튜플 리스트로 반환
    return list(zip(answers1, answers2, finals))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Batch Inference")
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="배치 모드로 실행 (여러 질문을 한 번에 처리)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="배치 내 병렬 처리 워커 수 (기본값: 8)"
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="질문(들). 배치 모드에서는 여러 질문을 공백으로 구분하거나 따옴표로 각각 묶어서 입력"
    )

    args = parser.parse_args()

    # 예제 질문 (입력이 없을 경우 사용)
    example_questions = [
        "5 + 3 * 2 = ?",
        "12명이 피자 3판을 나눠먹으면 1인당 몇 조각? (피자 한 판은 8조각)",
        "사과 5개에서 2개 먹고 3개 더 샀다. 몇 개?",
    ]

    if not args.questions:
        print("질문이 입력되지 않아 예제 질문을 사용합니다.\n")
        questions = example_questions
        args.batch = True  # 예제는 배치로 실행
    else:
        questions = args.questions

    if args.batch or len(questions) > 1:
        # 배치 모드
        print("\n" + "=" * 50)
        print("MULTI-AGENT BATCH INFERENCE")
        print("=" * 50)
        print(f"총 {len(questions)}개 질문 처리 (workers={args.workers})\n")

        t0 = time.time()
        results = solve_with_multiagent_batch(questions, batch_workers=args.workers)
        elapsed = time.time() - t0

        for i, (q, (ans1, ans2, final)) in enumerate(zip(questions, results), 1):
            print("-" * 50)
            print(f"[질문 {i}] {q}\n")
            print("[Agent 1 - 직관]")
            print(ans1.strip(), "\n")
            print("[Agent 2 - 분석]")
            print(ans2.strip(), "\n")
            print("[중재자 - 최종]")
            print(final.strip(), "\n")

        print("=" * 50)
        print(f"총 {len(questions)}개 질문 처리 완료")
        print(f"총 소요 시간: {elapsed:.2f}초")
        print(f"질문당 평균 시간: {elapsed / len(questions):.2f}초\n")

    else:
        # 단일 질문 모드 (기존 방식)
        question = questions[0]

        print("\n" + "=" * 50)
        print("MULTI-AGENT PARALLEL INFERENCE")
        print("=" * 50)
        print(f"질문: {question}\n")

        t0 = time.time()
        ans1, ans2, final = solve_with_multiagent(question)
        elapsed = time.time() - t0

        print("[Agent 1 - 직관]")
        print(ans1.strip(), "\n")

        print("[Agent 2 - 분석]")
        print(ans2.strip(), "\n")

        print("[중재자 - 최종]")
        print(final.strip(), "\n")

        print(f"총 소요 시간: {elapsed:.2f}초\n")