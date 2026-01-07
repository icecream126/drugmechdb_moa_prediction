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
Simple Toy Multi-Agent Example

쉬운 수학 문제와 일상생활 질문을 multi-agent 방식으로 풀어보는 예제입니다.

Usage:
    python toy_multiagent.py debate "5 + 3 * 2 = ?"
    python toy_multiagent.py verify "12명이 피자 3판을 나눠먹으면 1인당 몇 조각?"
    python toy_multiagent.py stepwise "사과 5개에서 2개 먹고 3개 더 샀다. 몇 개?"
    python toy_multiagent.py all      # 모든 전략으로 테스트
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# vLLM 서버에서 돌고 있는 Qwen 모델 이름
MODEL = "Qwen/Qwen3-4B-Thinking-2507-FP8"


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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python toy_multiagent.py \"질문 내용\"")
        sys.exit(0)

    question = " ".join(sys.argv[1:])

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