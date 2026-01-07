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
from openai import OpenAI


# -----------------------------------------------------------------------------
#   설정
# -----------------------------------------------------------------------------

def get_client():
    """vLLM 서버에 연결하는 OpenAI 클라이언트 생성"""
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8008/v1"
    )

MODEL = "Qwen/Qwen3-4B-Thinking-2507-FP8"


def call_llm(prompt: str, client=None) -> str:
    """LLM 호출"""
    if client is None:
        client = get_client()
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content


# -----------------------------------------------------------------------------
#   Multi-Agent 전략들
# -----------------------------------------------------------------------------

def strategy_debate(question: str, verbose: bool = True) -> str:
    """
    Debate 전략: 2명의 에이전트가 각자 답하고, 중재자가 최종 결정
    """
    client = get_client()
    
    if verbose:
        print("\n" + "="*50)
        print("전략: DEBATE (토론)")
        print("="*50)
        print(f"질문: {question}\n")
    
    # Agent 1: 빠른 직관적 답변
    prompt1 = f"""당신은 빠른 직관으로 답하는 Agent 1입니다.
질문: {question}
짧게 답하세요. (1-2문장)"""
    
    answer1 = call_llm(prompt1, client)
    if verbose:
        print(f"[Agent 1 - 직관] {answer1}\n")
    
    # Agent 2: 신중한 분석적 답변
    prompt2 = f"""당신은 신중하게 분석하는 Agent 2입니다.
질문: {question}
단계별로 생각하고 짧게 답하세요. (1-2문장)"""
    
    answer2 = call_llm(prompt2, client)
    if verbose:
        print(f"[Agent 2 - 분석] {answer2}\n")
    
    # Moderator: 최종 결정
    prompt_mod = f"""당신은 중재자입니다. 두 답변을 보고 최종 답을 결정하세요.

질문: {question}

Agent 1의 답: {answer1}
Agent 2의 답: {answer2}

최종 답을 한 문장으로 말하세요."""
    
    final = call_llm(prompt_mod, client)
    if verbose:
        print(f"[중재자 - 최종] {final}")
    
    return final


def strategy_verify(question: str, verbose: bool = True) -> str:
    """
    Verify 전략: 한 에이전트가 답하고, 다른 에이전트가 검증
    """
    client = get_client()
    
    if verbose:
        print("\n" + "="*50)
        print("전략: VERIFY (검증)")
        print("="*50)
        print(f"질문: {question}\n")
    
    # Solver: 문제 풀이
    prompt1 = f"""질문: {question}
답과 간단한 풀이를 적으세요."""
    
    answer1 = call_llm(prompt1, client)
    if verbose:
        print(f"[Solver] {answer1}\n")
    
    # Verifier: 검증
    prompt2 = f"""질문: {question}

제안된 답: {answer1}

이 답이 맞는지 검증하세요. 
틀렸다면 올바른 답을 알려주세요.
맞다면 "검증 완료: 정답입니다"라고 하세요."""
    
    verified = call_llm(prompt2, client)
    if verbose:
        print(f"[Verifier] {verified}")
    
    return verified


def strategy_stepwise(question: str, verbose: bool = True) -> str:
    """
    Stepwise 전략: 단계별로 다른 에이전트가 처리
    """
    client = get_client()
    
    if verbose:
        print("\n" + "="*50)
        print("전략: STEPWISE (단계별)")
        print("="*50)
        print(f"질문: {question}\n")
    
    # Step 1: 문제 이해
    prompt1 = f"""질문: {question}
이 문제에서 주어진 정보와 구해야 할 것을 정리하세요. (2-3줄)"""
    
    step1 = call_llm(prompt1, client)
    if verbose:
        print(f"[Step 1 - 문제 파악] {step1}\n")
    
    # Step 2: 풀이 방법
    prompt2 = f"""문제: {question}
분석: {step1}

풀이 방법을 간단히 설명하세요. (1-2줄)"""
    
    step2 = call_llm(prompt2, client)
    if verbose:
        print(f"[Step 2 - 풀이 방법] {step2}\n")
    
    # Step 3: 최종 답
    prompt3 = f"""문제: {question}
분석: {step1}
방법: {step2}

최종 답을 계산하고 한 문장으로 답하세요."""
    
    final = call_llm(prompt3, client)
    if verbose:
        print(f"[Step 3 - 최종 답] {final}")
    
    return final


# -----------------------------------------------------------------------------
#   테스트용 예제 질문들
# -----------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    # 수학 문제
    "5 + 3 * 2 = ?",
    "100원짜리 사탕 3개와 200원짜리 초콜릿 2개의 총 가격은?",
    "12명이 피자 3판(각 8조각)을 똑같이 나눠먹으면 1인당 몇 조각?",
    
    # 논리/일상 문제
    "비가 올 확률이 80%일 때, 우산을 가져가야 할까?",
    "냉장고에 우유가 3일 전에 유통기한이 지났다. 마셔도 될까?",
    "친구가 30분 늦는다고 했는데 이미 20분 기다렸다. 얼마나 더 기다려야 할까?",
]


def run_all_tests():
    """모든 전략으로 모든 예제 질문 테스트"""
    strategies = [
        ("DEBATE", strategy_debate),
        ("VERIFY", strategy_verify),
        ("STEPWISE", strategy_stepwise),
    ]
    
    print("\n" + "#"*60)
    print("  TOY MULTI-AGENT TEST")
    print("#"*60)
    
    for i, q in enumerate(EXAMPLE_QUESTIONS[:3], 1):  # 처음 3개만
        print(f"\n\n{'*'*60}")
        print(f"  질문 {i}: {q}")
        print('*'*60)
        
        for name, fn in strategies:
            try:
                fn(q, verbose=True)
            except Exception as e:
                print(f"[{name}] Error: {e}")
        
        print()


# -----------------------------------------------------------------------------
#   Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n사용 가능한 명령:")
        print("  debate <질문>   - 토론 전략으로 풀기")
        print("  verify <질문>   - 검증 전략으로 풀기")
        print("  stepwise <질문> - 단계별 전략으로 풀기")
        print("  all             - 모든 전략으로 예제 테스트")
        print("  examples        - 예제 질문들 보기")
        sys.exit(0)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "all":
        run_all_tests()
    
    elif cmd == "examples":
        print("\n예제 질문들:")
        for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
            print(f"  {i}. {q}")
        print()
    
    elif cmd in ["debate", "verify", "stepwise"]:
        if len(sys.argv) < 3:
            # 기본 예제 사용
            question = EXAMPLE_QUESTIONS[0]
            print(f"질문이 없어서 기본 예제 사용: {question}")
        else:
            question = " ".join(sys.argv[2:])
        
        if cmd == "debate":
            strategy_debate(question)
        elif cmd == "verify":
            strategy_verify(question)
        elif cmd == "stepwise":
            strategy_stepwise(question)
    
    else:
        print(f"알 수 없는 명령: {cmd}")
        print("'python toy_multiagent.py' 로 도움말을 확인하세요.")

