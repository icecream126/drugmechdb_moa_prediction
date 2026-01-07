"""
Multi-Agent LLM Testing for AddLink / EditLink Counterfactuals

This module implements several multi-agent strategies:
1. Debate: Multiple agents discuss and reach consensus
2. Chain-of-Experts: Specialized agents (Bio, Pharma, Reviewer) collaborate
3. Self-Refine: Initial response is critiqued and refined

Usage:
    $ python -m drugmechcf.llmx.test_multiagent debate \
        ../Data/Counterfactuals/AddLink_pos_dpi_r1k.json \
        ../Data/Sessions/Models/MultiAgent/addlink_pos_dpi_debate.json \
        -m Qwen3-4B-Thinking-2507-FP8 \
        -n 5

    $ python -m drugmechcf.llmx.test_multiagent experts \
        ../Data/Counterfactuals/AddLink_pos_dpi_r1k.json \
        ../Data/Sessions/Models/MultiAgent/addlink_pos_dpi_experts.json \
        -m Qwen3-4B-Thinking-2507-FP8 \
        -n 5
"""

import concurrent.futures
import dataclasses
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from drugmechcf.data.drugmechdb import load_drugmechdb, DrugMechDB
from drugmechcf.llm.openai import (
    OpenAICompletionOpts, OpenAICompletionClient, CompletionOutput, MODEL_KEYS
)
from drugmechcf.llm.prompt_types import QueryType, DrugDiseasePromptInfo, EditLinkInfo
from drugmechcf.llmx.prompts_addlink import PromptBuilder as AddLinkPromptBuilder
from drugmechcf.kgproc.addlink import create_new_moa_add_link
from drugmechcf.data.moagraph import MoaGraph
from drugmechcf.utils.misc import (
    NpEncoder, buffered_stdout, pp_funcargs, pp_underlined_hdg, check_output_file_dir
)


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

MAX_TEST_SAMPLES = 0
DEBUG_NO_LLM = False


# -----------------------------------------------------------------------------
#   Multi-Agent Prompt Templates
# -----------------------------------------------------------------------------

# Agent personas for the Debate strategy
DEBATE_AGENT_PERSONAS = {
    "agent_1": """You are Agent 1, a biomedical researcher specializing in drug mechanisms. 
You will analyze whether a drug can treat a disease given a novel interaction observation.
Focus on molecular pathways and biological plausibility.""",
    
    "agent_2": """You are Agent 2, a clinical pharmacologist with expertise in drug-disease relationships.
You will analyze whether a drug can treat a disease given a novel interaction observation.
Focus on clinical evidence and therapeutic outcomes.""",
    
    "moderator": """You are a neutral Moderator tasked with synthesizing the analyses from two expert agents.
Review both arguments carefully and provide a final consensus answer.
If the agents agree, summarize their reasoning.
If they disagree, weigh the evidence and make a final determination."""
}

DEBATE_INITIAL_PROMPT = """
{agent_persona}

-Task-
{drug_disease_prompt}

Please analyze this scenario and provide your assessment:
1. First, state whether the drug could be useful (YES or NO)
2. If YES, provide a potential Mechanism of Action using the format:
   <source_entity_type>: <source_entity_name> | <interaction_relationship> | <target_entity_type>: <target_entity_name>
3. Explain your reasoning briefly.
"""

DEBATE_MODERATOR_PROMPT = """
{moderator_persona}

-Original Question-
{drug_disease_prompt}

-Agent 1's Analysis-
{agent_1_response}

-Agent 2's Analysis-
{agent_2_response}

Based on both analyses, please provide the final answer:
1. State the consensus decision (YES or NO)
2. If YES, provide the most accurate Mechanism of Action
3. Briefly explain how you reached this consensus
"""


# Expert collaboration prompts
EXPERT_PERSONAS = {
    "biologist": """You are a Molecular Biologist specializing in protein interactions and cellular pathways.
Your expertise: protein-protein interactions, gene regulation, biological processes.
Focus on: identifying molecular mechanisms and pathway connections.""",

    "pharmacologist": """You are a Clinical Pharmacologist specializing in drug mechanisms.
Your expertise: drug targets, pharmacodynamics, therapeutic effects.
Focus on: evaluating drug-target interactions and clinical relevance.""",

    "reviewer": """You are a Senior Biomedical Reviewer who synthesizes expert opinions.
Your role: evaluate and combine analyses from specialists.
Focus on: creating a comprehensive, accurate mechanism of action."""
}

EXPERT_ANALYSIS_PROMPT = """
{expert_persona}

-Scenario-
{drug_disease_prompt}

Please provide your expert analysis:
1. From your specialization perspective, can this drug treat the disease?
2. What key {focus_area} support or refute this?
3. Suggest relevant entities and interactions within your expertise.

Keep your analysis focused and specific to your area of expertise.
"""

EXPERT_SYNTHESIS_PROMPT = """
{reviewer_persona}

-Original Question-
{drug_disease_prompt}

-Molecular Biology Analysis-
{biologist_response}

-Pharmacology Analysis-
{pharmacologist_response}

Based on both expert analyses, provide the final integrated answer:
1. State YES if the drug could treat the disease, or NO if not.
2. If YES, provide a complete Mechanism of Action integrating insights from both experts:
   Use format: <source_entity_type>: <source_entity_name> | <interaction_relationship> | <target_entity_type>: <target_entity_name>
3. Explain the integrated reasoning.
"""


# Self-Refine prompts
SELF_REFINE_INITIAL_PROMPT = """
You are a biomedical research assistant evaluating drug-disease relationships.

-Scenario-
{drug_disease_prompt}

Please provide your initial analysis:
1. State YES if the drug could treat the disease, or NO if not.
2. If YES, provide a Mechanism of Action.
3. Explain your reasoning.
"""

SELF_REFINE_CRITIQUE_PROMPT = """
You are a Critical Reviewer of biomedical research.

-Original Question-
{drug_disease_prompt}

-Initial Analysis-
{initial_response}

Please critically evaluate this analysis:
1. Are there any errors or weaknesses in the reasoning?
2. Is the Mechanism of Action complete and biologically plausible?
3. What improvements or corrections would you suggest?
4. Rate confidence (Low/Medium/High) in the initial answer.

Provide specific, actionable feedback.
"""

SELF_REFINE_FINAL_PROMPT = """
You are a biomedical research assistant. You previously provided an analysis that was critiqued.

-Original Question-
{drug_disease_prompt}

-Your Initial Analysis-
{initial_response}

-Critique Received-
{critique_response}

Based on this critique, provide your revised final answer:
1. State YES if the drug could treat the disease, or NO if not.
2. If YES, provide the improved Mechanism of Action.
3. Note what changes you made based on the critique.
"""


# -----------------------------------------------------------------------------
#   Data Classes
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class AddLinkTask:
    """Task data for testing"""
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    edit_link_info: EditLinkInfo
    is_negative_sample: bool
    moa: MoaGraph = None
    query_type: str = "ADD_LINK"


@dataclasses.dataclass
class MultiAgentResponse:
    """Response from multi-agent system"""
    strategy: str
    agent_responses: Dict[str, str]
    final_response: str
    final_decision: str  # "YES" or "NO"
    num_llm_calls: int
    
    def to_serialized(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# -----------------------------------------------------------------------------
#   Multi-Agent Test Class
# -----------------------------------------------------------------------------

class TestMultiAgent:
    """Multi-agent testing framework for drug mechanism counterfactuals"""
    
    def __init__(self,
                 model_key: str = "Qwen3-4B-Thinking-2507-FP8",
                 api_key: str = "EMPTY",
                 base_url: str = None,
                 timeout_secs: int = 600,
                 n_worker_threads: int = 1,
                 prompt_version: int = 0,
                 insert_known_moas: bool = False,
                 ):
        
        if base_url is None:
            vllm_port = os.environ.get("VLLM_PORT", "8000")
            base_url = f"http://localhost:8008/v1"
        
        if "/" not in model_key:
            model_key = "Qwen/" + model_key
        
        self.model = model_key
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_secs = timeout_secs
        self.n_worker_threads = n_worker_threads
        
        self.llm_opts = OpenAICompletionOpts(
            model=model_key,
            reasoning_effort="medium",
            seed=42,
            temperature=1.0
        )
        
        self.drugmechdb = load_drugmechdb()
        self.prompt_builder = AddLinkPromptBuilder(
            self.drugmechdb, prompt_version, include_examples=False, insert_known_moas=insert_known_moas
        )
        
        self.show_full_prompt = False
        self.show_response = False
        
        # Track LLM calls
        self.llm_call_count = 0
    
    def create_llm_client(self) -> OpenAICompletionClient:
        return OpenAICompletionClient(
            self.llm_opts,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_secs=self.timeout_secs
        )
    
    def call_llm(self, prompt: str, client: OpenAICompletionClient = None) -> Tuple[str, bool]:
        """Call LLM and return response text and success flag"""
        if client is None:
            client = self.create_llm_client()
        
        self.llm_call_count += 1
        
        try:
            response = client(user_prompt=prompt)
            if response.is_complete_response():
                return response.message, True
            else:
                return f"[Incomplete: {response.finish_reason}]", False
        except Exception as e:
            return f"[Error: {str(e)}]", False
    
    def create_sample_task(self, sample_data: Dict[str, Any]) -> AddLinkTask:
        """Create task from sample data"""
        sample_data = sample_data.copy()
        edit_link_info = EditLinkInfo(**sample_data["edit_link_info"])
        sample_data["edit_link_info"] = edit_link_info
        task = AddLinkTask(**sample_data)
        
        if not task.is_negative_sample:
            source_moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.source_moa_id)
            target_moa = self.drugmechdb.get_indication_graph_with_id(edit_link_info.target_moa_id)
            task.moa = create_new_moa_add_link(
                self.drugmechdb,
                source_moa, edit_link_info.source_node,
                edit_link_info.new_relation,
                target_moa, edit_link_info.target_node,
                source_moa_drug_node=task.drug_id,
                target_moa_disease_node=task.disease_id
            )
        
        return task
    
    def get_drug_disease_prompt(self, task: AddLinkTask) -> str:
        """Build the drug-disease prompt for multi-agent use"""
        prompt_info = self.prompt_builder.build_prompt_info(
            task.drug_id, task.drug_name,
            task.disease_id, task.disease_name,
            edit_link_info=task.edit_link_info,
            is_negative_sample=task.is_negative_sample,
            moa=task.moa
        )
        return self.prompt_builder.get_drug_disease_prompt(prompt_info)
    
    def extract_decision(self, response: str) -> str:
        """Extract YES/NO decision from response"""
        # Look for explicit YES or NO at the start of lines
        lines = response.strip().split('\n')
        for line in lines:
            line_clean = line.strip().upper()
            if line_clean.startswith('YES'):
                return "YES"
            elif line_clean.startswith('NO'):
                return "NO"
        
        # Check for patterns in the text
        if re.search(r'\byes\b', response.lower()):
            return "YES"
        elif re.search(r'\bno\b', response.lower()):
            return "NO"
        
        return "UNKNOWN"
    
    # -------------------------------------------------------------------------
    #   Strategy 1: Debate
    # -------------------------------------------------------------------------
    
    def run_debate(self, task: AddLinkTask, verbose: bool = False) -> MultiAgentResponse:
        """Run debate strategy with 2 agents and a moderator"""
        
        drug_disease_prompt = self.get_drug_disease_prompt(task)
        client = self.create_llm_client()
        
        agent_responses = {}
        call_count_start = self.llm_call_count
        
        # Agent 1 analysis
        agent1_prompt = DEBATE_INITIAL_PROMPT.format(
            agent_persona=DEBATE_AGENT_PERSONAS["agent_1"],
            drug_disease_prompt=drug_disease_prompt
        )
        agent_responses["agent_1"], _ = self.call_llm(agent1_prompt, client)
        
        if verbose:
            print("\n=== Agent 1 Response ===")
            print(agent_responses["agent_1"][:500] + "..." if len(agent_responses["agent_1"]) > 500 else agent_responses["agent_1"])
        
        # Agent 2 analysis
        agent2_prompt = DEBATE_INITIAL_PROMPT.format(
            agent_persona=DEBATE_AGENT_PERSONAS["agent_2"],
            drug_disease_prompt=drug_disease_prompt
        )
        agent_responses["agent_2"], _ = self.call_llm(agent2_prompt, client)
        
        if verbose:
            print("\n=== Agent 2 Response ===")
            print(agent_responses["agent_2"][:500] + "..." if len(agent_responses["agent_2"]) > 500 else agent_responses["agent_2"])
        
        # Moderator synthesis
        moderator_prompt = DEBATE_MODERATOR_PROMPT.format(
            moderator_persona=DEBATE_AGENT_PERSONAS["moderator"],
            drug_disease_prompt=drug_disease_prompt,
            agent_1_response=agent_responses["agent_1"],
            agent_2_response=agent_responses["agent_2"]
        )
        final_response, _ = self.call_llm(moderator_prompt, client)
        agent_responses["moderator"] = final_response
        
        if verbose:
            print("\n=== Moderator Final Response ===")
            print(final_response)
        
        return MultiAgentResponse(
            strategy="debate",
            agent_responses=agent_responses,
            final_response=final_response,
            final_decision=self.extract_decision(final_response),
            num_llm_calls=self.llm_call_count - call_count_start
        )
    
    # -------------------------------------------------------------------------
    #   Strategy 2: Chain of Experts
    # -------------------------------------------------------------------------
    
    def run_experts(self, task: AddLinkTask, verbose: bool = False) -> MultiAgentResponse:
        """Run expert collaboration strategy"""
        
        drug_disease_prompt = self.get_drug_disease_prompt(task)
        client = self.create_llm_client()
        
        agent_responses = {}
        call_count_start = self.llm_call_count
        
        # Biologist analysis
        bio_prompt = EXPERT_ANALYSIS_PROMPT.format(
            expert_persona=EXPERT_PERSONAS["biologist"],
            drug_disease_prompt=drug_disease_prompt,
            focus_area="molecular mechanisms and pathway connections"
        )
        agent_responses["biologist"], _ = self.call_llm(bio_prompt, client)
        
        if verbose:
            print("\n=== Biologist Analysis ===")
            print(agent_responses["biologist"][:500] + "..." if len(agent_responses["biologist"]) > 500 else agent_responses["biologist"])
        
        # Pharmacologist analysis
        pharma_prompt = EXPERT_ANALYSIS_PROMPT.format(
            expert_persona=EXPERT_PERSONAS["pharmacologist"],
            drug_disease_prompt=drug_disease_prompt,
            focus_area="drug-target interactions and clinical effects"
        )
        agent_responses["pharmacologist"], _ = self.call_llm(pharma_prompt, client)
        
        if verbose:
            print("\n=== Pharmacologist Analysis ===")
            print(agent_responses["pharmacologist"][:500] + "..." if len(agent_responses["pharmacologist"]) > 500 else agent_responses["pharmacologist"])
        
        # Reviewer synthesis
        synthesis_prompt = EXPERT_SYNTHESIS_PROMPT.format(
            reviewer_persona=EXPERT_PERSONAS["reviewer"],
            drug_disease_prompt=drug_disease_prompt,
            biologist_response=agent_responses["biologist"],
            pharmacologist_response=agent_responses["pharmacologist"]
        )
        final_response, _ = self.call_llm(synthesis_prompt, client)
        agent_responses["reviewer"] = final_response
        
        if verbose:
            print("\n=== Reviewer Synthesis ===")
            print(final_response)
        
        return MultiAgentResponse(
            strategy="experts",
            agent_responses=agent_responses,
            final_response=final_response,
            final_decision=self.extract_decision(final_response),
            num_llm_calls=self.llm_call_count - call_count_start
        )
    
    # -------------------------------------------------------------------------
    #   Strategy 3: Self-Refine
    # -------------------------------------------------------------------------
    
    def run_self_refine(self, task: AddLinkTask, verbose: bool = False) -> MultiAgentResponse:
        """Run self-refinement strategy"""
        
        drug_disease_prompt = self.get_drug_disease_prompt(task)
        client = self.create_llm_client()
        
        agent_responses = {}
        call_count_start = self.llm_call_count
        
        # Initial response
        initial_prompt = SELF_REFINE_INITIAL_PROMPT.format(
            drug_disease_prompt=drug_disease_prompt
        )
        agent_responses["initial"], _ = self.call_llm(initial_prompt, client)
        
        if verbose:
            print("\n=== Initial Response ===")
            print(agent_responses["initial"][:500] + "..." if len(agent_responses["initial"]) > 500 else agent_responses["initial"])
        
        # Critique
        critique_prompt = SELF_REFINE_CRITIQUE_PROMPT.format(
            drug_disease_prompt=drug_disease_prompt,
            initial_response=agent_responses["initial"]
        )
        agent_responses["critique"], _ = self.call_llm(critique_prompt, client)
        
        if verbose:
            print("\n=== Critique ===")
            print(agent_responses["critique"][:500] + "..." if len(agent_responses["critique"]) > 500 else agent_responses["critique"])
        
        # Refined final response
        refine_prompt = SELF_REFINE_FINAL_PROMPT.format(
            drug_disease_prompt=drug_disease_prompt,
            initial_response=agent_responses["initial"],
            critique_response=agent_responses["critique"]
        )
        final_response, _ = self.call_llm(refine_prompt, client)
        agent_responses["refined"] = final_response
        
        if verbose:
            print("\n=== Refined Response ===")
            print(final_response)
        
        return MultiAgentResponse(
            strategy="self_refine",
            agent_responses=agent_responses,
            final_response=final_response,
            final_decision=self.extract_decision(final_response),
            num_llm_calls=self.llm_call_count - call_count_start
        )
    
    # -------------------------------------------------------------------------
    #   Batch Testing
    # -------------------------------------------------------------------------
    
    def test_batch(self,
                   samples_data_file: str,
                   output_json_file: str = None,
                   strategy: str = "debate",
                   max_samples: int = 0,
                   show_response: bool = False):
        """Run batch test with specified multi-agent strategy"""
        
        pp_underlined_hdg(f"Multi-Agent Batch Test: {strategy}", linechar='~', overline=True)
        
        self.show_response = show_response
        
        if output_json_file is not None:
            check_output_file_dir(output_json_file)
        
        # Load samples
        with open(samples_data_file) as f:
            test_samples = json.load(f)
        
        print(f"Total samples in file: {len(test_samples):,d}")
        
        # Apply sample limit
        if max_samples > 0:
            test_samples = test_samples[:max_samples]
        
        print(f"Samples to test: {len(test_samples):,d}")
        print(f"Strategy: {strategy}")
        print(f"Model: {self.model}")
        print()
        
        # Select strategy function
        strategy_fn = {
            "debate": self.run_debate,
            "experts": self.run_experts,
            "self_refine": self.run_self_refine,
        }.get(strategy)
        
        if strategy_fn is None:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from: debate, experts, self_refine")
        
        # Run tests
        results = []
        decision_counts = Counter()
        
        for i, sample_data in enumerate(test_samples, start=1):
            print(f"\n--- Sample {i}/{len(test_samples)} ---")
            
            task = self.create_sample_task(sample_data)
            print(f"Drug: {task.drug_name} | Disease: {task.disease_name}")
            print(f"Is negative sample: {task.is_negative_sample}")
            
            # Run multi-agent strategy
            ma_response = strategy_fn(task, verbose=show_response)
            
            print(f"Final decision: {ma_response.final_decision}")
            print(f"LLM calls: {ma_response.num_llm_calls}")
            
            decision_counts[ma_response.final_decision] += 1
            
            # Store result
            result = {
                "sample_info": {
                    "drug_id": task.drug_id,
                    "drug_name": task.drug_name,
                    "disease_id": task.disease_id,
                    "disease_name": task.disease_name,
                    "is_negative_sample": task.is_negative_sample,
                },
                "multi_agent_response": ma_response.to_serialized(),
                "expected_positive": not task.is_negative_sample,
            }
            results.append(result)
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(test_samples)}")
        print(f"Total LLM calls: {self.llm_call_count}")
        print(f"Decision distribution: {dict(decision_counts)}")
        
        # Calculate accuracy
        n_correct = sum(
            1 for r in results 
            if (r["multi_agent_response"]["final_decision"] == "YES") == r["expected_positive"]
        )
        accuracy = n_correct / len(results) if results else 0
        print(f"Accuracy: {accuracy:.2%} ({n_correct}/{len(results)})")
        
        # Save results
        if output_json_file:
            output_data = {
                "args": {
                    "model": self.model,
                    "strategy": strategy,
                    "samples_data_file": samples_data_file,
                    "max_samples": max_samples,
                },
                "metrics": {
                    "total_samples": len(test_samples),
                    "total_llm_calls": self.llm_call_count,
                    "decision_counts": dict(decision_counts),
                    "accuracy": accuracy,
                },
                "session": results,
            }
            
            with open(output_json_file, "w") as f:
                json.dump(output_data, f, indent=4, cls=NpEncoder)
            
            print(f"\nResults saved to: {output_json_file}")
        
        return results


# -----------------------------------------------------------------------------
#   Command Line Interface
# -----------------------------------------------------------------------------

def test_multiagent_cmd(samples_data_file: str,
                        output_json_file: str = None,
                        *,
                        model_key: str = "Qwen3-4B-Thinking-2507-FP8",
                        strategy: str = "debate",
                        max_samples: int = 0,
                        insert_known_moas: bool = False,
                        show_response: bool = False,
                        timeout_secs: int = 600):
    """Command to run multi-agent test"""
    
    pp_funcargs(test_multiagent_cmd)
    
    tester = TestMultiAgent(
        model_key=model_key,
        timeout_secs=timeout_secs,
        insert_known_moas=insert_known_moas,
    )
    
    tester.test_batch(
        samples_data_file,
        output_json_file,
        strategy=strategy,
        max_samples=max_samples,
        show_response=show_response,
    )


# ======================================================================================================
#   Main
# ======================================================================================================

if __name__ == "__main__":
    
    import argparse
    from drugmechcf.utils.misc import print_cmd
    
    _argparser = argparse.ArgumentParser(
        description='Multi-Agent LLM Testing for Drug Mechanism Counterfactuals',
    )
    
    _subparsers = _argparser.add_subparsers(dest='subcmd', title='Available commands')
    _subparsers.required = True
    
    # ... debate strategy
    _sub_cmd_parser = _subparsers.add_parser('debate',
                                             help="Run debate strategy with 2 agents + moderator")
    _sub_cmd_parser.add_argument('-m', '--model', type=str, default="Qwen3-4B-Thinking-2507-FP8",
                                 help="Model name")
    _sub_cmd_parser.add_argument('-n', '--max_samples', type=int, default=0,
                                 help="Max samples to test (0 = all)")
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true',
                                 help="Insert known MoAs (closed-world)")
    _sub_cmd_parser.add_argument('-r', '--show_response', action='store_true',
                                 help="Show agent responses")
    _sub_cmd_parser.add_argument('-t', '--timeout', type=int, default=600,
                                 help="Timeout in seconds")
    _sub_cmd_parser.add_argument('samples_data_file', type=str)
    _sub_cmd_parser.add_argument('output_json_file', nargs='?', type=str, default=None)
    
    # ... experts strategy
    _sub_cmd_parser = _subparsers.add_parser('experts',
                                             help="Run expert collaboration strategy")
    _sub_cmd_parser.add_argument('-m', '--model', type=str, default="Qwen3-4B-Thinking-2507-FP8")
    _sub_cmd_parser.add_argument('-n', '--max_samples', type=int, default=0)
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true')
    _sub_cmd_parser.add_argument('-r', '--show_response', action='store_true')
    _sub_cmd_parser.add_argument('-t', '--timeout', type=int, default=600)
    _sub_cmd_parser.add_argument('samples_data_file', type=str)
    _sub_cmd_parser.add_argument('output_json_file', nargs='?', type=str, default=None)
    
    # ... self_refine strategy
    _sub_cmd_parser = _subparsers.add_parser('self_refine',
                                             help="Run self-refinement strategy")
    _sub_cmd_parser.add_argument('-m', '--model', type=str, default="Qwen3-4B-Thinking-2507-FP8")
    _sub_cmd_parser.add_argument('-n', '--max_samples', type=int, default=0)
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true')
    _sub_cmd_parser.add_argument('-r', '--show_response', action='store_true')
    _sub_cmd_parser.add_argument('-t', '--timeout', type=int, default=600)
    _sub_cmd_parser.add_argument('samples_data_file', type=str)
    _sub_cmd_parser.add_argument('output_json_file', nargs='?', type=str, default=None)
    
    # ... compare all strategies
    _sub_cmd_parser = _subparsers.add_parser('compare',
                                             help="Compare all strategies on same samples")
    _sub_cmd_parser.add_argument('-m', '--model', type=str, default="Qwen3-4B-Thinking-2507-FP8")
    _sub_cmd_parser.add_argument('-n', '--max_samples', type=int, default=5)
    _sub_cmd_parser.add_argument('-k', '--insert_known_moas', action='store_true')
    _sub_cmd_parser.add_argument('-r', '--show_response', action='store_true')
    _sub_cmd_parser.add_argument('-t', '--timeout', type=int, default=600)
    _sub_cmd_parser.add_argument('samples_data_file', type=str)
    _sub_cmd_parser.add_argument('output_dir', nargs='?', type=str, default=None)
    
    _args = _argparser.parse_args()
    
    # -----------------------------------------------------------------------------
    
    start_time = datetime.now()
    
    print("=" * 70)
    print_cmd()
    print()
    
    if _args.subcmd in ['debate', 'experts', 'self_refine']:
        test_multiagent_cmd(
            _args.samples_data_file,
            _args.output_json_file,
            model_key=_args.model,
            strategy=_args.subcmd,
            max_samples=_args.max_samples,
            insert_known_moas=_args.insert_known_moas,
            show_response=_args.show_response,
            timeout_secs=_args.timeout,
        )
    
    elif _args.subcmd == 'compare':
        # Run all strategies and compare
        print("Running comparison of all strategies...")
        print()
        
        for strategy in ['debate', 'experts', 'self_refine']:
            print(f"\n{'='*70}")
            print(f"Strategy: {strategy.upper()}")
            print('='*70)
            
            output_file = None
            if _args.output_dir:
                os.makedirs(_args.output_dir, exist_ok=True)
                output_file = os.path.join(_args.output_dir, f"{strategy}_results.json")
            
            test_multiagent_cmd(
                _args.samples_data_file,
                output_file,
                model_key=_args.model,
                strategy=strategy,
                max_samples=_args.max_samples,
                insert_known_moas=_args.insert_known_moas,
                show_response=_args.show_response,
                timeout_secs=_args.timeout,
            )
    
    print('\n' + '='*70)
    print(f'Total Run time = {datetime.now() - start_time}')
    print()

