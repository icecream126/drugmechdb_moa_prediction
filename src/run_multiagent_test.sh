#!/bin/bash
#
#  Run Multi-Agent test on counterfactual queries
#
#  Strategies:
#    - debate:      2 agents debate + moderator synthesis
#    - experts:     Biologist + Pharmacologist + Reviewer
#    - self_refine: Initial response -> Critique -> Refined response
#
#  Usage:
#    $ source $PROJDIR/.venv/bin/activate
#    (dmcf) $ cd $PROJDIR/src
#    (dmcf) $ ./run_multiagent_test.sh
#

CMD=$(basename $0)

start_date=$(date)
echo "Start: ${start_date}"
echo

# -- Configuration

MODEL="Qwen3-4B-Thinking-2507-FP8"
MAX_SAMPLES=5              # Small number for testing; set to 0 for all samples
SHOW_RESPONSE="-r"         # Remove this flag to hide intermediate responses

# -- Paths

DATA_DIR=../Data/Counterfactuals
DEST_DIR=../Data/Sessions/Models/MultiAgent

# Create output directory if it doesn't exist
mkdir -p $DEST_DIR

# -- Test data file (pick one)

# Positive samples
DATA_FILE=$DATA_DIR/AddLink_pos_dpi_r1k.json

# Negative samples
# DATA_FILE=$DATA_DIR/AddLink_neg_dpi_r1k.json

# -- Run tests

echo "=============================================="
echo "Testing Multi-Agent Strategies"
echo "=============================================="
echo "Model: $MODEL"
echo "Data file: $DATA_FILE"
echo "Max samples: $MAX_SAMPLES"
echo

# Strategy 1: Debate
echo "----------------------------------------"
echo "Strategy: DEBATE"
echo "----------------------------------------"
python -m drugmechcf.llmx.test_multiagent debate \
    -m $MODEL \
    -n $MAX_SAMPLES \
    $SHOW_RESPONSE \
    $DATA_FILE \
    $DEST_DIR/debate_test.json

# Strategy 2: Experts
echo "----------------------------------------"
echo "Strategy: EXPERTS"
echo "----------------------------------------"
python -m drugmechcf.llmx.test_multiagent experts \
    -m $MODEL \
    -n $MAX_SAMPLES \
    $SHOW_RESPONSE \
    $DATA_FILE \
    $DEST_DIR/experts_test.json

# Strategy 3: Self-Refine
echo "----------------------------------------"
echo "Strategy: SELF_REFINE"
echo "----------------------------------------"
python -m drugmechcf.llmx.test_multiagent self_refine \
    -m $MODEL \
    -n $MAX_SAMPLES \
    $SHOW_RESPONSE \
    $DATA_FILE \
    $DEST_DIR/self_refine_test.json

echo
echo "=============================================="
echo "All tests completed!"
echo "=============================================="
echo "Results saved to: $DEST_DIR/"
echo "Started at: ${start_date}"
echo "Completed at:" $(date)

