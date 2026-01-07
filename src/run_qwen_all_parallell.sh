#!/bin/bash
#
#  Run all the Counterfactual queries against LLM models hosted by `vllm`.
#
#  See `MODELS` below for list of models invoked by this script.
#  Output file names follow the convention expected by `drugmechcf.exp.cfvariances.read_all_session_files()`,
#  the root name is based on the query-data file, with an optional "-k" suffix indicating closed-world run.
#
#  To run:
#	$ source $PROJDIR/.venv/bin/activate
#	(dmcf) $ cd $PROJDIR/src
#	(dmcf) $ ./run_qwen_all.sh
#

CMD=`basename $0`

start_date=`date`
echo "Start: ${start_date}"


# -- Paths

DATA_DIR=../Data/Counterfactuals

DEST_DIR=../Data/Sessions/Models


# -- Data files

ADD_LINK_FILES=($DATA_DIR/AddLink_*.json)

CHANGE_LINK_FILES=($DATA_DIR/change_*.json)

DELETE_LINK_FILES=($DATA_DIR/delete_*.json)

DATA_FILES=("${ADD_LINK_FILES[@]}"  "${CHANGE_LINK_FILES[@]}"  "${DELETE_LINK_FILES[@]}")

echo
echo "Data files:"
echo "   nbr Add-Link files = ${#ADD_LINK_FILES[@]}"
echo "   nbr Invert-Link files = ${#CHANGE_LINK_FILES[@]}"
echo "   nbr Delete-Link files = ${#DELETE_LINK_FILES[@]}"
echo "   Total nbr files = ${#DATA_FILES[@]}"
echo


# -- Options

MODELS=("Qwen3-4B-Thinking-2507-FP8")

# ... Open-world and Closed-world
OPTIONS=("" "-k")

# ... Number of worker threads for parallel processing
# Increase this to process more samples in parallel (e.g., 4, 8, 16)
# Note: Too many threads may overwhelm the vLLM server
N_WORKER_THREADS=1


# -- GPU assignments and ports
# ADD_LINK_FILES -> GPU 0, port 8000
# CHANGE_LINK_FILES -> GPU 1, port 8001
# DELETE_LINK_FILES -> GPU 2, port 8002
#
# NOTE: Make sure vLLM servers are running on each GPU with corresponding ports:
#   GPU 0: vllm serve ... --port 8000 (already running)
#   GPU 1: CUDA_VISIBLE_DEVICES=1 vllm serve ... --port 8001
#   GPU 2: CUDA_VISIBLE_DEVICES=2 vllm serve ... --port 8002

# Function to process files for a specific GPU and port
process_files_on_gpu() {
    local gpu_id=$1
    local port=$2
    local model=$3
    local dest=$4
    shift 4
    local files=("$@")
    
    for dataf in "${files[@]}"; do
        dataf_base=`basename ${dataf}`
        rootf="${dataf_base%.*}"

        for opt in "${OPTIONS[@]}"; do
            if [ -z "$opt" ]; then
                outroot="$rootf"
            else
                outroot="${rootf}${opt}"
            fi

            jsonf="${dest}/${outroot}.json"
            logf="${dest}/${outroot}_log.txt"

            cmd_opts="-m ${model} ${opt} -w ${N_WORKER_THREADS}"

            echo "----------------------------------------"
            echo "[GPU ${gpu_id}, Port ${port}] opts = ${cmd_opts}"
            echo "[GPU ${gpu_id}, Port ${port}] JSON file = ${jsonf}"
            echo "[GPU ${gpu_id}, Port ${port}] LOG file = ${logf}"
            echo

            VLLM_PORT=${port} CUDA_VISIBLE_DEVICES=${gpu_id} python -m drugmechcf.exp.test_vllm batch ${cmd_opts} ${dataf} ${jsonf} > ${logf} 2>&1

            echo "[GPU ${gpu_id}, Port ${port}] ${CMD}:  ${cmd_opts} ${dataf} ... completed"
            echo
        done
    done
}

# -- Process files in parallel on different GPUs

for model in "${MODELS[@]}"; do

    DEST="$DEST_DIR/$model"

    if [ ! -d "${DEST}" ]; then
        echo "Creating dir:"
        mkdir -pv $DEST
    fi

    # Process ADD_LINK_FILES on GPU 0, port 8000 (background)
    # NOTE: GPU 0 is already running ADD_LINK, so this is commented out
    # if [ ${#ADD_LINK_FILES[@]} -gt 0 ]; then
    #     echo "Starting ADD_LINK processing on GPU 0, port 8000..."
    #     process_files_on_gpu 0 8000 "${model}" "${DEST}" "${ADD_LINK_FILES[@]}" &
    #     ADD_LINK_PID=$!
    # fi

    # Process CHANGE_LINK_FILES on GPU 1, port 8001 (background)
    if [ ${#CHANGE_LINK_FILES[@]} -gt 0 ]; then
        echo "Starting CHANGE_LINK processing on GPU 1, port 8001..."
        process_files_on_gpu 1 8001 "${model}" "${DEST}" "${CHANGE_LINK_FILES[@]}" &
        CHANGE_LINK_PID=$!
    fi

    # Process DELETE_LINK_FILES on GPU 2, port 8002 (background)
    if [ ${#DELETE_LINK_FILES[@]} -gt 0 ]; then
        echo "Starting DELETE_LINK processing on GPU 2, port 8002..."
        process_files_on_gpu 2 8002 "${model}" "${DEST}" "${DELETE_LINK_FILES[@]}" &
        DELETE_LINK_PID=$!
    fi

    # Wait for all background processes to complete
    echo "Waiting for all GPU processes to complete..."
    wait

    echo "All GPU processes completed for model: ${model}"
    echo

done

echo "-- ${CMD} --"
echo "Started at: ${start_date}"
echo "All Completed at:" `date`
