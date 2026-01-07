# vllm 시작
```
# 하나만 임의로 일단 실행 (GPU 0)
vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 --max-model-len 200000 --reasoning-parser deepseek_r1
```
```
# run_qwen_all.sh를 위해서 각 GPU마다 vllm server 실행
# GPU 0에서 vLLM 서버 실행 (포트 8001)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --port 8001 \
    --max-model-len 200000 \
    --reasoning-parser deepseek_r1

# GPU 1에서 vLLM 서버 실행 (포트 8001)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --port 8001 \
    --max-model-len 200000 \
    --reasoning-parser deepseek_r1

# GPU 2에서 vLLM 서버 실행 (포트 8002)
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --port 8002 \
    --max-model-len 200000 \
    --reasoning-parser deepseek_r1


# GPU 3에서 vLLM 서버 실행 (포트 8001)
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 \
    --port 8008 \
    --max-model-len 100000 \
    --reasoning-parser deepseek_r1
```

# 일단 무조건 src로 이동하고 시작
```
cd src
```
# [Option 1] qwen 성능 뽑아보기
```
./run_qwen_all.sh 
```

# [Option 2] 그냥 임시로 작게 qwen 돌려보고 싶으면
python -m drugmechcf.exp.test_vllm batch -m Qwen3-4B-Thinking-2507-FP8 -n 3 ../Data/Counterfactuals/AddLink_neg_dpi_r1k.json ../Data/Sessions/Models/Qwen3-4B-Thinking-2507-FP8/AddLink_neg_dpi_r1k_test.json > ../Data/Sessions/Models/Qwen3-4B-Thinking-2507-FP8/AddLink_neg_dpi_r1k_test_log.txt 2>&1


# Data/Sessions/Models/모델명/ 안에 있는 파일들 해석
* 파일명에 "_dpi_" 가 포함되어있으면 : surface 
* 파일명에 "-k" 가 포함되어 있으면 : open world setting
* 파일명에 "-k" 가 없으면: closed world setting

# toy example
```
python toy_multiagent.py debate "5 + 3 * 2 = ?"
```