
# vllm 시작
```
vllm serve Qwen/Qwen3-4B-Thinking-2507-FP8 --max-model-len 200000 --reasoning-parser deepseek_r1
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