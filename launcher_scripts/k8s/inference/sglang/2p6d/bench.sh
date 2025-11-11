#!/bin/bash

python3 -m sglang.bench_serving \
  --backend sglang \
  --host 172.17.157.235 \
  --port 8080 \
  --model /model/models/default/deepseek-r1/ \
  --dataset-name random \
  --dataset-path /dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 6100 \
  --random-output-len 1300 \
  --num-prompts 1000 \
  --random-range-ratio 0.999 \
  --max-concurrency 160 \
  --request-rate 3.6
  # --extra-request-body '{"chat_template_kwargs": {"thinking": true}}' \

# single instance bench
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1  \
  --port 30000 \
  --model /models/default/deepseek-v3-1/ \
  --dataset-name random \
  --dataset-path /dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 1 \
  --random-output-len 1300 \
  --num-prompts 1000 \
  --max-concurrency 1000 \
  --random-range-ratio 0.999 \
  --request-rate 2

# curl
# add `"chat_template_kwargs": {"thinking": true}` to enable thinking
curl https://console.scitix.ai/siflow/cetus/simaas/admin/deepseek-v3-1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3.1",
    "messages": [
      {"role": "user", "content": "9.11和9.8哪个更大？"}
    ],
    "chat_template_kwargs": {"thinking": true}
  }'

python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url https://console.scitix.ai/siflow/cetus/simaas/admin/deepseek-v3-1 \
  --model /models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/ \
  --dataset-name random \
  --dataset-path /volume/datasets/ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 6100 \
  --random-output-len 1300 \
  --num-prompts 1000 \
  --random-range-ratio 0.999 \
  --max-concurrency 180 \
  --request-rate 3.6 \
  --ttft-mode from_request_arrival

python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://ingress-deepseek-v3-1-default.t-simaas-admin.svc.cluster.local \
  --model /models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/ \
  --dataset-name random \
  --dataset-path /volume/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 3000 \
  --random-output-len 2240 \
  --num-prompts 1000 \
  --random-range-ratio 0.999 \
  --request-rate 1 \
  --ttft-mode from_request_arrival


python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url http://ingress-qwen235-fork-default.t-simaas-admin.svc.cluster.local \
  --model /models/preset/Qwen/Qwen3-235B-A22B/v1.0/ \
  --dataset-name random \
  --dataset-path /volume/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 2500 \
  --random-output-len 24 \
  --num-prompts 1000 \
  --random-range-ratio 0.999 \
  --request-rate 20 \
  --ttft-mode from_request_arrival
