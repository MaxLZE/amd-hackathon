# Fine-tuning the local model on AMD Developer Cloud (step 3)

Goal: turn the distillation dataset (`out/distill/{train,eval}.jsonl`) into a
quantized GGUF that answers more categories locally (0 Fireworks tokens),
tersely enough to fit the 2-vCPU / ~500s grading budget.

## 0. Pick the base model

Use `out/local_bench_report.json` + `out/local_eval_*.json`: choose the
largest base whose projected run time fits after assuming the fine-tune cuts
average output length ~40% (terse training targets). As of the last bench,
Qwen2.5-1.5B-Instruct is the throughput-safe pick; Llama-3.2-3B decodes
equally fast but at 2x the RAM. Gemma 4 small (explicitly hackathon-allowed)
is worth benching once you have the HF checkpoint — same pipeline below.

## 1. Provision

- AMD Developer Cloud → GPU Droplets → 1× MI300X (192 GB HBM), ROCm 6.x
  PyTorch image (hackathon credits cover hours; a run here is <1 hr).
- `pip install llamafactory` (LLaMA-Factory has ROCm wheels and supports
  Qwen/Llama/Gemma with the same YAML) — torchtune is a fine alternative.
- Copy `out/distill/train.jsonl` and `eval.jsonl` up (scp).

## 2. Train (bf16 LoRA — no QLoRA needed at 192 GB)

Dataset is already OpenAI-messages format; register it in
`data/dataset_info.json` as `{"distill": {"file_name": "train.jsonl", "formatting": "sharegpt", "columns": {"messages": "messages"}}}`.

```yaml
# distill_lora.yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
stage: sft
finetuning_type: lora
lora_rank: 16          # 16-32; bigger != better on 10k examples
lora_alpha: 32
lora_target: all
dataset: distill
template: qwen
cutoff_len: 2048
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
lr_scheduler_type: cosine
num_train_epochs: 2.0
bf16: true
packing: true
val_size: 0.03
eval_strategy: steps
eval_steps: 50
output_dir: out-lora
```

Run: `llamafactory-cli train distill_lora.yaml`

**Stop at the overfit knee**: when eval loss flattens/rises while train loss
keeps falling, you're memorizing phrasings — exactly the failure mode the
"unseen prompt variants" rule punishes. 2 epochs is usually past enough.

## 3. Merge + convert + quantize

```bash
llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path out-lora --template qwen --export_dir merged
git clone --depth 1 https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py merged --outfile student-f16.gguf
llama.cpp/build/bin/llama-quantize student-f16.gguf student-q4_k_m.gguf Q4_K_M
```

Judge quality AFTER quantization — Q4 costs a point or two and that's the
artifact that ships.

## 4. Validate (back on the dev box)

```bash
python3 bench/local_bench.py --models student        # add its path to CANDIDATES or symlink
python3 bench/local_eval.py --model ~/models/student-q4_k_m.gguf --out out/local_eval_student.json
```

Gate per category: ≥90% judged accuracy earns a slot in
`LOCAL_MODEL_CATEGORIES` (step 4 of the roadmap, after the next judging
round). Verify decode tok/s improved effective throughput via shorter
answers, not just raw speed.

## 5. Ship

- Dockerfile: replace `LOCAL_MODEL_URL` with the student GGUF (host it on HF
  or COPY it into the build context); confirm compressed image ≤ 10 GB.
- Smoke: `taskset -c 0,1 python3 run_local.py` with the full sample suite.
- Bank the current known-good image tag before submitting (10/hour limit).

## Licenses

Qwen2.5: Apache-2.0. Llama 3.2: Llama license (fine for competition use).
Gemma: derivatives allowed with Gemma Terms attached. All fine to bundle.
