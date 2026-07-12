# Judging VM is linux/amd64 — build with:
#   docker buildx build --platform linux/amd64 -t <registry>/frugal-router:latest --push .
# (or let .github/workflows/docker.yml build and push it)

FROM python:3.12-slim AS build

# llama-cpp-python ships no manylinux wheel for every version, so build it
# from source; the toolchain stays in this stage only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-image.txt .
# llama.cpp defaults to -march=native, which bakes the BUILD machine's CPU
# instructions into the wheel and SIGILLs (exit 132) on any host missing
# them (e.g. CI runners have AVX-512, most other machines do not). Cap the
# build at AVX2 — the portable baseline for any cloud VM since ~2013.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON"
RUN pip install --no-cache-dir --prefix=/install -r requirements-image.txt

# Bundled local model: answers for local categories cost zero Fireworks
# tokens. Qwen2.5-1.5B beat both 3B candidates on judged accuracy (19/22
# across all 8 categories vs the 3B's 0/3 NER), decodes ~1.5x faster on
# 2 vCPU, and halves peak RSS (1.9 -> 1.1 GB file, ~1.9 GB resident).
# Will be replaced by its fine-tuned student (docs/FINETUNE.md).
ARG LOCAL_MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
RUN mkdir -p /models \
    && curl -L --fail --retry 8 --retry-all-errors --retry-delay 2 -C - \
        -o /models/local.gguf "$LOCAL_MODEL_URL"


FROM python:3.12-slim

# libgomp is llama.cpp's OpenMP runtime dependency.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /install /usr/local
COPY --from=build /models /app/models

ENV LOCAL_MODEL_PATH=/app/models/local.gguf \
    LOCAL_MODEL_THREADS=2 \
    LOCAL_MODEL_CTX=4096

COPY agent.py fireworks_client.py router_core.py local_engine.py self_heal.py .

ENTRYPOINT ["python", "agent.py"]
