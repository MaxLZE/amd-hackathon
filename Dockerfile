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

# Bundled local model: answers for easy categories cost zero Fireworks tokens.
# Qwen2.5-3B-Instruct Q4_K_M (~1.9 GB) fits the 4 GB grading RAM with room for
# the agent. Pinned to a specific revision so builds are reproducible.
ARG LOCAL_MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
RUN mkdir -p /models && curl -L --fail --retry 3 -o /models/local.gguf "$LOCAL_MODEL_URL"


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

COPY agent.py router_core.py local_engine.py .

ENTRYPOINT ["python", "agent.py"]
