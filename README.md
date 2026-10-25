## System Specs
* Laptop GPU: NVIDIA RTX 3060, 6GB VRAM

Model: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

cd llama.cpp
source .venv/Scripts/activate
hf download Qwen/Qwen3-1.7B --local-dir /e/llm-inference-internals/models/qwen3-1.7b

hf download Qwen/Qwen3-1.7B-GGUF --include "*Q8_0.gguf" --local-dir /e/llm-inference-internals/models
mv /e/llm-inference-internals/models/*Q8_0.gguf /e/llm-inference-internals/models/qwen3-1.7b-q8_0.gguf

python convert_hf_to_gguf.py /e/llm-inference-internals/models/qwen3-1.7b --outtype bf16 --outfile /e/llm-inference-internals/models/qwen3-1.7b-bf16.gguf

cd /e/llm-inference-internals/models
ls -l
sha256sum *.gguf qwen3-1.7b/*.safetensors > hashes.txt
deactivate