import modal

# Configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
APP_NAME = "arthasetu-llm"

# 1. UPDATED: Explicitly install fastapi[standard]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.7.0", # Use a stable 2025/2026 version
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)
cache_vol = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

@app.cls(
    gpu="A10G",
    scaledown_window=120, # 2. UPDATED: renamed from container_idle_timeout
    volumes={"/cache": cache_vol},
    image=image
)
class DeepSeekModel:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        # vLLM will download to /cache if not present
        self.llm = LLM(model=MODEL_ID, download_dir="/cache")

    @modal.method()
    def generate(self, user_prompt: str, context: str = ""):
        from vllm import SamplingParams

        # DeepSeek-R1 prompt format
        full_prompt = f"Context: {context}\n\nQuestion: {user_prompt}\n\nAnswer:"

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=1024
        )

        results = self.llm.generate([full_prompt], sampling_params)
        return results[0].outputs[0].text

@app.function()
@modal.fastapi_endpoint(method="POST") # 3. UPDATED: renamed from web_endpoint
def ask(data: dict):
    model = DeepSeekModel()
    answer = model.generate.remote(
        user_prompt=data.get("prompt"),
        context=data.get("context", "")
    )
    return {"answer": answer}