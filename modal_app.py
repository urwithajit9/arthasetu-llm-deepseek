import modal

# Configuration
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
APP_NAME = "arthasetu-brain"

# Define the container environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3",
        "huggingface_hub[hf_transfer]",
        "flashinfer-python==0.1.6"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(APP_NAME)

# Persistent volume to store model weights (prevents re-downloading)
cache_vol = modal.Volume.from_name("model-weights-vol", create_if_missing=True)

@app.cls(
    gpu="A10G",  # Perfect balance for 7B models
    container_idle_timeout=120, # Shut down after 2 mins of inactivity
    volumes={"/cache": cache_vol},
    image=image
)
class DeepSeekModel:
    @modal.enter()
    def load_model(self):
        from vllm import LLM
        self.llm = LLM(model=MODEL_ID, download_dir="/cache")

    @modal.method()
    def generate(self, user_prompt: str, context: str = ""):
        from vllm import SamplingParams

        # Construct the reasoning-focused prompt
        full_prompt = f"Context: {context}\n\nQuestion: {user_prompt}\n\nAnswer:"

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=1024
        )

        results = self.llm.generate([full_prompt], sampling_params)
        return results[0].outputs[0].text

@app.function()
@modal.web_endpoint(method="POST")
def ask(data: dict):
    """The API endpoint for your Django app to hit"""
    model = DeepSeekModel()
    answer = model.generate.remote(
        user_prompt=data.get("prompt"),
        context=data.get("context", "")
    )
    return {"answer": answer}