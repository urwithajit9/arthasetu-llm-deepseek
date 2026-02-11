"""
ArthaSeetu Brain - Budget Edition SIMPLIFIED (<$30/month)
Minimal configuration, maximum compatibility
"""

import modal
import os
from typing import Optional, Dict
from datetime import datetime

# ============================================================================
# CONFIGURATION - BUDGET OPTIMIZED
# ============================================================================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
APP_NAME = "arthasetu-brain"
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# BUDGET: T4 GPU, no warm containers, aggressive scaledown
GPU_CONFIG = "T4"  # $0.40/hr (cheapest)
REQUEST_TIMEOUT = 120  # 2 minutes

# Request Limits
MAX_PROMPT_LENGTH = 2000
MAX_CONTEXT_LENGTH = 4000
MAX_TOKENS_OUTPUT = 1024

# ============================================================================
# IMAGE DEFINITION
# ============================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]==0.115.0",
        "vllm==0.7.0",
        "huggingface_hub[hf_transfer]==0.26.2",
        "pydantic==2.10.3",
        "python-multipart==0.0.19",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

app = modal.App(APP_NAME, image=image)

# ============================================================================
# PERSISTENT VOLUMES
# ============================================================================
cache_vol = modal.Volume.from_name(
    f"model-weights-{ENVIRONMENT}", create_if_missing=True
)

# ============================================================================
# SECRETS
# ============================================================================
try:
    api_secret = modal.Secret.from_name("arthasetu-api")
except Exception:
    print("⚠️  Warning: arthasetu-api secret not found")
    api_secret = None


# ============================================================================
# MODEL SERVER CLASS - SIMPLIFIED
# ============================================================================
@app.cls(
    gpu=GPU_CONFIG,
    timeout=REQUEST_TIMEOUT,
    volumes={"/cache": cache_vol},
    secrets=[api_secret] if api_secret else [],
    # Modal 1.0 API (no deprecated parameters)
    min_containers=0,  # No warm containers (cold starts OK)
    scaledown_window=60,  # Scale down after 1 minute
)
class DeepSeekModel:
    """
    Simplified budget model server:
    - T4 GPU ($0.40/hr)
    - No warm containers
    - Target: <$30/month
    """

    @modal.enter()
    def load_model(self):
        """Initialize model once per container"""
        import logging
        from vllm import LLM
        import time

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading {MODEL_ID} on {GPU_CONFIG}")
        start_time = time.time()

        self.llm = LLM(
            model=MODEL_ID,
            download_dir="/cache",
            # tensor_parallel_size=1,
            gpu_memory_utilization=0.80,
            max_model_len=2048,
            trust_remote_code=True,
            enforce_eager=True,
            # dtype="auto",
            dtype="half",

        )

        load_time = time.time() - start_time
        self.logger.info(f"Model loaded in {load_time:.2f}s")

        self.request_count = 0
        self.start_time = time.time()

    @modal.method()
    def generate(
        self,
        user_prompt: str,
        context: str = "",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> Dict:
        """Generate AI response"""
        from vllm import SamplingParams
        import time

        start_time = time.time()
        self.request_count += 1
        request_id = f"req_{self.request_count}_{int(start_time)}"

        try:
            # Input validation
            if len(user_prompt) > MAX_PROMPT_LENGTH:
                raise ValueError(f"Prompt too long (max {MAX_PROMPT_LENGTH} chars)")
            if len(context) > MAX_CONTEXT_LENGTH:
                raise ValueError(f"Context too long (max {MAX_CONTEXT_LENGTH} chars)")

            # Build prompt
            if context.strip():
                full_prompt = (
                    f"Context: {context}\n\nQuestion: {user_prompt}\n\nAnswer:"
                )
            else:
                full_prompt = f"Question: {user_prompt}\n\nAnswer:"

            # Generate
            sampling_params = SamplingParams(
                temperature=max(0.1, min(temperature, 1.0)),
                top_p=max(0.1, min(top_p, 1.0)),
                max_tokens=min(max_tokens, MAX_TOKENS_OUTPUT),
                repetition_penalty=1.05,
                stop=["\n\nQuestion:", "\n\nContext:"],
            )

            results = self.llm.generate([full_prompt], sampling_params)
            output_text = results[0].outputs[0].text.strip()
            tokens_generated = len(results[0].outputs[0].token_ids)
            latency = time.time() - start_time

            self.logger.info(
                f"{request_id} | Tokens: {tokens_generated} | Latency: {latency:.2f}s"
            )

            return {
                "text": output_text,
                "tokens_generated": tokens_generated,
                "latency_seconds": round(latency, 3),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        except Exception as e:
            self.logger.error(f"{request_id} | Error: {str(e)}")
            raise

    @modal.method()
    def health_check(self) -> Dict:
        """Health check"""
        import time

        uptime = time.time() - self.start_time
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "gpu": GPU_CONFIG,
            "uptime_seconds": round(uptime, 2),
            "requests_processed": self.request_count,
        }


# ============================================================================
# FASTAPI WEB ENDPOINT
# ============================================================================
@app.function(
    min_containers=0,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("arthasetu-api")],
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI app with all imports inside function"""
    from fastapi import FastAPI, HTTPException, Header, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator
    import secrets
    import time

    web_app = FastAPI(
        title="ArthaSeetu LLM API ",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )

    # CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://yourdomain.com",  # UPDATE THIS
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # Request/Response Models
    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
        context: str = Field(default="", max_length=MAX_CONTEXT_LENGTH)
        temperature: float = Field(default=0.6, ge=0.1, le=1.0)
        top_p: float = Field(default=0.9, ge=0.1, le=1.0)
        max_tokens: int = Field(default=512, ge=1, le=MAX_TOKENS_OUTPUT)

        @field_validator("prompt", "context")
        @classmethod
        def strip_whitespace(cls, v):
            return v.strip()

    class GenerateResponse(BaseModel):
        answer: str
        tokens_generated: int
        latency_seconds: float
        request_id: str
        timestamp: str

    # Simple rate limiter
    class RateLimiter:
        def __init__(self, rpm: int = 30):
            self.rpm = rpm
            self.requests = {}

        def check(self, client_id: str) -> bool:
            now = time.time()
            minute_ago = now - 60

            if client_id in self.requests:
                self.requests[client_id] = [
                    t for t in self.requests[client_id] if t > minute_ago
                ]
            else:
                self.requests[client_id] = []

            if len(self.requests[client_id]) >= self.rpm:
                return True

            self.requests[client_id].append(now)
            return False

    rate_limiter = RateLimiter(rpm=30)

    # API key verification
    def verify_api_key(x_api_key: str = Header(...)) -> str:
        expected = os.getenv("API_KEY")
        if not expected:
            return "dev"
        if not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return x_api_key

    # Endpoints
    @web_app.post("/v1/generate", response_model=GenerateResponse)
    async def generate(
        payload: GenerateRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Generate AI response (cold start ~20-30s first time)"""
        try:
            if rate_limiter.check(api_key):
                raise HTTPException(429, "Rate limit: 30 req/min")

            model = DeepSeekModel()
            result = model.generate.remote(
                user_prompt=payload.prompt,
                context=payload.context,
                temperature=payload.temperature,
                top_p=payload.top_p,
                max_tokens=payload.max_tokens,
            )

            return GenerateResponse(
                answer=result["text"],
                tokens_generated=result["tokens_generated"],
                latency_seconds=result["latency_seconds"],
                request_id=result["request_id"],
                timestamp=result["timestamp"],
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {str(e)}")

    # @web_app.get("/health")
    # async def health():
    #     """Health check"""
    #     try:
    #         model = DeepSeekModel()
    #         health_data = model.health_check.remote()
    #         return {
    #             "status": "healthy",
    #             "version": "1.0.0-budget",
    #             "model_info": health_data,
    #             "note": "Budget mode: <$30/month target, cold starts ~20-30s",
    #         }
    #     except Exception as e:
    #         return JSONResponse(
    #             503,
    #             {
    #                 "status": "unhealthy",
    #                 "error": str(e),
    #                 "timestamp": datetime.utcnow().isoformat() + "Z",
    #             },
    #         )

    @web_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "arthasetu-brain",
            "mode": "budget",
            "cold_start_expected": True,
        }

    @web_app.get("/health/model")
    async def model_health():
        model = DeepSeekModel()
        return model.health_check.remote()

    @web_app.get("/")
    async def root():
        """API info"""
        return {
            "service": "ArthaSeetu LLM API",
            "version": "1.0.0",
            "gpu": GPU_CONFIG,
            "cost": "<$30/month",
            "cold_start": "~20-30s",
            "endpoints": {
                "generate": "POST /v1/generate",
                "health": "GET /health",
                "model_health": "GET /health/model",
            },
        }

    return web_app
