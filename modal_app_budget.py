"""
ArthaSeetu Brain - Budget Edition (<$30/month)
Optimized for minimal cost with Modal 1.0 API
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

# BUDGET OPTIMIZATION: <$30/month
# T4 GPU: $0.40/hr vs L4: $0.60/hr vs A10G: $1.10/hr
# No keep-warm: Save $288/month (accept 20-30s cold start)
# Target: <75 hours/month usage = <$30/month

GPU_CONFIG = "T4"  # $0.40/hr (cheapest)
MIN_CONTAINERS = 0  # No warm containers (was keep_warm=1)
SCALEDOWN_WINDOW = 60  # Scale down after 1 minute idle
MAX_CONCURRENT = 10  # Reduced from 15

# Request Configuration
MAX_PROMPT_LENGTH = 2000  # Reduced from 4000
MAX_CONTEXT_LENGTH = 4000  # Reduced from 8000
MAX_TOKENS_OUTPUT = 1024  # Reduced from 2048
REQUEST_TIMEOUT = 120  # 2 minutes (reduced from 3)

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
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
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
# SECRETS MANAGEMENT
# ============================================================================
try:
    api_secret = modal.Secret.from_name("arthasetu-api")
except Exception:
    print("⚠️  Warning: arthasetu-api secret not found. API will be unsecured!")
    api_secret = None


# ============================================================================
# MODEL SERVER CLASS - MODAL 1.0 API
# ============================================================================
@app.cls(
    gpu=GPU_CONFIG,
    # Modal 1.0 API - Fixed deprecation warnings
    scaledown_window=SCALEDOWN_WINDOW,  # Was: container_idle_timeout
    min_containers=MIN_CONTAINERS,  # Was: keep_warm
    timeout=REQUEST_TIMEOUT,
    volumes={"/cache": cache_vol},
    secrets=[api_secret] if api_secret else [],
)
@modal.concurrent()  # Modal 1.0 API - Decorator on CLASS, not method
class DeepSeekModel:
    """
    Budget-optimized model server:
    - T4 GPU ($0.40/hr)
    - No warm containers (cold starts ~20-30s)
    - Aggressive scaledown (1 minute)
    - Target: <$30/month
    """

    @modal.enter()
    def load_model(self):
        """Initialize model once per container"""
        import logging
        from vllm import LLM
        import time

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"🚀 Loading {MODEL_ID} on {GPU_CONFIG}")
        start_time = time.time()

        # vLLM Optimizations for T4
        self.llm = LLM(
            model=MODEL_ID,
            download_dir="/cache",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,  # Lower for T4
            max_model_len=3072,  # Reduced for budget
            trust_remote_code=True,
            enforce_eager=False,
            dtype="auto",
        )

        load_time = time.time() - start_time
        self.logger.info(f"✅ Model loaded in {load_time:.2f}s")

        # Metrics
        self.request_count = 0
        self.total_tokens_generated = 0
        self.start_time = time.time()

    @modal.method()
    def generate(
        self,
        user_prompt: str,
        context: str = "",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,  # Lower default for budget
    ) -> Dict:
        """Generate response with validation and metrics"""
        from vllm import SamplingParams
        import time

        start_time = time.time()
        self.request_count += 1
        request_id = f"req_{self.request_count}_{int(start_time)}"

        try:
            # Input validation
            if len(user_prompt) > MAX_PROMPT_LENGTH:
                raise ValueError(f"Prompt exceeds {MAX_PROMPT_LENGTH} characters")
            if len(context) > MAX_CONTEXT_LENGTH:
                raise ValueError(f"Context exceeds {MAX_CONTEXT_LENGTH} characters")

            # Build prompt
            if context.strip():
                full_prompt = (
                    f"Context: {context}\n\nQuestion: {user_prompt}\n\nAnswer:"
                )
            else:
                full_prompt = f"Question: {user_prompt}\n\nAnswer:"

            # Sampling parameters with safety limits
            sampling_params = SamplingParams(
                temperature=max(0.1, min(temperature, 1.0)),
                top_p=max(0.1, min(top_p, 1.0)),
                max_tokens=min(max_tokens, MAX_TOKENS_OUTPUT),
                repetition_penalty=1.05,
                stop=["\n\nQuestion:", "\n\nContext:"],
            )

            # Generate
            results = self.llm.generate([full_prompt], sampling_params)
            output_text = results[0].outputs[0].text.strip()
            tokens_generated = len(results[0].outputs[0].token_ids)

            # Update metrics
            self.total_tokens_generated += tokens_generated
            latency = time.time() - start_time

            self.logger.info(
                f"✓ {request_id} | Tokens: {tokens_generated} | "
                f"Latency: {latency:.2f}s | Total requests: {self.request_count}"
            )

            return {
                "text": output_text,
                "tokens_generated": tokens_generated,
                "latency_seconds": round(latency, 3),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        except Exception as e:
            self.logger.error(f"✗ {request_id} | Error: {str(e)}")
            raise

    @modal.method()
    def health_check(self) -> Dict:
        """Health and metrics endpoint"""
        import time

        uptime = time.time() - self.start_time
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "gpu": GPU_CONFIG,
            "budget_mode": "enabled",
            "uptime_seconds": round(uptime, 2),
            "requests_processed": self.request_count,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_request": (
                round(self.total_tokens_generated / self.request_count, 2)
                if self.request_count > 0
                else 0
            ),
        }


# ============================================================================
# FASTAPI WEB ENDPOINT - IMPORTS INSIDE FUNCTION
# ============================================================================
@app.function(
    min_containers=0,  # No warm containers for web endpoint
    scaledown_window=120,  # 2 minutes
)
@modal.asgi_app()
def fastapi_app():
    """
    Deploy FastAPI app
    Note: All imports inside function to avoid GitHub Actions errors
    """
    # Import inside function to avoid module not found errors
    from fastapi import FastAPI, HTTPException, Header, Request, Depends
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator
    import secrets
    import time
    import hashlib
    import hmac

    web_app = FastAPI(
        title="ArthaSeetu Brain API (Budget Edition)",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )

    # CORS Configuration
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://yourdomain.com",
            "https://www.yourdomain.com",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
        max_age=3600,
    )

    # ========================================================================
    # REQUEST/RESPONSE MODELS
    # ========================================================================
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

    class ErrorResponse(BaseModel):
        error: str
        detail: Optional[str] = None
        timestamp: str

    # ========================================================================
    # SECURITY
    # ========================================================================
    class RateLimiter:
        """Simple in-memory rate limiter"""

        def __init__(self, requests_per_minute: int = 30):  # Reduced from 60
            self.requests_per_minute = requests_per_minute
            self.requests = {}

        def check_rate_limit(self, client_id: str) -> bool:
            """Returns True if rate limit exceeded"""
            now = time.time()
            minute_ago = now - 60

            if client_id in self.requests:
                self.requests[client_id] = [
                    req_time
                    for req_time in self.requests[client_id]
                    if req_time > minute_ago
                ]
            else:
                self.requests[client_id] = []

            if len(self.requests[client_id]) >= self.requests_per_minute:
                return True

            self.requests[client_id].append(now)
            return False

    rate_limiter = RateLimiter(requests_per_minute=30)

    def verify_api_key(x_api_key: str = Header(...)) -> str:
        """Verify API key from header"""
        expected_key = os.getenv("API_KEY")

        if not expected_key:
            return "dev"

        if not secrets.compare_digest(x_api_key, expected_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        return x_api_key

    # ========================================================================
    # API ENDPOINTS
    # ========================================================================
    @web_app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        responses={
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def generate_endpoint(
        request: Request,
        payload: GenerateRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Generate AI response (Budget Edition)

        Note: Cold start ~20-30s on first request after idle
        """
        try:
            # Rate limiting
            if rate_limiter.check_rate_limit(api_key):
                raise HTTPException(
                    status_code=429, detail="Rate limit exceeded (30 requests/minute)"
                )

            # Generate response
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
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    @web_app.get("/health")
    async def health_endpoint():
        """Public health check endpoint"""
        try:
            model = DeepSeekModel()
            health = model.health_check.remote()
            return {
                "status": "healthy",
                "api_version": "1.0.0-budget",
                "model_info": health,
                "note": "Budget mode: cold starts ~20-30s, <$30/month target",
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
            )

    @web_app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "service": "ArthaSeetu Brain API (Budget Edition)",
            "version": "1.0.0",
            "gpu": GPU_CONFIG,
            "cost_target": "<$30/month",
            "cold_start": "~20-30s after idle",
            "endpoints": {
                "generate": "POST /v1/generate",
                "health": "GET /health",
            },
            "authentication": "Required: X-API-Key header",
        }

    return web_app
