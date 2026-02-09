"""
Django Integration for ArthaSeetu Brain API
Add this to your Django app (e.g., apps/ai/client.py)
"""
import requests
import hmac
import hashlib
import time
import json
from typing import Dict, Optional
from django.conf import settings
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)


class ArthaSetuBrainClient:
    """
    Secure client for ArthaSeetu Brain API with:
    - API key authentication
    - Optional request signing
    - Automatic retries
    - Response caching
    - Error handling
    """

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        signing_secret: str = None,
        timeout: int = 180,
    ):
        """
        Initialize client

        Args:
            api_url: Modal API endpoint (from settings.ARTHASETU_API_URL)
            api_key: API key (from settings.ARTHASETU_API_KEY)
            signing_secret: Optional signing secret (from settings.ARTHASETU_SIGNING_SECRET)
            timeout: Request timeout in seconds
        """
        self.api_url = (api_url or settings.ARTHASETU_API_URL).rstrip('/')
        self.api_key = api_key or settings.ARTHASETU_API_KEY
        self.signing_secret = signing_secret or getattr(settings, 'ARTHASETU_SIGNING_SECRET', None)
        self.timeout = timeout

        if not self.api_url or not self.api_key:
            raise ValueError("API URL and API key are required")

    def _sign_request(self, body: str, timestamp: int) -> str:
        """Generate HMAC signature for request"""
        if not self.signing_secret:
            return None

        message = f"{timestamp}{body}".encode()
        signature = hmac.new(
            self.signing_secret.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Dict = None,
        max_retries: int = 3,
    ) -> Dict:
        """Make HTTP request with retries and error handling"""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        # Add request signing if enabled
        if data and self.signing_secret:
            body = json.dumps(data)
            timestamp = int(time.time())
            signature = self._sign_request(body, timestamp)
            headers["X-Signature"] = signature
            headers["X-Timestamp"] = str(timestamp)

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    timeout=self.timeout,
                )

                # Log request
                logger.info(
                    f"API Request: {method} {endpoint} | "
                    f"Status: {response.status_code} | "
                    f"Attempt: {attempt + 1}/{max_retries}"
                )

                # Handle response
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise PermissionError("Invalid API key or signature")
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    raise Exception("Rate limit exceeded")
                elif response.status_code >= 500:
                    if attempt < max_retries - 1:
                        logger.warning(f"Server error. Retrying...")
                        time.sleep(2)
                        continue
                    raise Exception(f"Server error: {response.status_code}")
                else:
                    response.raise_for_status()

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed: {e}. Retrying...")
                    time.sleep(2)
                    continue
                raise Exception(f"Request failed after {max_retries} attempts: {e}")

        raise Exception(f"Request failed after {max_retries} attempts")

    def generate(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ) -> Dict:
        """
        Generate AI response

        Args:
            prompt: User question/prompt
            context: Additional context (optional)
            temperature: Sampling temperature (0.1-1.0)
            top_p: Nucleus sampling (0.1-1.0)
            max_tokens: Maximum tokens to generate (1-2048)
            use_cache: Enable response caching
            cache_ttl: Cache time-to-live in seconds

        Returns:
            Dict with keys: answer, tokens_generated, latency_seconds, request_id, timestamp
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(prompt, context, temperature, top_p, max_tokens)
            cached_response = cache.get(cache_key)
            if cached_response:
                logger.info("Cache hit for prompt")
                return cached_response

        # Make API request
        data = {
            "prompt": prompt,
            "context": context,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        try:
            response = self._make_request("v1/generate", method="POST", data=data)

            # Cache response
            if use_cache and response:
                cache.set(cache_key, response, cache_ttl)

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def health_check(self) -> Dict:
        """Check API health status"""
        try:
            return self._make_request("health", method="GET")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _get_cache_key(self, prompt: str, context: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """Generate cache key for request"""
        content = f"{prompt}|{context}|{temperature}|{top_p}|{max_tokens}"
        return f"arthasetu:brain:{hashlib.md5(content.encode()).hexdigest()}"


# ============================================================================
# DJANGO VIEW EXAMPLES
# ============================================================================

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required


@login_required
@require_http_methods(["POST"])
def ask_ai_view(request):
    """
    Django view for AI queries

    Example usage:
        POST /api/ai/ask/
        {
            "prompt": "What is the capital of France?",
            "context": "User is learning geography"
        }
    """
    try:
        data = json.loads(request.body)
        prompt = data.get('prompt', '').strip()
        context = data.get('context', '').strip()

        if not prompt:
            return JsonResponse(
                {"error": "Prompt is required"},
                status=400
            )

        # Initialize client
        client = ArthaSetuBrainClient()

        # Generate response
        result = client.generate(
            prompt=prompt,
            context=context,
            temperature=data.get('temperature', 0.6),
            max_tokens=data.get('max_tokens', 1024),
        )

        return JsonResponse({
            "success": True,
            "answer": result["answer"],
            "metadata": {
                "tokens_generated": result["tokens_generated"],
                "latency_seconds": result["latency_seconds"],
                "request_id": result["request_id"],
            }
        })

    except Exception as e:
        logger.exception("AI query failed")
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


@login_required
@require_http_methods(["GET"])
def health_check_view(request):
    """Health check endpoint for AI service"""
    try:
        client = ArthaSetuBrainClient()
        health = client.health_check()

        return JsonResponse({
            "ai_service": health,
            "django_app": "healthy",
        })
    except Exception as e:
        return JsonResponse(
            {
                "ai_service": "unhealthy",
                "error": str(e),
                "django_app": "healthy",
            },
            status=503
        )


# ============================================================================
# DJANGO MANAGEMENT COMMAND
# ============================================================================
"""
Create: apps/ai/management/commands/test_ai.py
"""
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Test ArthaSeetu Brain API connection'

    def add_arguments(self, parser):
        parser.add_argument(
            '--prompt',
            type=str,
            default='What is 2+2?',
            help='Test prompt'
        )

    def handle(self, *args, **options):
        client = ArthaSetuBrainClient()

        self.stdout.write("🔍 Testing API connection...")

        # Health check
        health = client.health_check()
        self.stdout.write(
            self.style.SUCCESS(f"✅ Health: {health.get('status')}")
        )

        # Test generation
        self.stdout.write(f"\n📝 Testing with prompt: {options['prompt']}")
        result = client.generate(prompt=options['prompt'])

        self.stdout.write(self.style.SUCCESS("\n✅ Response received:"))
        self.stdout.write(f"Answer: {result['answer']}")
        self.stdout.write(f"Tokens: {result['tokens_generated']}")
        self.stdout.write(f"Latency: {result['latency_seconds']}s")


# ============================================================================
# DJANGO SETTINGS CONFIGURATION
# ============================================================================
"""
Add to settings.py or settings/production.py:

# ArthaSeetu Brain API
ARTHASETU_API_URL = env('ARTHASETU_API_URL')  # e.g., https://your-app--arthasetu-brain-fastapi-app.modal.run
ARTHASETU_API_KEY = env('ARTHASETU_API_KEY')  # Your secret API key
ARTHASETU_SIGNING_SECRET = env('ARTHASETU_SIGNING_SECRET', default=None)  # Optional

# Cache configuration (for response caching)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': env('REDIS_URL', default='redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'arthasetu',
        'TIMEOUT': 3600,
    }
}

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'apps.ai.client': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}
"""