# 💰 Budget Edition - Under $30/Month Guide

## ✅ What Changed to Hit <$30/Month

### Cost Breakdown

**Original Configuration:**
```python
GPU: L4 ($0.60/hr)
Keep-warm: 1 container
= $0.60 × 24 hours × 30 days = $432/month
With scaledown: ~$100-200/month
```

**Budget Configuration:**
```python
GPU: T4 ($0.40/hr)
Keep-warm: 0 containers (NO warm containers)
Scaledown: 60 seconds (aggressive)
= $0.40 × actual usage hours only
Target: <75 hours/month = <$30/month
```

---

## 💡 Budget vs Premium Comparison

| Feature | **Budget (<$30/mo)** | **Premium ($100-200/mo)** |
|---------|----------------------|---------------------------|
| **GPU** | T4 ($0.40/hr) | L4 ($0.60/hr) |
| **Warm Containers** | 0 (none) | 1 (always ready) |
| **Cold Start** | 20-30 seconds | <1 second |
| **Scaledown** | 60 seconds | 300 seconds |
| **Max Tokens** | 1024 | 2048 |
| **Concurrent** | 10 requests | 15 requests |
| **Rate Limit** | 30 req/min | 60 req/min |
| **Best For** | Personal, Dev, Low traffic | Production, High traffic |

---

## 📊 Budget Edition Usage Limits

To stay under $30/month at $0.40/hr:

```
$30 ÷ $0.40/hr = 75 hours/month maximum
75 hours ÷ 30 days = 2.5 hours/day average

Practical scenarios:
✅ 150 requests/day @ 1min each = 2.5 hrs/day = $30/month
✅ 300 requests/day @ 30sec each = 2.5 hrs/day = $30/month
✅ Light usage: <100 requests/day = $10-15/month
❌ Heavy usage: >500 requests/day = >$30/month (upgrade to Premium)
```

---

## 🔧 What Was Changed (Technical)

### 1. Fixed Modal 1.0 Deprecation Warnings

**Old (Deprecated):**
```python
@app.cls(
    keep_warm=1,                    # ❌ Deprecated
    container_idle_timeout=600,     # ❌ Deprecated
    allow_concurrent_inputs=15,     # ❌ Deprecated
)
```

**New (Modal 1.0):**
```python
@app.cls(
    min_containers=0,               # ✅ New API
    scaledown_window=60,            # ✅ New API
)
class DeepSeekModel:
    @modal.method()
    @modal.concurrent()             # ✅ New decorator
    def generate(...):
```

### 2. Fixed Import Error

**Problem:**
```python
# ❌ Imports at module level cause errors in GitHub Actions
from fastapi import FastAPI
```

**Solution:**
```python
# ✅ All imports inside function
@app.function()
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI  # Import here
    ...
```

### 3. Budget Optimizations

```python
# GPU downgrade
GPU_CONFIG = "T4"  # Was: "L4"

# No warm containers
MIN_CONTAINERS = 0  # Was: keep_warm=1

# Aggressive scaledown
SCALEDOWN_WINDOW = 60  # Was: 300

# Reduced limits
MAX_TOKENS_OUTPUT = 1024  # Was: 2048
MAX_CONCURRENT = 10  # Was: 15
RATE_LIMIT = 30/min  # Was: 60/min
```

---

## 🚀 Deployment Commands

### Deploy Budget Edition

```bash
# 1. Create secret (if not done)
modal secret create arthasetu-api \
  API_KEY="your-secret-key" \
  SIGNING_SECRET="your-signing-secret"

# 2. Deploy budget version
modal deploy modal_app_budget.py

# 3. Get URL
# https://your-workspace--arthasetu-brain-fastapi-app.modal.run
```

### Test It

```bash
# First request (expect 20-30s cold start)
time curl -X POST https://your-url.modal.run/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }'

# Second request immediately after (should be fast ~1-2s)
time curl -X POST https://your-url.modal.run/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "prompt": "What is 3+3?",
    "max_tokens": 50
  }'

# Wait 2 minutes, then request again (cold start again)
```

---

## 📈 Cost Monitoring

### Track Your Usage

```bash
# Check Modal dashboard
https://modal.com/usage

# Set billing alerts
1. Go to https://modal.com/settings/billing
2. Set alert at $25 (before hitting $30)
3. Get email when approaching limit
```

### Estimate Monthly Cost

```python
# Use the cost calculator
python cost_calculator.py --mode quick

# Output example:
# T4 GPU: $0.40/hr
# Average 2 hours/day usage
# Estimated: $24/month ✅
```

### Monitor in Django

```python
# Track API calls in Django
from django.core.cache import cache

def track_api_usage():
    """Track daily API calls"""
    today = datetime.now().date().isoformat()
    key = f"api_calls:{today}"

    calls = cache.get(key, 0)
    cache.set(key, calls + 1, 86400)  # 24 hour TTL

    # Alert if approaching limits
    if calls > 250:  # ~2.5 hours at 1min/request
        logger.warning(f"High API usage: {calls} calls today")
```

---

## ⚡ Optimization Tips

### 1. Cache Aggressively in Django

```python
from django.core.cache import cache
from apps.ai.client import ArthaSetuBrainClient

def get_ai_response(prompt, context=""):
    # Check cache first
    cache_key = f"ai:{hash(prompt+context)}"
    cached = cache.get(cache_key)
    if cached:
        return cached  # Save API call!

    # Only call API if not cached
    client = ArthaSetuBrainClient()
    result = client.generate(prompt, context)

    # Cache for 1 hour
    cache.set(cache_key, result, 3600)
    return result
```

**Savings:** 50-70% reduction in API calls

### 2. Batch Similar Requests

```python
# Bad: Multiple sequential calls
for question in questions:
    result = client.generate(question)  # 10 API calls

# Good: Batch into one call
all_questions = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
result = client.generate(f"Answer these questions:\n{all_questions}")
# 1 API call (10x cheaper!)
```

### 3. Use Shorter max_tokens

```python
# Default in budget mode
max_tokens=512  # Good for most cases

# For yes/no or short answers
max_tokens=50   # 10x cheaper per request

# Only use high values when needed
max_tokens=1024  # For longer explanations
```

### 4. Implement Request Queuing (Django)

```python
# Instead of making API call immediately
from django_q.tasks import async_task

def process_ai_request(user_id, prompt):
    # Queue the request
    async_task(
        'apps.ai.tasks.generate_response',
        user_id,
        prompt
    )
    return "Processing... check back in 30 seconds"

# User doesn't wait for cold start
```

---

## 🎯 When to Upgrade to Premium

Upgrade if you experience:

- ❌ >500 requests/day consistently
- ❌ Cold starts are annoying users
- ❌ Monthly cost exceeds $30
- ❌ Need <1s response time
- ❌ Peak traffic requires instant response

Premium advantages:
- ✅ No cold starts (always warm)
- ✅ Faster GPU (L4 vs T4)
- ✅ Higher rate limits (60 vs 30 req/min)
- ✅ Better for production traffic

---

## 📊 Real-World Budget Scenarios

### Scenario 1: Personal Blog (Light Usage)
```
Users: 100/day
AI queries: 50/day @ 30sec each = 25 min/day
Monthly: 25 min × 30 days = 12.5 hours
Cost: 12.5 hrs × $0.40 = $5/month ✅
```

### Scenario 2: Small Team App (Medium Usage)
```
Users: 500/day
AI queries: 200/day @ 45sec each = 150 min/day
Monthly: 150 min × 30 days = 75 hours
Cost: 75 hrs × $0.40 = $30/month ✅ (at limit)
```

### Scenario 3: Growing Startup (Heavy Usage)
```
Users: 2000/day
AI queries: 800/day @ 1min each = 800 min/day
Monthly: 800 min × 30 days = 400 hours
Cost: 400 hrs × $0.40 = $160/month ❌ (upgrade to Premium)
```

---

## ✅ Budget Edition Checklist

Before deploying:

- [ ] Understand cold starts (20-30s first request after idle)
- [ ] Set up billing alerts in Modal dashboard at $25
- [ ] Implement caching in Django (saves 50-70% costs)
- [ ] Monitor usage for first week
- [ ] Keep max_tokens low (512 default)
- [ ] Batch requests when possible
- [ ] Use async processing for non-urgent requests

---

## 🔄 Switching Between Versions

You can deploy both and switch:

```bash
# Deploy budget version
modal deploy modal_app_budget.py

# Or deploy premium version
modal deploy modal_app.py

# Update Django .env to point to desired endpoint
ARTHASETU_API_URL=https://...modal.run  # Choose one
```

---

## 💡 Summary

**Budget Edition (<$30/month):**
- T4 GPU, no warm containers
- Cold starts ~20-30s
- Perfect for: Personal projects, dev, light traffic
- Requires: Django caching, usage monitoring

**Premium Edition ($100-200/month):**
- L4 GPU, 1 warm container
- No cold starts
- Perfect for: Production, business apps, high traffic
- Better: User experience, reliability

**Recommendation:** Start with Budget, upgrade when you hit $30/month consistently.

🚀 **You're ready to deploy under budget!**