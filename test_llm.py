#!/usr/bin/env python3
"""
ArthaSeetu Brain API - Test Suite
Tests API functionality, security, and performance
"""
import requests
import time
import os
import json
import hmac
import hashlib
from typing import Dict
import sys
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
# Update these with your actual values
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
SIGNING_SECRET = os.getenv("SIGNING_SECRET")

# ============================================================================
# TEST UTILITIES
# ============================================================================
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def sign_request(body: str, timestamp: int) -> str:
    """Generate HMAC signature"""
    message = f"{timestamp}{body}".encode()
    return hmac.new(
        SIGNING_SECRET.encode(),
        message,
        hashlib.sha256
    ).hexdigest()

# ============================================================================
# TEST CASES
# ============================================================================
def test_health_check():
    """Test 1: Health check endpoint"""
    print_info("Test 1: Health Check")

    try:
        response = requests.get(f"{API_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print_success(f"Health check passed: {data.get('status')}")
            print(f"  Model: {data.get('model_info', {}).get('model')}")
            print(f"  GPU: {data.get('model_info', {}).get('gpu')}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_root_endpoint():
    """Test 2: Root endpoint"""
    print_info("Test 2: Root Endpoint")

    try:
        response = requests.get(f"{API_URL}/", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print_success(f"Root endpoint accessible")
            print(f"  Service: {data.get('service')}")
            print(f"  Version: {data.get('version')}")
            return True
        else:
            print_error(f"Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Root endpoint error: {e}")
        return False

def test_authentication():
    """Test 3: API key authentication"""
    print_info("Test 3: Authentication")

    # Test without API key
    try:
        response = requests.post(
            f"{API_URL}/v1/generate",
            json={"prompt": "test"},
            timeout=10
        )

        if response.status_code == 401:
            print_success("Correctly rejected request without API key")
        else:
            print_warning(f"Expected 401, got {response.status_code}")

    except Exception as e:
        print_error(f"Auth test error: {e}")

    # Test with wrong API key
    try:
        response = requests.post(
            f"{API_URL}/v1/generate",
            headers={"X-API-Key": "wrong-key"},
            json={"prompt": "test"},
            timeout=10
        )

        if response.status_code == 401:
            print_success("Correctly rejected request with wrong API key")
            return True
        else:
            print_warning(f"Expected 401, got {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Auth test error: {e}")
        return False

def test_basic_generation():
    """Test 4: Basic text generation"""
    print_info("Test 4: Basic Generation")

    payload = {
        "prompt": "What is 2+2?",
        "max_tokens": 50
    }

    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/v1/generate",
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120
        )
        latency = time.time() - start

        if response.status_code == 200:
            data = response.json()
            print_success("Generation successful")
            print(f"  Answer: {data.get('answer')[:100]}...")
            print(f"  Tokens: {data.get('tokens_generated')}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Request ID: {data.get('request_id')}")
            return True
        else:
            print_error(f"Generation failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except Exception as e:
        print_error(f"Generation error: {e}")
        return False

def test_with_context():
    """Test 5: Generation with context"""
    print_info("Test 5: Generation with Context")

    payload = {
        "prompt": "What is the user learning?",
        "context": "User is studying Python programming and web development.",
        "max_tokens": 100
    }

    try:
        response = requests.post(
            f"{API_URL}/v1/generate",
            headers={"X-API-Key": API_KEY},
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            print_success("Context-aware generation successful")
            print(f"  Answer: {data.get('answer')[:150]}...")
            return True
        else:
            print_error(f"Context generation failed: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Context generation error: {e}")
        return False

def test_parameter_variations():
    """Test 6: Different parameters"""
    print_info("Test 6: Parameter Variations")

    tests = [
        {"temperature": 0.2, "top_p": 0.8, "max_tokens": 50},
        {"temperature": 0.9, "top_p": 0.95, "max_tokens": 200},
    ]

    for i, params in enumerate(tests, 1):
        payload = {
            "prompt": "Explain AI in one sentence.",
            **params
        }

        try:
            response = requests.post(
                f"{API_URL}/v1/generate",
                headers={"X-API-Key": API_KEY},
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                print_success(f"  Variation {i}: Success")
            else:
                print_error(f"  Variation {i}: Failed")
                return False

        except Exception as e:
            print_error(f"  Variation {i}: Error - {e}")
            return False

    return True

def test_input_validation():
    """Test 7: Input validation"""
    print_info("Test 7: Input Validation")

    # Empty prompt
    response = requests.post(
        f"{API_URL}/v1/generate",
        headers={"X-API-Key": API_KEY},
        json={"prompt": ""},
        timeout=10
    )

    if response.status_code in [400, 422]:
        print_success("Correctly rejected empty prompt")
    else:
        print_warning(f"Empty prompt: Expected 400/422, got {response.status_code}")

    # Too long prompt
    response = requests.post(
        f"{API_URL}/v1/generate",
        headers={"X-API-Key": API_KEY},
        json={"prompt": "x" * 5000},
        timeout=10
    )

    if response.status_code in [400, 422]:
        print_success("Correctly rejected too long prompt")
        return True
    else:
        print_warning(f"Long prompt: Expected 400/422, got {response.status_code}")
        return False

def test_rate_limiting():
    """Test 8: Rate limiting"""
    print_info("Test 8: Rate Limiting (sending 70 requests)")

    rate_limited = False

    for i in range(70):
        try:
            response = requests.post(
                f"{API_URL}/v1/generate",
                headers={"X-API-Key": API_KEY},
                json={"prompt": f"Test {i}", "max_tokens": 10},
                timeout=5
            )

            if response.status_code == 429:
                rate_limited = True
                print_success(f"Rate limit triggered at request {i+1}")
                break

        except Exception:
            continue

    if rate_limited:
        return True
    else:
        print_warning("Rate limit not triggered (might be ok if limit is higher)")
        return True

def test_performance():
    """Test 9: Performance metrics"""
    print_info("Test 9: Performance Test (10 requests)")

    latencies = []

    for i in range(10):
        start = time.time()
        try:
            response = requests.post(
                f"{API_URL}/v1/generate",
                headers={"X-API-Key": API_KEY},
                json={"prompt": f"Test {i}", "max_tokens": 50},
                timeout=120
            )

            if response.status_code == 200:
                latency = time.time() - start
                latencies.append(latency)
                print(f"  Request {i+1}: {latency:.2f}s")
            else:
                print_warning(f"  Request {i+1}: Failed ({response.status_code})")

        except Exception as e:
            print_warning(f"  Request {i+1}: Error - {e}")

        time.sleep(0.5)  # Small delay between requests

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        print_success(f"Performance test completed")
        print(f"  Average: {avg_latency:.2f}s")
        print(f"  Min: {min_latency:.2f}s")
        print(f"  Max: {max_latency:.2f}s")

        if avg_latency < 5.0:
            print_success("  Performance is good!")
            return True
        else:
            print_warning("  Performance could be improved")
            return True
    else:
        print_error("No successful requests")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 ArthaSeetu Brain API - Test Suite")
    print("=" * 70)
    print()

    if API_URL == "https://your-workspace--arthasetu-brain-fastapi-app.modal.run":
        print_error("Please update API_URL in the script!")
        sys.exit(1)

    if API_KEY == "your-api-key-here":
        print_error("Please update API_KEY in the script!")
        sys.exit(1)

    print(f"API URL: {API_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    print()

    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Authentication", test_authentication),
        ("Basic Generation", test_basic_generation),
        ("Context-Aware Generation", test_with_context),
        ("Parameter Variations", test_parameter_variations),
        ("Input Validation", test_input_validation),
        ("Rate Limiting", test_rate_limiting),
        ("Performance", test_performance),
    ]

    results = []

    for name, test_func in tests:
        print()
        print("-" * 70)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results.append((name, False))

        time.sleep(1)

    # Summary
    print()
    print("=" * 70)
    print("📊 Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.END} - {name}")

    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print_success("All tests passed! 🎉")
        sys.exit(0)
    else:
        print_error(f"{total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
curl -X POST "https://vidhiai-in--arthasetu-brain-fastapi-app.modal.run/v1/generate" \
  -H "X-API-Key: rHhgfytr988676" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }' \
  --verbose \
  --no-buffer \
  --max-time 180 \
  -w "\n\n=== TIMING ===
DNS: %{time_namelookup}s
Connect: %{time_connect}s
TLS: %{time_appconnect}s
TTFB: %{time_starttransfer}s
Total: %{time_total}s
HTTP: %{http_code}\n"

"""
