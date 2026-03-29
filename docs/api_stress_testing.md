# API Stress Testing Guide

## Purpose
Measure latency and throughput for API endpoints, then identify bottlenecks.

## Prerequisite
Start the API server from project root:

```bash
uvicorn src.api_app:app --host 0.0.0.0 --port 8000
```

## Stress Test Script
Use the load script:

```bash
python scripts/stress_test_api.py --url http://127.0.0.1:8000/predict-health --requests 500 --concurrency 25
```

Optional custom payload file:

```bash
python scripts/stress_test_api.py --url http://127.0.0.1:8000/predict-health --payload-file payload.json --requests 500 --concurrency 25
```

## Metrics Reported
- total_requests
- success
- failures
- duration_seconds
- requests_per_second
- avg_latency_ms
- p50_latency_ms
- p95_latency_ms
- p99_latency_ms
- status_breakdown

## Suggested Benchmark Profiles
1. Smoke profile
- requests: 50
- concurrency: 5

2. Baseline load profile
- requests: 500
- concurrency: 25

3. Stress profile
- requests: 2000
- concurrency: 100

## Bottleneck Analysis Checklist
1. If p95/p99 rises sharply while RPS stalls:
- likely CPU saturation or model inference bottleneck.

2. If failures increase at high concurrency:
- likely worker/thread limits or timeout settings.

3. If latency is high even at low concurrency:
- check model loading strategy and payload validation overhead.

## Notes
- For stable benchmarking, use the same payload shape across runs.
- Compare runs before and after code changes to quantify impact.
