import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean


def single_request(url, payload_bytes, timeout):
    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            _ = response.read()
            status = response.status
    except urllib.error.HTTPError as err:
        status = err.code
    except Exception:
        status = 0
    latency_ms = (time.perf_counter() - start) * 1000.0

    return status, latency_ms


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    weight = k - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def run_load_test(url, total_requests, concurrency, timeout, payload):
    payload_bytes = json.dumps(payload).encode("utf-8")

    start_time = time.perf_counter()
    latencies = []
    statuses = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(single_request, url, payload_bytes, timeout)
            for _ in range(total_requests)
        ]

        for fut in as_completed(futures):
            status, latency_ms = fut.result()
            statuses.append(status)
            latencies.append(latency_ms)

    total_time = time.perf_counter() - start_time

    success = sum(1 for status in statuses if 200 <= status < 300)
    failures = len(statuses) - success
    rps = len(statuses) / total_time if total_time > 0 else 0.0

    result = {
        "total_requests": len(statuses),
        "success": success,
        "failures": failures,
        "duration_seconds": round(total_time, 4),
        "requests_per_second": round(rps, 4),
        "avg_latency_ms": round(mean(latencies), 4) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 50), 4),
        "p95_latency_ms": round(percentile(latencies, 95), 4),
        "p99_latency_ms": round(percentile(latencies, 99), 4),
        "status_breakdown": {
            str(code): sum(1 for status in statuses if status == code)
            for code in sorted(set(statuses))
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Simple API stress test for POST endpoints.")
    parser.add_argument("--url", required=True, help="Endpoint URL to test")
    parser.add_argument("--requests", type=int, default=200, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=20, help="Parallel workers")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    parser.add_argument(
        "--payload-file",
        default=None,
        help="Optional JSON file containing request body",
    )
    args = parser.parse_args()

    if args.payload_file:
        with open(args.payload_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = {
            "features": {
                "Store": 1,
                "DayOfWeek": 2,
                "Promo": 1,
                "StateHoliday": 0,
                "SchoolHoliday": 0,
                "StoreType": 1,
                "Assortment": 1,
                "CompetitionDistance": 500.0,
                "Year": 2015,
                "Month": 7,
                "Day": 15,
            }
        }

    result = run_load_test(
        url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
        payload=payload,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()