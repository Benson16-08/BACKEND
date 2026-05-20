import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

import httpx

API_BASE = "http://localhost:8000"
QUERY_ENDPOINT = f"{API_BASE}/api/v1/query"
HEALTH_ENDPOINT = f"{API_BASE}/api/v1/health"

NFR_MEAN_TARGET_S = 5.0
NFR_SUCCESS_RATE = 1.0

CONCURRENT_USERS_LEVELS = [1, 5, 10]
REQUESTS_PER_LEVEL = 5

TEST_QUERIES = [
    {"symptoms": "fever with chills and severe headache for three days"},
    {"symptoms": "persistent cough with difficulty breathing and chest pain"},
    {"symptoms": "abdominal pain with nausea and fever for two days"},
    {"symptoms": "joint pain and skin rash with mild fever"},
    {"symptoms": "diarrhoea and vomiting with dehydration signs"},
    {"symptoms": "high fever with rigors and loss of appetite"},
    {"symptoms": "sore throat with difficulty swallowing and neck pain"},
    {"symptoms": "confusion and high temperature in elderly patient"},
    {"symptoms": "rash with fever and painful lymph nodes"},
    {"symptoms": "heavy menstrual bleeding with dizziness and fatigue"},
]

async def _single_request(
    client: httpx.AsyncClient,
    query: dict,
    user_id: int,
    request_id: int,
) -> Tuple[float, int, bool]:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            QUERY_ENDPOINT,
            json=query,
            timeout=90.0,
        )
        elapsed = time.perf_counter() - t0
        success = r.status_code == 200
        return elapsed, r.status_code, success
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [user={user_id} req={request_id}] ERROR: {exc}")
        return elapsed, 0, False

async def _run_concurrent_level(
    n_users: int,
    queries: List[dict],
) -> List[Tuple[float, int, bool]]:
    async def _user_session(user_id: int) -> List[Tuple[float, int, bool]]:
        results = []
        async with httpx.AsyncClient() as client:
            for req_id, query in enumerate(queries):
                result = await _single_request(client, query, user_id, req_id)
                results.append(result)
                await asyncio.sleep(0.1)
        return results

    tasks = [_user_session(i) for i in range(n_users)]
    all_results = await asyncio.gather(*tasks)

    flat: List[Tuple[float, int, bool]] = []
    for user_results in all_results:
        flat.extend(user_results)
    return flat


def _compute_stats(results: List[Tuple[float, int, bool]]) -> dict:
    times = [r[0] for r in results]
    statuses = [r[1] for r in results]
    successes = [r[2] for r in results]

    times_sorted = sorted(times)
    n = len(times)

    def _percentile(p: float) -> float:
        idx = int(n * p / 100)
        return times_sorted[min(idx, n - 1)]

    return {
        "n": n,
        "success_rate": sum(successes) / n if n > 0 else 0,
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "p95_s": _percentile(95),
        "p99_s": _percentile(99),
        "stdev_s": statistics.stdev(times) if n > 1 else 0,
        "status_codes": {str(s): statuses.count(s) for s in set(statuses)},
    }


def _divider(char: str = "─", width: int = 65) -> str:
    return char * width


def _print_level_report(n_users: int, stats: dict) -> None:
    mean_ok = stats["mean_s"] < NFR_MEAN_TARGET_S
    rate_ok = stats["success_rate"] >= NFR_SUCCESS_RATE

    print(f"\n  {'─'*55}")
    print(f"  Concurrent users : {n_users}")
    print(f"  Total requests   : {stats['n']}")
    print(f"  {'─'*55}")
    print(f"  Mean             : {stats['mean_s']:.2f}s  "
          f"{'✓ < 5s NFR' if mean_ok else '✗ EXCEEDS 5s NFR'}")
    print(f"  Median           : {stats['median_s']:.2f}s")
    print(f"  P95              : {stats['p95_s']:.2f}s")
    print(f"  P99              : {stats['p99_s']:.2f}s")
    print(f"  Min              : {stats['min_s']:.2f}s")
    print(f"  Max              : {stats['max_s']:.2f}s")
    print(f"  Std dev          : {stats['stdev_s']:.2f}s")
    print(f"  Success rate     : {stats['success_rate']*100:.1f}%  "
          f"{'✓' if rate_ok else '✗'}")
    print(f"  Status codes     : {stats['status_codes']}")


async def _check_health() -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(HEALTH_ENDPOINT, timeout=5)
            return r.status_code == 200
    except Exception:
        return False


async def main() -> None:
    print(_divider("═"))
    print("  MediAssist Backend — NFR Benchmark")
    print(f"  Target: mean < {NFR_MEAN_TARGET_S}s under {max(CONCURRENT_USERS_LEVELS)} concurrent users")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print(_divider("═"))

    print("\nChecking API health...")
    if not await _check_health():
        print("  FAIL: API not reachable. Start with:")
        print("    uvicorn src.api.main:app --reload --workers 2")
        sys.exit(1)
    print("  API is healthy ✓")

    all_level_stats = []
    queries = TEST_QUERIES[:REQUESTS_PER_LEVEL]

    for n_users in CONCURRENT_USERS_LEVELS:
        print(f"\nRunning {n_users} concurrent user(s) × {len(queries)} requests...")
        t_level = time.perf_counter()
        results = await _run_concurrent_level(n_users, queries)
        level_elapsed = time.perf_counter() - t_level

        stats = _compute_stats(results)
        stats["n_users"] = n_users
        stats["wall_clock_s"] = level_elapsed
        all_level_stats.append(stats)

        _print_level_report(n_users, stats)

    print(f"\n{_divider('═')}")
    print("  NFR VERDICT")
    print(_divider("═"))

    final = all_level_stats[-1]
    nfr_mean_pass = final["mean_s"] < NFR_MEAN_TARGET_S
    nfr_rate_pass = final["success_rate"] >= NFR_SUCCESS_RATE

    print(f"\n  At {final['n_users']} concurrent users:")
    print(f"    Mean {final['mean_s']:.2f}s < {NFR_MEAN_TARGET_S}s : "
          f"{'PASS ✓' if nfr_mean_pass else 'FAIL ✗'}")
    print(f"    Success rate {final['success_rate']*100:.1f}%     : "
          f"{'PASS ✓' if nfr_rate_pass else 'FAIL ✗'}")
    print(f"    P95 latency  {final['p95_s']:.2f}s            : reported")
    print(f"    P99 latency  {final['p99_s']:.2f}s            : reported")

    overall_pass = nfr_mean_pass and nfr_rate_pass
    print(f"\n  Overall: {'ALL NFR TARGETS MET ✓' if overall_pass else 'NFR TARGETS NOT MET ✗'}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nfr_mean_target": NFR_MEAN_TARGET_S,
        "levels": all_level_stats,
        "nfr_pass": overall_pass,
    }
    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {report_path}")
    print(_divider("═"))

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
