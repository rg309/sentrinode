from __future__ import annotations

import asyncio
import os
import random
import time


async def simulate_io_latency(min_ms: int = 5, max_ms: int = 50) -> None:
    await asyncio.sleep(random.uniform(min_ms / 1000, max_ms / 1000))


def artificial_cpu_work(duration_ms: int) -> None:
    start = time.perf_counter()
    duration = duration_ms / 1000.0
    while time.perf_counter() - start < duration:
        sum(i * i for i in range(100))


def cpu_saturation_budget() -> int:
    return int(os.getenv("CPU_WORK_MS", "5"))
