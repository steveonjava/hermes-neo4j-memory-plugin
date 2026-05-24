import asyncio
import concurrent.futures
import threading
import time

import pytest

from neo4j_memory._async_bridge import AsyncLoopThread


async def _thread_id_result(value):
    await asyncio.sleep(0)
    return value, threading.get_ident()


def test_submit_runs_coroutine_on_loop_thread_and_returns_result():
    bridge = AsyncLoopThread()
    try:
        result, loop_thread_id = bridge.submit(_thread_id_result("ok"))
        assert result == "ok"
        assert loop_thread_id != threading.get_ident()
    finally:
        bridge.shutdown()


def test_start_is_idempotent_and_keeps_same_thread():
    bridge = AsyncLoopThread()
    try:
        bridge.start()
        first_thread = bridge._thread
        bridge.start()
        assert bridge._thread is first_thread
        assert first_thread is not None
        assert first_thread.is_alive()
    finally:
        bridge.shutdown()


def test_submit_propagates_coroutine_exception_unchanged():
    class SentinelError(RuntimeError):
        pass

    error = SentinelError("boom")

    async def raises():
        raise error

    bridge = AsyncLoopThread()
    try:
        with pytest.raises(SentinelError) as exc_info:
            bridge.submit(raises())
        assert exc_info.value is error
    finally:
        bridge.shutdown()


def test_submit_rejects_non_coroutine():
    bridge = AsyncLoopThread()
    with pytest.raises(TypeError, match="coroutine"):
        bridge.submit("not-a-coroutine")
    bridge.shutdown()


def test_submit_timeout_cancels_future_and_raises_timeout():
    bridge = AsyncLoopThread()
    task_cancelled = threading.Event()

    async def waits_forever():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            task_cancelled.set()
            raise

    try:
        with pytest.raises(concurrent.futures.TimeoutError):
            bridge.submit(waits_forever(), timeout=0.001)

        deadline = time.monotonic() + 1.0
        while not task_cancelled.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert task_cancelled.is_set()
    finally:
        bridge.shutdown()


def test_shutdown_before_start_is_noop():
    bridge = AsyncLoopThread()
    bridge.shutdown()
    bridge.shutdown()


def test_shutdown_is_idempotent_and_joins_thread():
    bridge = AsyncLoopThread()
    bridge.start()
    thread = bridge._thread

    bridge.shutdown()
    bridge.shutdown()

    assert thread is not None
    assert not thread.is_alive()


def test_shutdown_cancels_pending_tasks_before_stopping_loop():
    bridge = AsyncLoopThread()
    task_cancelled = threading.Event()

    async def pending_task():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            task_cancelled.set()
            raise

    bridge.start()
    assert bridge._loop is not None
    asyncio.run_coroutine_threadsafe(pending_task(), bridge._loop)

    deadline = time.monotonic() + 1.0
    while not asyncio.all_tasks(bridge._loop) and time.monotonic() < deadline:
        time.sleep(0.01)

    bridge.shutdown()

    assert task_cancelled.is_set()


def test_submit_after_shutdown_restarts_bridge():
    bridge = AsyncLoopThread()
    bridge.start()
    first_thread = bridge._thread
    bridge.shutdown()

    try:
        result, second_thread_id = bridge.submit(_thread_id_result("again"))
        assert result == "again"
        assert first_thread is not bridge._thread
        assert bridge._thread is not None
        assert bridge._thread.ident == second_thread_id
    finally:
        bridge.shutdown()


def test_1000_concurrent_submits_are_thread_safe():
    bridge = AsyncLoopThread()

    async def identity(value):
        await asyncio.sleep(0)
        return value

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(bridge.submit, identity(i), 10.0) for i in range(1000)
            ]
            results = [future.result(timeout=10.0) for future in futures]
    finally:
        bridge.shutdown()

    assert sorted(results) == list(range(1000))
