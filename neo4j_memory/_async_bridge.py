"""Asyncio event-loop bridge for sync callers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
from collections.abc import Coroutine
from typing import Any


class AsyncLoopThread:
    """Run an asyncio event loop in a daemon thread for sync→async bridging."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    def start(self) -> "AsyncLoopThread":
        """Start the loop thread if needed.

        The method is idempotent: if a live loop thread already exists, this is
        a no-op. If a prior shutdown completed, calling start creates a fresh
        loop thread.
        """
        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
                and self._loop is not None
            ):
                return self

            self._loop = None
            self._startup_error = None
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="neo4j-memory-async-loop",
                daemon=True,
            )
            self._thread.start()

        self._ready.wait()

        with self._lock:
            if self._startup_error is not None:
                raise self._startup_error
            if self._loop is None:
                raise RuntimeError("async loop thread did not initialize")

        return self

    def submit(self, coro: Coroutine[Any, Any, Any], timeout: float = 30.0) -> Any:
        """Run *coro* on the loop thread and block for its result.

        Exceptions raised by the coroutine propagate unchanged from
        ``Future.result()``. TimeoutError from the wait is also allowed to
        propagate, with the scheduled coroutine cancelled to avoid leaking work.
        """
        if not inspect.iscoroutine(coro):
            raise TypeError("submit() expects a coroutine object")

        self.start()
        with self._lock:
            loop = self._loop
            thread = self._thread

        if loop is None or thread is None or not thread.is_alive():
            coro.close()
            raise RuntimeError("async loop thread is not running")

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the loop thread and wait for it to exit.

        The method is idempotent and safe to call multiple times. Pending tasks
        are cancelled before the loop is stopped; the thread is joined with the
        provided timeout.
        """
        with self._lock:
            loop = self._loop
            thread = self._thread

        if loop is None or thread is None:
            return

        if thread is threading.current_thread():
            if loop.is_running():
                loop.call_soon(loop.stop)
            return

        if thread.is_alive() and loop.is_running():
            cleanup_timeout = max(timeout / 2.0, 0.1) if timeout is not None else None
            cleanup_future = asyncio.run_coroutine_threadsafe(
                self._cancel_pending(loop), loop
            )
            try:
                cleanup_future.result(cleanup_timeout)
            except concurrent.futures.TimeoutError:
                cleanup_future.cancel()
            except Exception:
                # Cleanup is best-effort; shutdown must still stop and join the thread.
                pass

            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout)
        elif thread.is_alive():
            thread.join(timeout)

        if not thread.is_alive():
            with self._lock:
                if self._thread is thread:
                    self._thread = None
                if self._loop is loop:
                    self._loop = None

    def _run_loop(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                self._loop = loop
            self._ready.set()
            loop.run_forever()
        except BaseException as exc:  # pragma: no cover - defensive startup path
            with self._lock:
                self._startup_error = exc
            self._ready.set()
        finally:
            loop = self._loop
            if loop is not None:
                try:
                    loop.run_until_complete(self._cancel_pending(loop))
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()
                    with self._lock:
                        if self._loop is loop:
                            self._loop = None

    async def _cancel_pending(self, loop: asyncio.AbstractEventLoop) -> None:
        current = asyncio.current_task(loop=loop)
        pending = [
            task
            for task in asyncio.all_tasks(loop)
            if task is not current and not task.done()
        ]
        if not pending:
            return

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
