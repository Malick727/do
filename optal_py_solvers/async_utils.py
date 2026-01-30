# This is a utility module, to use optal solver without changing the d-o API.
import asyncio
import threading
from typing import TypeVar, Awaitable

T = TypeVar("T")


def run_async_synchronously(coroutine: Awaitable[T]) -> T:
    """
    Runs a coroutine synchronously, blocking until it completes.

    This helper handles two cases:
    1. No event loop is running (standard script): Uses asyncio.run()
    2. An event loop is running (e.g., Jupyter): Runs the coroutine in a
       separate thread to avoid blocking the existing loop/raising runtime errors.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Case 2: Event loop is running. Run in a separate thread.
        result = None
        exception = None

        def runner():
            nonlocal result, exception
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coroutine)
            except Exception as e:
                exception = e
            finally:
                new_loop.close()

        t = threading.Thread(target=runner)
        t.start()
        t.join()

        if exception:
            raise exception
        return result
    else:
        # Case 1: No loop running. Standard run.
        return asyncio.run(coroutine)