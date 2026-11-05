# Draft issue for langchain-ai/langchain (not filed yet)

Search the existing issues for `ClientResponse.json` and `_generate_response_from_error` before filing,
and check the versions and line numbers against the latest release.

---

**Title:** `_generate_response_from_error` calls `response.json()` without await on async responses, causing "coroutine was never awaited"

### Summary

When a chat model fails, `_generate_response_from_error` in
`langchain_core/language_models/chat_models.py` reads the exception's `.response` and calls
`response.json()` synchronously. It assumes a synchronous response such as `httpx.Response`. If the
response's `json` is a coroutine function, as on `aiohttp.ClientResponse`, the call creates a coroutine
that is never awaited:

- Python emits `RuntimeWarning: coroutine '...json' was never awaited`, at whatever point garbage collection happens to run.
- `metadata["body"]` holds a coroutine object instead of the response body, so callbacks never get the body.
- The `try/except` around the call doesn't catch this, because nothing raises.

This affects `langchain-google-genai` in practice. The `google-genai` client uses aiohttp for async
calls when it's installed, and its `APIError.response` is then an `aiohttp.ClientResponse`. Every
`ainvoke` that gets an API error, such as a 503, produces the warning.

### Minimal reproduction

This uses only `langchain-core`, with no network or API key:

```python
import asyncio
import gc
import tracemalloc
import warnings
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


class AsyncResponse:
    """The shape of aiohttp.ClientResponse: json() is a coroutine function."""

    headers: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return {"error": "overloaded"}


class ProviderError(Exception):
    """A provider SDK error that exposes the raw response, as google.genai.errors.APIError does."""

    def __init__(self) -> None:
        super().__init__("503 UNAVAILABLE")
        self.response = AsyncResponse()


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        raise ProviderError()


tracemalloc.start(10)
warnings.simplefilter("always")


async def main() -> None:
    try:
        await FailingChatModel().ainvoke("hi")
    except ProviderError as error:
        print("caught:", error)


asyncio.run(main())
gc.collect()
```

### Actual output

The original `ProviderError` is raised correctly, and the warning traces back to the unawaited call
(paths shortened, asyncio frames removed):

```text
RuntimeWarning: coroutine 'AsyncResponse.json' was never awaited
Object allocated at (most recent call last):
  File "langchain_core/language_models/chat_models.py", lineno 552
    llm_result = await self.agenerate_prompt(
  File "langchain_core/language_models/chat_models.py", lineno 1931
    return await self.agenerate(
  File "langchain_core/language_models/chat_models.py", lineno 1867
    generations_with_error_metadata = _generate_response_from_error(res)
  File "langchain_core/language_models/chat_models.py", lineno 119
    metadata["body"] = response.json()
caught: 503 UNAVAILABLE
```

`ChatGoogleGenerativeAI.ainvoke`, receiving a real HTTP 503 from a local test server over aiohttp, shows
the same frames with `coroutine 'ClientResponse.json' was never awaited`.

### Expected behavior

No warning, and no coroutine object in `response_metadata`.

### Possible fix

`_generate_response_from_error` is synchronous, so it can't await the call. It could leave the body out
when `json` is asynchronous:

```python
if hasattr(response, "json") and not inspect.iscoroutinefunction(response.json):
    ...
```

`agenerate` could also read the body with `await` in a new async variant, if keeping the body on async
paths matters.

### Environment

- langchain-core 1.7.6
- langchain-google-genai 4.4.3
- google-genai 2.34.0
- aiohttp 3.14.5
- Python 3.12.9, Windows 11
