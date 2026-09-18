from typing import Any

import httpx

from app.llm.types import LLMResponse, ProviderError


def _strip_for_gemini(schema: dict[str, Any], is_properties: bool = False) -> dict[str, Any]:
    """Gemini's responseSchema rejects a few JSON-Schema keywords Pydantic emits.

    Keys of a `properties` map are field names, not keywords (a field called `title` must
    survive), so they are never dropped."""
    drop = {"title", "default", "additionalProperties"}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if not is_properties and k in drop:
            continue
        if isinstance(v, dict):
            out[k] = _strip_for_gemini(v, is_properties=(k == "properties" and not is_properties))
        elif isinstance(v, list):
            out[k] = [_strip_for_gemini(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


class GeminiClient:
    def __init__(self, api_key: str, base_url: str, http: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=60)

    async def generate(self, model: str, prompt: str, json_schema: dict[str, Any]) -> LLMResponse:
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _strip_for_gemini(json_schema),
            },
        }
        try:
            resp = await self._http.post(
                f"{self._base}/models/{model}:generateContent",
                headers={"x-goog-api-key": self._key},
                json=body,
            )
        except httpx.HTTPError as e:
            raise ProviderError(503, str(e)) from e
        if resp.status_code != 200:
            ra = resp.headers.get("retry-after")
            raise ProviderError(resp.status_code, resp.text[:200], float(ra) if ra else None)
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return LLMResponse(text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))
