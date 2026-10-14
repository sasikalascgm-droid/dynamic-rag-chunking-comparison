
import json
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"


class OllamaClientError(Exception):
    """Raised when an Ollama request fails."""

    def __init__(self, message, status_code=502):
        super().__init__(message)
        self.status_code = status_code


def generate_answer(
    prompt,
    model=DEFAULT_MODEL,
    timeout=180
):
    """
    Send a prompt to Ollama and return its generated answer.
    """

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        answer = data.get("response", "").strip()

        if not answer:
            raise OllamaClientError(
                "Ollama returned an empty answer."
            )

        return answer

    except urllib.error.HTTPError as error:
        raise OllamaClientError(
            f"Ollama returned HTTP error {error.code}.",
            status_code=502
        ) from error

    except (urllib.error.URLError, TimeoutError) as error:
        raise OllamaClientError(
            f"Could not connect to Ollama: {error}",
            status_code=503
        ) from error

    except json.JSONDecodeError as error:
        raise OllamaClientError(
            "Ollama returned an invalid JSON response.",
            status_code=502
        ) from error
