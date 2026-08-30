# from __future__ import annotations
# import os
# from pathlib import Path
# from typing import Optional


# def load_env_file(env_path: str | Path = '.env') -> None:
#     env_path = Path(env_path)
#     if not env_path.exists():
#         return
#     for line in env_path.read_text(encoding='utf-8', errors='ignore').splitlines():
#         line = line.strip()
#         if not line or line.startswith('#') or '=' not in line:
#             continue
#         key, value = line.split('=', 1)
#         key = key.strip()
#         value = value.strip().strip('"').strip("'")
#         if key and key not in os.environ:
#             os.environ[key] = value


# class GroqClient:
#     """Groq client using OpenAI-compatible Responses API."""

#     def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None, env_path: str | Path = '.env'):
#         load_env_file(env_path)
#         self.api_key = api_key or os.getenv('GROQ_API_KEY')
#         self.base_url = base_url or os.getenv('GROQ_BASE_URL') or 'https://api.groq.com/openai/v1'
#         self.model = model or os.getenv('GROQ_MODEL') or 'openai/gpt-oss-120b'
#         if not self.api_key:
#             raise ValueError('GROQ_API_KEY not found. Add it to your .env file.')

#     def get_openai_client(self):
#         try:
#             from openai import OpenAI
#         except ImportError as exc:
#             raise ImportError('OpenAI SDK missing. Install it with: pip install openai') from exc
#         return OpenAI(api_key=self.api_key, base_url=self.base_url)

#     @staticmethod
#     def extract_output_text(response) -> str:
#         if hasattr(response, 'output_text') and response.output_text:
#             return str(response.output_text).strip()
#         try:
#             output = getattr(response, 'output', None)
#             if output:
#                 chunks = []
#                 for item in output:
#                     content = getattr(item, 'content', None)
#                     if content:
#                         for c in content:
#                             text = getattr(c, 'text', None)
#                             if text:
#                                 chunks.append(str(text))
#                 if chunks:
#                     return '\n'.join(chunks).strip()
#         except Exception:
#             pass
#         return str(response)

#     def respond(self, input_text: str, temperature: Optional[float] = 0.2, max_output_tokens: Optional[int] = 900) -> str:
#         client = self.get_openai_client()
#         request = {'model': self.model, 'input': input_text}
#         if temperature is not None:
#             request['temperature'] = temperature
#         if max_output_tokens is not None:
#             request['max_output_tokens'] = max_output_tokens
#         try:
#             response = client.responses.create(**request)
#         except TypeError:
#             response = client.responses.create(model=self.model, input=input_text)
#         except Exception as exc:
#             msg = str(exc).lower()
#             if 'max_output_tokens' in msg or 'temperature' in msg:
#                 response = client.responses.create(model=self.model, input=input_text)
#             else:
#                 raise
#         return self.extract_output_text(response)

#     def test_connection(self) -> dict:
#         reply = self.respond('Reply with exactly: GROQ_CONNECTION_OK', temperature=0, max_output_tokens=20)
#         return {'status': 'success' if 'GROQ_CONNECTION_OK' in reply else 'unexpected_reply', 'model': self.model, 'base_url': self.base_url, 'reply': reply}


from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: str | Path = ".env") -> None:
    env_path = Path(env_path)

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


class GroqClient:
    """
    Groq client using OpenAI-compatible Responses API.

    Required .env:
    GROQ_API_KEY=your_key
    GROQ_BASE_URL=https://api.groq.com/openai/v1
    GROQ_MODEL=openai/gpt-oss-120b
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        env_path: str | Path = ".env",
    ):
        load_env_file(env_path)

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
        self.model = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    def get_openai_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI SDK missing. Install it with: pip install openai") from exc

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=90,
        )

    @staticmethod
    def extract_output_text(response) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text).strip()

        try:
            output = getattr(response, "output", None)
            if output:
                chunks = []

                for item in output:
                    content = getattr(item, "content", None)

                    if content:
                        for c in content:
                            text = getattr(c, "text", None)
                            if text:
                                chunks.append(str(text))

                if chunks:
                    return "\n".join(chunks).strip()
        except Exception:
            pass

        return str(response).strip()

    def respond(
        self,
        input_text: str,
        temperature: Optional[float] = 0.2,
        max_output_tokens: Optional[int] = 900,
    ) -> str:
        client = self.get_openai_client()

        request = {
            "model": self.model,
            "input": input_text,
        }

        if temperature is not None:
            request["temperature"] = temperature

        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        try:
            response = client.responses.create(**request)
        except TypeError:
            response = client.responses.create(
                model=self.model,
                input=input_text,
            )
        except Exception as exc:
            message = str(exc).lower()

            if "max_output_tokens" in message or "temperature" in message:
                response = client.responses.create(
                    model=self.model,
                    input=input_text,
                )
            else:
                raise

        return self.extract_output_text(response)

    def test_connection(self) -> dict:
        reply = self.respond(
            "Reply with exactly: GROQ_CONNECTION_OK",
            temperature=0,
            max_output_tokens=20,
        )

        return {
            "status": "success" if "GROQ_CONNECTION_OK" in reply else "unexpected_reply",
            "model": self.model,
            "base_url": self.base_url,
            "reply": reply,
        }