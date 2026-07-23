"""
LLM provider abstraction.

One interface, four backends. Every call returns (text, usage_dict) so that
token spend is tracked from turn one -- forecast vs. actual is how you find
out the budget is wrong in week 4 instead of week 9.

Parameters are fixed here deliberately (see execution plan S3):
  temperature = 0.0   -- variance should come from environment seeds, not
                         token sampling. Also the only setting that means the
                         same thing across every model family.
  max_tokens  = 512   -- an action plus brief reasoning fits easily.
"""
import os
import time

TEMPERATURE = 0.0
MAX_TOKENS = 512


class Provider:
    name = "base"

    def generate(self, system, messages):
        raise NotImplementedError


class MockProvider(Provider):
    """Deterministic stand-in. Lets you test the whole loop with no keys."""
    name = "mock"

    def __init__(self, model="mock", script=None):
        self.model = model
        self.script = script or ["Monitor"]
        self.i = 0

    def generate(self, system, messages):
        out = self.script[self.i % len(self.script)]
        self.i += 1
        return out, {"input_tokens": 100, "output_tokens": 5}


class OllamaProvider(Provider):
    """Local models, rungs M0-M4. No rate limits, just RAM and patience."""
    name = "ollama"

    def __init__(self, model):
        import ollama
        self.model = model
        self.client = ollama.Client()

    def generate(self, system, messages):
        msgs = [{"role": "system", "content": system}] + messages
        r = self.client.chat(
            model=self.model,
            messages=msgs,
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS, "seed": 0},
        )
        usage = {
            "input_tokens": r.get("prompt_eval_count", 0),
            "output_tokens": r.get("eval_count", 0),
        }
        return r["message"]["content"], usage


class OpenAICompatProvider(Provider):
    """Groq and Cerebras. Same API shape, different base_url."""
    name = "openai_compat"

    def __init__(self, model, base_url, api_key_env):
        from openai import OpenAI
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} not set")
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=key)

    def generate(self, system, messages):
        msgs = [{"role": "system", "content": system}] + messages
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs,
                    temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                )
                return r.choices[0].message.content, {
                    "input_tokens": r.usage.prompt_tokens,
                    "output_tokens": r.usage.completion_tokens,
                }
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(2 ** attempt)   # rate limits are the wall, not tokens
                    continue
                raise
        raise RuntimeError("rate limited after 5 attempts")


class AnthropicProvider(Provider):
    """Anchor A2 only. The one paid line item."""
    name = "anthropic"

    def __init__(self, model="claude-opus-4-8", effort="low"):
        import anthropic
        self.model = model
        self.effort = effort       # NEVER leave at default 'high' -- see plan S3
        self.client = anthropic.Anthropic()

    def generate(self, system, messages):
        r = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],  # menu is stable -> cache it
            messages=messages,
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens,
            "cache_read_tokens": getattr(r.usage, "cache_read_input_tokens", 0),
        }


def get_provider(spec):
    """
    spec examples:
      "mock"
      "ollama:qwen2.5:7b"
      "groq:llama-3.3-70b-versatile"
      "cerebras:qwen-3-32b"
      "anthropic:claude-opus-4-8"
    """
    if spec == "mock":
        return MockProvider()
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        return OllamaProvider(model)
    if kind == "groq":
        return OpenAICompatProvider(model, "https://api.groq.com/openai/v1", "GROQ_API_KEY")
    if kind == "cerebras":
        return OpenAICompatProvider(model, "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY")
    if kind == "anthropic":
        return AnthropicProvider(model)
    raise ValueError(f"unknown provider spec: {spec}")
