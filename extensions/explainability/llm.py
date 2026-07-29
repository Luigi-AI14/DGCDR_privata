"""Minimal Ollama client for the semantic audit.

`verbalize.py` already speaks the OpenAI-compatible dialect, but the audit needs
one thing that dialect does not expose: control over Ollama's thinking mode.
Qwen3.5 reasons before answering by default, which is wasted effort when the
answer is a six-word label and useful when the answer is a judgement. Being able
to switch it per phase matters enough to talk to Ollama's own endpoint.
"""

import json
import re
import urllib.error
import urllib.request

THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


class OllamaClient:
    """Talks to a local Ollama server, with the thinking mode under control."""

    def __init__(self, model, host='http://localhost:11434', think=False,
                 temperature=0.2, timeout=300):
        self.model = model
        self.host = host.rstrip('/')
        self.think = think
        self.temperature = temperature
        self.timeout = timeout
        self.calls = 0

    def ask(self, prompt, system=None, max_tokens=300):
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': False,
            'think': self.think,
            'options': {'temperature': self.temperature, 'num_predict': max_tokens},
        }
        request = urllib.request.Request(
            f'{self.host}/api/chat',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST')

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode('utf-8'))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ollama request to {self.host} failed: {exc}. Is the server "
                f"running, and is '{self.model}' pulled?") from exc

        self.calls += 1
        content = (body.get('message') or {}).get('content', '')
        # Some builds return the reasoning inline even with think disabled.
        return THINK_BLOCK.sub('', content).strip()

    def rate(self, prompt, system=None, lo=0, hi=10):
        """Ask for a number in [lo, hi]. Returns None if the answer is unusable.

        A graded answer carries far more than a forced choice: the same number
        of calls buys a distribution instead of a count of hits, which is what
        makes a paired test possible on a few dozen concepts.
        """
        answer = self.ask(prompt, system=system, max_tokens=2000 if self.think else 60)
        match = re.search(r'-?\d+(?:[.,]\d+)?', answer.replace(',', '.'))
        if not match:
            return None
        try:
            value = float(match.group(0))
        except ValueError:
            return None
        return value if lo <= value <= hi else None

    def choose(self, prompt, options, system=None):
        """Ask for one option out of several, and return its index.

        Returns None when the answer cannot be read as a choice, which is kept
        distinct from a wrong choice: a model that fails to answer the format is
        not the same as a model that answers incorrectly.
        """
        letters = [chr(ord('A') + i) for i in range(len(options))]
        answer = self.ask(prompt, system=system, max_tokens=2000 if self.think else 200)

        # The letter on its own, or the first letter-like token in the answer.
        cleaned = answer.strip().upper()
        for letter in letters:
            if cleaned == letter or cleaned.startswith(letter + ')') \
                    or cleaned.startswith(letter + '.') or cleaned.startswith(letter + ' '):
                return letters.index(letter)
        match = re.search(r'\b([A-%s])\b' % letters[-1], cleaned)
        if match:
            return letters.index(match.group(1))
        return None
