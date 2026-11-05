"""Tiny Ollama client (standard library only)."""
import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, url='http://localhost:11434', model='gemma2:2b', timeout=60.0):
        self.url = url.rstrip('/')
        self.model = model
        self.timeout = timeout

    def available(self, timeout=2.0):
        """True when the Ollama server answers and has the model pulled."""
        try:
            with urllib.request.urlopen(self.url + '/api/tags', timeout=timeout) as r:
                tags = json.load(r)
        except (urllib.error.URLError, OSError, ValueError):
            return False
        names = {m.get('name', '') for m in tags.get('models', [])}
        base = self.model.split(':')[0]
        return self.model in names or any(n.split(':')[0] == base for n in names)

    def generate(self, prompt):
        """Raw text reply, or raises RuntimeError with a readable reason."""
        body = json.dumps({'model': self.model, 'prompt': prompt, 'stream': False,
                           'format': 'json', 'options': {'temperature': 0}}).encode()
        req = urllib.request.Request(self.url + '/api/generate', data=body,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r).get('response', '')
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'Ollama HTTP {e.code}: {e.read()[:200]!r}') from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f'Ollama not reachable at {self.url} ({e})') from e
        except ValueError as e:
            raise RuntimeError(f'Ollama sent invalid JSON ({e})') from e
