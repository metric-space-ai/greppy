"""Provider adapters return visible annotation text and numeric usage only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from contracts import canonical, prompt_for, response_schema


class ProviderFailure(Exception):
    def __init__(self, kind, retry_after=None):
        super().__init__(kind)
        self.kind, self.retry_after = kind, retry_after


def safe_usage(value):
    if not isinstance(value, dict):
        return {}
    return {k:v for k,v in value.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}


def visible_response_text(value):
    """Never consume reasoning/thinking items, summaries or encrypted content."""
    if isinstance(value, dict) and isinstance(value.get('output_text'), str):
        return value['output_text']
    parts = []
    for item in value.get('output', []) if isinstance(value,dict) else []:
        if item.get('type') != 'message' or item.get('role') != 'assistant':
            continue
        for block in item.get('content', []):
            if block.get('type') == 'output_text' and isinstance(block.get('text'),str):
                parts.append(block['text'])
    if not parts:
        raise ProviderFailure('schema')
    return '\n'.join(parts)


def minimax(job):
    key = os.environ.get('MINIMAX_API_KEY')
    if not key:
        raise ProviderFailure('auth')
    body = {'model':job['model'],'input':[{'role':'user','content':prompt_for(job['examples'])}],
            'max_output_tokens':16384, 'store':False}
    request = urllib.request.Request('https://api.minimax.io/v1/responses', data=canonical(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Idempotency-Key':job['id']})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read(8*1024*1024+1)
        if len(raw)>8*1024*1024:
            raise ProviderFailure('schema')
        value = json.loads(raw)
    except urllib.error.HTTPError as error:
        if error.code in (401,403):
            raise ProviderFailure('auth') from None
        if error.code == 429:
            retry = error.headers.get('Retry-After','1800')
            try: delay = float(retry)
            except ValueError: delay = 1800
            raise ProviderFailure('quota',delay) from None
        raise ProviderFailure('transient' if error.code >= 500 else 'permanent') from None
    except (urllib.error.URLError,TimeoutError):
        # Delivery/completion is unknown; do not silently replay a possibly completed request.
        raise ProviderFailure('uncertain') from None
    if value.get('status') in ('incomplete','failed') or value.get('error'):
        code = str((value.get('error') or {}).get('code','')).lower()
        if 'quota' in code or 'rate' in code:
            raise ProviderFailure('quota')
        raise ProviderFailure('schema')
    try:
        annotations = json.loads(visible_response_text(value))
    except (ValueError,TypeError):
        raise ProviderFailure('schema') from None
    return annotations, safe_usage(value.get('usage'))


def grok(job, *, executable=None, workdir=None):
    executable = executable or os.environ.get('GROK_BIN', str(Path.home()/'.grok/bin/grok'))
    if sys.platform == 'darwin':
        if not Path('/Volumes/tmp').is_mount():
            raise ProviderFailure('permanent')
        default_scratch = '/Volumes/tmp/dev-artifacts/greppy/embedding-heads-optimization/grok'
    else:
        default_scratch = '/mnt/nvme1/greppy-heads-optimization-20260905/scratch/grok'
    workdir = Path(workdir or os.environ.get('HEADS_SCRATCH_ROOT',default_scratch))
    workdir.mkdir(parents=True,exist_ok=True)
    # Isolated cwd prevents repository instructions, side inputs and previous sessions
    # from contaminating a blind annotation. No tools or subagents are enabled.
    with tempfile.TemporaryDirectory(prefix='grok-head-audit-',dir=workdir) as temp:
        prompt = Path(temp)/'prompt.txt'; prompt.write_text(prompt_for(job['examples']),encoding='utf-8')
        argv = [executable,'--model',job['model'],'--prompt-file',str(prompt),
                '--json-schema',canonical(response_schema(job['examples'])),
                '--output-format','json','--max-turns','1','--no-subagents',
                '--disable-web-search','--tools','','--verbatim','--cwd',temp]
        try:
            result = subprocess.run(argv,cwd=temp,capture_output=True,text=True,timeout=360)
        except subprocess.TimeoutExpired:
            raise ProviderFailure('uncertain') from None
        except FileNotFoundError:
            raise ProviderFailure('auth') from None
        if result.returncode:
            diagnostic = (result.stderr+'\n'+result.stdout).lower()
            if any(word in diagnostic for word in ('quota','rate limit','429','usage limit')):
                raise ProviderFailure('quota')
            if any(word in diagnostic for word in ('login','logged out','authentication','unauthorized')):
                raise ProviderFailure('auth')
            raise ProviderFailure('transient')
        if len(result.stdout)>8*1024*1024:
            raise ProviderFailure('schema')
        try: value = json.loads(result.stdout)
        except ValueError: raise ProviderFailure('schema') from None
        if isinstance(value,dict) and set(value)=={'annotations'}:
            return value, {}
        # Known structured result containers only. Never search arbitrary trace fields.
        for name in ('structuredOutput','structured_output','result','text'):
            content = value.get(name) if isinstance(value,dict) else None
            if isinstance(content,str):
                try: content = json.loads(content)
                except ValueError: continue
            if isinstance(content,dict) and set(content)=={'annotations'}:
                return content, safe_usage(value.get('usage'))
        raise ProviderFailure('schema')
