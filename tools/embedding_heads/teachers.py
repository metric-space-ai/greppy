"""Provider adapters return visible annotation text and numeric usage only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from contracts import canonical, prompt_for, response_schema, strict_json, teacher_configuration


class ProviderFailure(Exception):
    def __init__(self, kind, retry_after=None, *, diagnostic=None, usage=None):
        super().__init__(kind)
        self.kind, self.retry_after = kind, retry_after
        self.diagnostic, self.usage = diagnostic, usage or {}


def bound_prompt(job):
    """Dispatch exactly the stored request even if preparation code changed."""
    from contracts import digest
    prompt = job.get('prompt')
    schema = job.get('output_schema')
    if not isinstance(prompt,str) or not prompt or not isinstance(schema,dict):
        raise ProviderFailure('permanent',diagnostic='job_binding_missing')
    expected = digest({'provider':job['provider'],'model':job['model'],'rubric':job['rubric'],
                       'prompt_sha256':digest(prompt),'configuration':job['configuration']})
    marker = '\nOUTPUT_SCHEMA\n'+canonical(schema)+'\nUNTRUSTED_EXAMPLES_JSON\n'
    if expected != job['id'] or marker not in prompt:
        raise ProviderFailure('permanent',diagnostic='job_binding_invalid')
    return prompt


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
    configuration = job.get('configuration')
    if configuration != teacher_configuration('minimax'):
        raise ProviderFailure('permanent')
    key = os.environ.get('MINIMAX_API_KEY')
    if not key:
        raise ProviderFailure('auth')
    body = {'model':job['model'],'input':[{'role':'user','content':bound_prompt(job)}],
            'max_output_tokens':configuration['max_output_tokens'], 'store':configuration['store']}
    request = urllib.request.Request(configuration['endpoint'], data=canonical(body).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Idempotency-Key':job['id']})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read(8*1024*1024+1)
        if len(raw)>8*1024*1024:
            raise ProviderFailure('schema')
        value = strict_json(raw)
        if not isinstance(value,dict):
            raise ProviderFailure('schema',diagnostic='provider_envelope_not_object')
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
        failure = value.get('error')
        code = str(failure.get('code','') if isinstance(failure,dict) else failure or '').lower()
        if 'quota' in code or 'rate' in code:
            raise ProviderFailure('quota')
        raise ProviderFailure('schema',diagnostic='provider_incomplete_or_failed',usage=safe_usage(value.get('usage')))
    try:
        annotations = strict_json(visible_response_text(value))
    except (ValueError,TypeError):
        raise ProviderFailure('schema',diagnostic='visible_annotation_json_invalid',usage=safe_usage(value.get('usage'))) from None
    return annotations, safe_usage(value.get('usage'))


def grok(job, *, executable=None, workdir=None):
    if job.get('configuration') != teacher_configuration('grok'):
        raise ProviderFailure('permanent')
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
        prompt = Path(temp)/'prompt.txt'; prompt.write_text(bound_prompt(job),encoding='utf-8')
        argv = [executable,'--model',job['model'],'--prompt-file',str(prompt),
                '--json-schema',canonical(job['output_schema']),
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
        try: value = strict_json(result.stdout)
        except ValueError: raise ProviderFailure('schema') from None
        if isinstance(value,dict) and set(value)=={'annotations'}:
            return value, {}
        # Known structured result containers only. Never search arbitrary trace fields.
        for name in ('structuredOutput','structured_output','result','text'):
            content = value.get(name) if isinstance(value,dict) else None
            if isinstance(content,str):
                try: content = strict_json(content)
                except ValueError: continue
            if isinstance(content,dict) and set(content)=={'annotations'}:
                return content, safe_usage(value.get('usage'))
        raise ProviderFailure('schema')
