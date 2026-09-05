"""Durable teacher job ledger with provider-wide pauses and fenced worker leases."""
import json
from contextlib import contextmanager
import sqlite3
import time
import uuid
from pathlib import Path

from contracts import RUBRIC_VERSION, canonical, digest, sanitized_example, validate_annotations, prompt_for, teacher_configuration, response_schema


class QueueStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL,
                  rubric TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0, retry_count INTEGER NOT NULL DEFAULT 0,
                  owner TEXT, lease_until REAL, available_at REAL NOT NULL DEFAULT 0,
                  created_at REAL NOT NULL, completed_at REAL, result TEXT, error_code TEXT);
                CREATE TABLE IF NOT EXISTS providers (
                  provider TEXT PRIMARY KEY, paused_until REAL NOT NULL, reason TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  sequence INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL,
                  job_id TEXT, event TEXT NOT NULL, detail TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sources (
                  source_id TEXT PRIMARY KEY, split TEXT NOT NULL, group_key TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS groups (
                  group_key TEXT PRIMARY KEY, split TEXT NOT NULL);
            ''')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}
            for column, default in [('configuration','{}'),('prompt',''),('output_schema','{}')]:
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT NOT NULL DEFAULT '{default}'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def event(db, job, event, detail, now):
        db.execute('INSERT INTO events(at,job_id,event,detail) VALUES(?,?,?,?)',
                   (now,job,event,canonical(detail)))

    def register_source(self, source_id, split, group_key):
        if split not in ('train','development','final','diagnostic'):
            raise ValueError('invalid split')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT split,group_key FROM sources WHERE source_id=?',(source_id,)).fetchone()
            group = db.execute('SELECT split FROM groups WHERE group_key=?',(group_key,)).fetchone()
            if old and (old['split'] != split or old['group_key'] != group_key):
                raise ValueError('source split/group is immutable')
            if group and group['split'] != split:
                raise ValueError('related source group crosses splits')
            db.execute('INSERT OR IGNORE INTO groups VALUES(?,?)',(group_key,split))
            db.execute('INSERT OR IGNORE INTO sources VALUES(?,?,?)',(source_id,split,group_key))

    def enqueue(self, provider, model, examples, rubric=RUBRIC_VERSION, now=None):
        if not examples:
            raise ValueError('empty batch')
        now = time.time() if now is None else now
        payload = [sanitized_example(x) for x in examples]
        # Version binds prompt as well as label rules; changing either creates new jobs.
        configuration = teacher_configuration(provider)
        prompt = prompt_for(payload); schema = response_schema(payload)
        key = digest({'provider':provider,'model':model,'rubric':rubric,
                      'prompt_sha256':digest(prompt), 'configuration':configuration})
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for x in examples:
                source = db.execute('SELECT split FROM sources WHERE source_id=?',(x['source_id'],)).fetchone()
                if not source or source['split'] != x['split']:
                    raise ValueError('source must be registered with matching split before enqueue')
                if x.get('privacy_review') not in ('public-redacted','synthetic'):
                    raise ValueError('external annotation requires explicit privacy admission')
            db.execute('INSERT OR IGNORE INTO jobs(id,provider,model,rubric,payload,status,created_at,configuration,prompt,output_schema) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (key,provider,model,rubric,canonical(payload),'queued',now,canonical(configuration),prompt,canonical(schema)))
            # Exact cache-key equality proves this reconstructed prompt/configuration
            # matches an older job. Preserve its result and never dispatch it again.
            old = db.execute('SELECT prompt,output_schema FROM jobs WHERE id=?',(key,)).fetchone()
            if old['prompt'] == '':
                db.execute('UPDATE jobs SET prompt=?,output_schema=? WHERE id=?',(prompt,canonical(schema),key))
                self.event(db,key,'binding_recovered_from_exact_cache_key',{},now)
            elif old['prompt'] != prompt or old['output_schema'] != canonical(schema):
                raise ValueError('stored job binding does not match cache key')
        return key

    def claim(self, provider, *, lease_seconds=900, now=None):
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            # A dead worker may have sent a billable request. Never silently replay it.
            expired = db.execute("SELECT id FROM jobs WHERE status='running' AND lease_until<=?",(now,)).fetchall()
            for row in expired:
                db.execute("UPDATE jobs SET status='uncertain',error_code='worker_lease_expired',owner=NULL WHERE id=?",(row['id'],))
                self.event(db,row['id'],'uncertain',{'reason':'worker_lease_expired'},now)
            pause = db.execute('SELECT paused_until FROM providers WHERE provider=?',(provider,)).fetchone()
            if pause and pause['paused_until'] > now:
                return None
            row = db.execute("SELECT * FROM jobs WHERE provider=? AND status='queued' AND available_at<=? ORDER BY created_at,id LIMIT 1",(provider,now)).fetchone()
            if row is None:
                return None
            owner = uuid.uuid4().hex
            db.execute("UPDATE jobs SET status='running',attempts=attempts+1,owner=?,lease_until=? WHERE id=?",(owner,now+lease_seconds,row['id']))
            self.event(db,row['id'],'started',{'attempt':row['attempts']+1},now)
            job = dict(row); job.update(owner=owner,attempts=row['attempts']+1)
            job['examples'] = json.loads(job.pop('payload'))
            job['configuration'] = json.loads(job['configuration'])
            job['output_schema'] = json.loads(job['output_schema'])
            return job

    def complete(self, job, result, usage=None, now=None):
        now = time.time() if now is None else now
        validate_annotations(result,job['examples'])
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status,owner FROM jobs WHERE id=?',(job['id'],)).fetchone()
            if not row or row['status'] != 'running' or row['owner'] != job['owner']:
                raise ValueError('stale worker completion refused')
            db.execute("UPDATE jobs SET status='done',result=?,completed_at=?,owner=NULL,lease_until=NULL,error_code=NULL WHERE id=?",
                       (canonical(result),now,job['id']))
            self.event(db,job['id'],'completed',{'result_sha256':digest(result),'usage':usage or {}},now)

    def fail(self, job, kind, *, retry_after=None, now=None, diagnostic=None, usage=None):
        if kind not in ('quota','auth','transient','schema','permanent','uncertain'):
            raise ValueError('unknown failure kind')
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone()
            if not row or row['status'] != 'running' or row['owner'] != job['owner']:
                raise ValueError('stale worker failure refused')
            retries = row['retry_count']
            available = now
            if kind in ('quota','auth'):
                delay = 1800 if retry_after is None else max(1,min(float(retry_after),86400))
                until = now+delay if kind == 'quota' else 253402300799.0
                db.execute('INSERT INTO providers VALUES(?,?,?) ON CONFLICT(provider) DO UPDATE SET paused_until=max(providers.paused_until,excluded.paused_until),reason=excluded.reason',
                           (job['provider'],until,kind))
                status = 'queued'; available = until
            elif kind == 'transient' and retries < 3:
                retries += 1; status = 'queued'; available = now+min(60,2**retries)
            else:
                status = 'uncertain' if kind == 'uncertain' else 'failed'
            db.execute('UPDATE jobs SET status=?,retry_count=?,available_at=?,owner=NULL,lease_until=NULL,error_code=? WHERE id=?',
                       (status,retries,available,kind,job['id']))
            self.event(db,job['id'],kind,{'retry_count':retries,'status':status,'available_at':available,
                       'diagnostic':diagnostic,'usage':usage or {}},now)

    def resume_provider(self, provider):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM providers WHERE provider=?',(provider,))
            db.execute("UPDATE jobs SET available_at=0 WHERE provider=? AND status='queued' AND error_code IN ('auth','quota')",(provider,))
            self.event(db,None,'provider_resumed',{'provider':provider},time.time())

    def status(self):
        with self.connect() as db:
            counts = [dict(r) for r in db.execute('SELECT provider,status,count(*) AS count FROM jobs GROUP BY provider,status')]
            pauses = [dict(r) for r in db.execute('SELECT * FROM providers')]
            return {'jobs':counts,'provider_pauses':pauses}
