"""Index a historical wall archive without exporting trace metadata or teacher text.

This inventory is diagnostic only: archive presence does not prove complete capture,
privacy admission, or independence from previous training.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from contracts import canonical, digest
from corpus import template_hash


def catalog(path, output):
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(output/'catalog.sqlite')
    counts = Counter(); lengths = Counter(); datasets = set()
    raw_hash = hashlib.sha256(); max_lines = 0; max_bytes = 0
    try:
        db.executescript('''
          CREATE TABLE sources (
            ordinal INTEGER PRIMARY KEY, archive_offset INTEGER NOT NULL,
            archive_bytes INTEGER NOT NULL, source_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL, lineage_id TEXT NOT NULL,
            wall_sha256 TEXT NOT NULL, template_sha256 TEXT NOT NULL,
            line_count INTEGER NOT NULL, byte_count INTEGER NOT NULL,
            exit_code INTEGER, capture_complete INTEGER,
            privacy_admitted INTEGER NOT NULL DEFAULT 0,
            previously_exposed INTEGER NOT NULL DEFAULT 1);
          CREATE INDEX source_hash ON sources(wall_sha256);
          CREATE INDEX template_hash ON sources(template_sha256);
        ''')
        with path.open('rb') as f:
            for ordinal, raw in enumerate(f, 1):
                offset = f.tell()-len(raw); raw_hash.update(raw)
                x = json.loads(raw)
                # Do not inspect or retain next_turn, model reasoning, headers, or
                # arbitrary metadata. Public-origin names alone are not admission.
                wall = x.get('wall'); source = x.get('source'); dataset = x.get('dataset')
                if not isinstance(wall,str) or not wall or not isinstance(source,str) or not isinstance(dataset,str):
                    counts['invalid_or_empty'] += 1
                    continue
                encoded = wall.encode('utf-8'); whash = hashlib.sha256(encoded).hexdigest()
                lines = len(wall.splitlines()); nbytes = len(encoded)
                exit_code = x.get('exit_code')
                if exit_code is not None and type(exit_code) is not int:
                    counts['invalid_exit_code'] += 1
                    exit_code = None
                dataset_id = digest(dataset); datasets.add(dataset_id)
                lineage_id = digest([dataset,source])
                sid = digest([dataset_id,lineage_id,whash])
                db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                           (ordinal,offset,len(raw),sid,dataset_id,lineage_id,whash,template_hash(wall),
                            lines,nbytes,exit_code,None,0,1))
                counts['records'] += 1
                lengths['1-63' if lines<64 else '64-999' if lines<1000 else '1000-9999' if lines<10000 else '10000-99999' if lines<100000 else '100000+'] += 1
                max_lines=max(max_lines,lines); max_bytes=max(max_bytes,nbytes)
        db.commit()
        unique = db.execute('SELECT count(DISTINCT wall_sha256),count(DISTINCT template_sha256),count(DISTINCT source_id) FROM sources').fetchone()
        report = {'schema':'greppy.heads.archive-inventory.v1', 'archive_sha256':raw_hash.hexdigest(),
                  'archive_bytes':path.stat().st_size, 'counts':dict(counts), 'line_buckets':dict(lengths),
                  'dataset_count':len(datasets), 'unique_contents':unique[0],
                  'unique_templates':unique[1], 'unique_source_ids':unique[2],
                  'maximum_lines':max_lines, 'maximum_bytes':max_bytes,
                  'capture_verified':0, 'privacy_admitted':0, 'eligible_final_sources':0,
                  'limitation':'Historical diagnostic archive; capture completeness, lineage and privacy still require review.'}
        with (output/'manifest.json').open('x') as f:
            f.write(canonical(report)+'\n')
        return report
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(canonical(catalog(args.archive,args.out)))


if __name__=='__main__': main()
