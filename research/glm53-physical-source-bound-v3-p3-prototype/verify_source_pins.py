#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
a=json.loads((p/'SOURCE_RECEIPT.json').read_text())
for rel,digest in a['files_sha256'].items():
 assert hashlib.sha256((p/rel).read_bytes()).hexdigest()==digest,rel
for rel,digest in a['dependencies_sha256'].items():
 assert hashlib.sha256((p.parent/rel).read_bytes()).hexdigest()==digest,rel
print('PASS complete synthetic physical-source bridge frozen source pins')
