#!/usr/bin/env python3
"""Read-only local source-pin check. No compiler/runtime/payload/network action."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
root=here.parent
receipt=json.loads((here/'SOURCE_RECEIPT.json').read_text())
for relative,expected in receipt['files'].items():
    path=root/relative
    raw=path.read_bytes()
    assert len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],relative
print('PASS',len(receipt['files']),'source pins; no execution evidence')
