#!/usr/bin/env python3
"""Source/hash-only; no coordinator import, compilation or runtime."""
from pathlib import Path
import hashlib
import json
HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block=f.read(65536)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
def main():
    with (HERE/'SOURCE_RECEIPT.json').open('rb') as f:
        text=f.read((1<<20)+1)
    if len(text)>1<<20:
        raise ValueError('receipt cap')
    receipt=json.loads(text)
    for name,expected in receipt['files'].items():
        if sha(HERE/name)!=expected:
            raise ValueError('source changed: '+name)
    for name,expected in receipt['input_pins'].items():
        if sha(HERE.parent/name)!=expected:
            raise ValueError('reviewed source input changed: '+name)
    print('PASS source-only pins:',len(receipt['files']),'package,',len(receipt['input_pins']),'review inputs')
if __name__=='__main__':
    main()
