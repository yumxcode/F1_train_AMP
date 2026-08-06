#!/usr/bin/env python3
"""Minimal test: print hello and exit. For debugging gm-run log visibility."""
import sys
print("HELLO_FROM_X1_TEST_SCRIPT", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Args: {sys.argv}", flush=True)

import os
print(f"CWD: {os.getcwd()}", flush=True)
print(f"Contents: {os.listdir('.')}", flush=True)

# Check if roboparty_train exists
if os.path.isdir('roboparty_train'):
    print(f"roboparty_train contents: {os.listdir('roboparty_train')}", flush=True)
else:
    print("roboparty_train NOT FOUND in CWD", flush=True)
    # Try parent
    parent = os.path.dirname(os.getcwd())
    if os.path.isdir(os.path.join(parent, 'roboparty_train')):
        print(f"Found in parent: {parent}", flush=True)

print("DONE", flush=True)
