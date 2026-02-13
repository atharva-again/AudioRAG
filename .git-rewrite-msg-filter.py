#!/usr/bin/env python3
import re
import sys

message = sys.stdin.read()
lines = message.splitlines()
filtered = []
for line in lines:
    if re.match(r"^Ultraworked with \[Sisyphus\]", line):
        continue
    if re.match(r"^Co-authored-by:\s*Sisyphus", line):
        continue
    filtered.append(line)

sys.stdout.write("\n".join(filtered).rstrip() + "\n")
