"""Test setup shared by every level.

With LOVELY_TENSORS=1, which the VS Code debug configurations set, tensors
print as one line of shape and statistics, both in the debugger's variables
and in assertion messages. No test depends on it.
"""

import os

import lovely_tensors

if os.environ.get("LOVELY_TENSORS") == "1":
    lovely_tensors.monkey_patch()
