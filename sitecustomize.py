from __future__ import annotations

import site
from pathlib import Path


LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"

if LOCAL_DEPS.exists():
    site.addsitedir(str(LOCAL_DEPS))
