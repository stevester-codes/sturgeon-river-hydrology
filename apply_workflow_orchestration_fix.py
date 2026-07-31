#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

# This script is intentionally text-based so it can safely patch existing workflow files.
WORKFLOWS = Path('.github/workflows')


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f'expected block not found in {path}: {old!r}')
    path.write_text(text.replace(old, new, 1))


def manual_only(path: Path) -> None:
    text = path.read_text()
    updated, count = re.subn(
        r"\Aname: ([^\n]+)\n\non:\n.*?\npermissions:\n",
        lambda match: f"name: {match.group(1)}\n\non:\n  workflow_dispatch:\n\npermissions:\n",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f'could not convert {path} to manual-only triggering')
    path.write_text(updated)


def main() -> None:
    operational = WORKFLOWS / 'sturgeon-operational.yml'
    replace_once(
        operational,
        "on:\n  workflow_dispatch:\n  schedule:\n    - cron: '17 * * * *'\n  push:\n",
        "on:\n  workflow_dispatch:\n    inputs:\n      trigger_reason:\n        description: 'Fresh upstream data or manual reason'\n        required: false\n        default: 'manual refresh'\n  push:\n",
    )
    replace_once(
        operational,
        "      - 'forecast_impacts_v2.py'\n      - 'requirements.txt'\n",
        "      - 'forecast_impacts_v2.py'\n      - 'operational_update_probe.py'\n      - 'requirements.txt'\n",
    )
    replace_once(
        operational,
        "concurrency:\n  group: sturgeon-hydrology-operational\n  cancel-in-progress: true\n",
        "concurrency:\n  group: sturgeon-hydrology-operational\n  cancel-in-progress: false\n",
    )

    shadow = WORKFLOWS / 'sturgeon-historical-response-shadow.yml'
    text = shadow.read_text()
    updated, count = re.subn(
        r"\Aname: Sturgeon historical response shadow\n\non:\n.*?\npermissions:\n",
        "name: Sturgeon historical response shadow\n\non:\n  workflow_dispatch:\n  workflow_run:\n    workflows: ['Sturgeon operational forecast']\n    types: [completed]\n\npermissions:\n",
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError('could not replace historical shadow trigger block')
    updated = updated.replace(
        "concurrency:\n  group: sturgeon-historical-response-shadow\n  cancel-in-progress: true\n",
        "concurrency:\n  group: sturgeon-historical-response-shadow\n  cancel-in-progress: false\n",
        1,
    )
    updated = updated.replace(
        "jobs:\n  compare:\n    runs-on: ubuntu-latest\n",
        "jobs:\n  compare:\n    if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success' }}\n    runs-on: ubuntu-latest\n",
        1,
    )
    shadow.write_text(updated)

    for filename in [
        'sturgeon-historical-event-backfill.yml',
        'sturgeon-historical-peak-reanalysis.yml',
        'sturgeon-historical-target-diagnostics.yml',
        'sturgeon-historical-censored-response.yml',
    ]:
        manual_only(WORKFLOWS / filename)

    print('Workflow orchestration migration completed.')


if __name__ == '__main__':
    main()
