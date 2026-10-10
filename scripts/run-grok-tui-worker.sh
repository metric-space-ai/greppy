#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h}
task_file="$repo_root/docs/GROK_TUI_PARITY_TASK.md"

if [[ ! -f "$task_file" ]]; then
  print -u2 "missing task file: $task_file"
  exit 2
fi

exec grok \
  --cwd "$repo_root" \
  --model grok-4.6 \
  --reasoning-effort high \
  --fullscreen \
  --always-approve \
  --max-turns 100 \
  --prompt-file "$task_file"
