"""Prospective tool onboarding, without task-specific selectors or solution scripts."""
import argparse
import json
from pathlib import Path
import re
import shlex

from dispatch import freeze_dispatch, task_goal

CONDITION = 'explicit_transport_v1'
CURRENT_BROWSER_CONDITION = 'browser_plugin_transport_v2'
CONDITIONS = (CONDITION, CURRENT_BROWSER_CONDITION)
COMMON = (
    'Work only through the visible browser UI and documented browser APIs. '
    'Do not read fixture source, host state files or application APIs. '
    'Discover the page yourself; use efficient batching when the next actions and targets are known. '
    'Verify the requested visible result. Do not write logs or benchmark reports. '
    'Finish with one short factual sentence.'
)


def participant_message(trial, condition=CONDITION):
    if condition not in CONDITIONS:
        raise ValueError('unsupported onboarding condition')
    goal = task_goal(trial)
    if trial.get('task_goal') != goal:
        raise ValueError('requires a prospectively recorded business goal')
    prefix = f"Complete this browser task on {trial['url']}\n\n{goal['text']}\n\n{COMMON}\n\n"
    if trial['arm'] == 'A' and condition == CURRENT_BROWSER_CONDITION:
        return prefix + (
            'Use only the standard Codex in-app Browser through mcp__node_repl__js. '
            'This study explicitly authorizes the standard browser despite repository defaults. '
            'Read and follow the installed Browser skill at '
            '/Users/michaelwelsch/.codex/plugins/cache/openai-bundled/browser/26.901.22334/skills/control-in-app-browser/SKILL.md. '
            'You may use the shell solely to read this skill, not for browser interaction. '
            'Initialize its browser-client runtime, select the in-app browser with agent.browsers.get("iab"), '
            'and emit its complete documentation before creating your own tab through iab.tabs.new(). '
            'Keep the browser in the background and retain its browser and tab bindings. '
            'Use the documented AX or Playwright APIs, including batching where useful. '
            'Do not use the former cua.createBrowserTab API, Greppy Web, application APIs, or another browser.'
        )
    if trial['arm'] == 'A':
        start = 'let tab = await cua.createBrowserTab("iab", ' + json.dumps(trial['url']) + ', {visible:false});'
        return prefix + (
            'Use only the standard Codex browser tool mcp__cua_repl in your own in-app browser tab (iab). '
            'This study explicitly authorizes standard CUA despite repository defaults. '
            'Create and retain the tab binding with this first call: ' + start + '\n'
            'Read the returned documentation and page state before subsequent calls. '
            'visible:false is required because visible tabs are unsupported in subagents. '
            'Use normal browser actions and the documented APIs; retain the tab binding for later calls. '
            'Do not use Greppy or shell/browser alternatives.'
        )
    if trial['arm'] != 'C':
        raise ValueError('this onboarding comparison contains only A and C; B needs its own runner')
    command = trial['cli_context']['command']
    if not re.fullmatch(r'gw-[a-z0-9-]+', command):
        raise ValueError('requires an isolated study command alias')
    start = json.dumps(command + ' web open ' + shlex.quote(trial['url']))
    return prefix + (
        f'Use only Greppy Web through the shell command {command}; it selects your isolated context and working directory. '
        f'The command shape is {command} web COMMAND ARGUMENTS. Keep those words separate; quote individual values such as URLs.\n'
        'Open the task page with this first functions.exec call: '
        'text(await tools.exec_command({cmd:' + start + '}));\n'
        'For subsequent commands, forward the complete exec_command result using the same text(await ...) form. '
        'When a result has a running session_id, poll that same session with write_stdin until terminal; '
        'an empty output chunk is not completion. Read the returned page state for the next decision. '
        'You may use --help. Use normal browser actions; do not use standard CUA, other browser tools, '
        'application APIs or direct state mutations.'
    )


def prepare_messages(series, name_prefix):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name_prefix):
        raise ValueError('use a lowercase task-name prefix')
    series = Path(series)
    plan = json.loads((series / 'plan.json').read_text())
    condition = plan.get('onboarding_condition')
    if condition not in CONDITIONS:
        raise ValueError('onboarding condition must be registered in the plan before dispatch')
    folder = series / 'prepared-dispatches'
    if folder.exists():
        raise FileExistsError('refusing to change or partially refill existing dispatches')
    messages = [(t, participant_message(t, condition)) for t in plan['trials']]
    return [freeze_dispatch(series, t['position'],
                            f"{name_prefix}_{t['case']}_{t['arm'].lower()}{t['repeat']}", message)
            for t, message in messages]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('series', type=Path)
    parser.add_argument('--name-prefix', required=True)
    args = parser.parse_args()
    records = prepare_messages(args.series, args.name_prefix)
    print(json.dumps({'state': 'prepared_not_sent', 'count': len(records),
                      'onboarding_condition': json.loads((args.series / 'plan.json').read_text())['onboarding_condition']}))
