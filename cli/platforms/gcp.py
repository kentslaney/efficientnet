# Copyright (C) 2020 by Kent Slaney <kent@slaney.org>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

from cli.remote import platforms, actions
from cli.utils import strftime
import subprocess

def local(action):
    subprocess.run(["gcloud", "ai-platform", "local", "train", "--module-name",
                    "cli", "--package-path", "cli/", "--"] + action)

def local_cli(parser):
    parser.add_argument("action", **actions)
    parser.set_defaults(call=local)

def remote(action, job_name):
    formatted = job_name.format(time=strftime())
    if formatted != job_name:
        print(f"submitting {formatted}")

    assert "--" in action
    split = action.index("--") + 1
    action = action[1:split] + action[:1] + action[split:]
    subprocess.run(
        ["gcloud", "ai-platform", "jobs", "submit", "training", formatted,
        "--module-name", "cli", "--package-path", "cli/", "--runtime-version",
        "2.2", "--python-version", "3.7"] + action)

def cli(parser):
    parser.add_argument("--job-name", default="job_{time}")
    parser.add_argument("action", **actions)
    parser.set_defaults(call=remote)

platforms["gcp-local"] = local_cli
platforms["gcp"] = cli
