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
