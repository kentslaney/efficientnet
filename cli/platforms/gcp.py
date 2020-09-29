from cli.remote import platforms, remote_action
from cli.utils import strftime
import subprocess

def local(action, argv):
    subprocess.run(["gcloud", "ai-platform", "local", "train", "--module-name",
                    "cli", "--package-path", "cli/", "--", action] + argv)

def local_cli(parser):
    parser.add_argument("action", action=remote_action)
    parser.set_defaults(call=local)

def remote(action, job_name, argv):
    formatted = job_name.format(time=strftime)
    if formatted != job_name:
        print(f"submitting job {formatted}")

    assert "--" in argv
    assert "--epochs" in argv
    split = argv.index("--")
    subprocess.run(
        ["gcloud", "ai-platform", "jobs", "submit", "training", formatted,
        "--module-name", "cli", "--package-path", "cli/", "--runtime-version",
        "2.2", "--python-version", "3.7"] + argv[:split] + ["--", action] +
        argv[split + 1:])

def cli(parser):
    parser.add_argument("action", action=remote_action)
    parser.add_argument("job-name", nargs="?", default="{time}")
    parser.set_defaults(call=remote)

platforms["gcp-local"] = local_cli
platforms["gcp"] = cli
