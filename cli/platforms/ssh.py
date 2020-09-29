from cli.remote import platforms, remote_action
from cli.utils import relpath
import subprocess, tempfile, os, shlex

def ssh(destination, action, argv):
    destination, path = destination.split(":", 1) if ":" in destination \
        else (destination, os.path.basename(relpath()))
    qualified = f"{destination}:{path}"
    cmd = action + " " + " ".join(map(shlex.quote, argv))
    with tempfile.TemporaryDirectory() as base:
        ctl = os.path.join(base, "ctl")
        rsync = ["rsync", "-e", f"ssh -S {shlex.quote(ctl)}", "--copy-links",
                 "--delete", "-r"]
        try:
            subprocess.run(["virtualenv", os.path.join(base, "venv")])
            subprocess.run(["ssh", "-nNfM", "-S", ctl, destination])
            subprocess.run(rsync + [relpath() + "/", qualified])
            subprocess.run(rsync + [os.path.join(base, "venv"), qualified])
            subprocess.run(["ssh", "-S", ctl, destination, (
                f"cd {shlex.quote(path)} && source venv/bin/activate && "
                f"pip install -e . && python cli {cmd}")])
        finally:
            subprocess.run(["ssh", "-O", "exit", "-S", ctl])

def cli(parser):
    parser.add_argument("destination", help=(
        "remote ip or hostname (accepts ~/.ssh/config aliases), optionally "
        "followed by a colon and the path on the remote machine relative to "
        "the user's home directory where the model should be stored"))
    parser.add_argument("action", action=remote_action)
    parser.set_defaults(call=ssh)

platforms["ssh"] = cli
