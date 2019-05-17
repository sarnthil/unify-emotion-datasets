#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import requests


def unknown(key, value, droot, dataset):
    print(f"==> Unknown action called {key}")


def download(_, target, droot, dataset):
    url = target["url"]
    fname = target.get("target", url.split("/")[-1])

    r = requests.get(
        url,
        stream=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1 Safari/605.1.15"
        },
    )
    chars = "-\\|/"
    with open(f"{droot}/{fname}", "wb") as f:
        for i, chunk in enumerate(r.iter_content(chunk_size=1024)):
            print(f"==> Downloading... {chars[i%len(chars)]}", end="\r")
            if chunk:
                f.write(chunk)

    if fname.endswith(".zip") or fname.endswith(".tar.gz"):
        print(f"==> Unpacking {fname}...")
        shutil.unpack_archive(f"{droot}/{fname}", droot)


def license(_, target, droot, dataset):
    if target.endswith(".txt"):
        input(
            f"==> You will now see the license of {dataset}. Press [q] to exit the pager program. Press [Return]."
        )
        subprocess.run(["less", target])
    else:
        print(target)
    if input(
        f"==> Do you agree with this license for {dataset}? [y/N] "
    ).strip().lower() not in ["yes", "y"]:
        raise PermissionError("Did not agree to license")


def message(_, target, droot, dataset):
    print(f"==> Message from {dataset}:")
    print(target)


def cite(_, target, droot, dataset):
    print(f"==> If using {dataset}, please cite:\n")
    print(target)


def command(_, target, droot, dataset):
    subprocess.run(target, cwd=droot, shell=True)


def git(_, target, droot, dataset):
    shutil.rmtree(droot)
    subprocess.run(["git", "clone", target, droot])


handlers = {
    "download": download,
    "license": license,
    "cite": cite,
    "command": command,
    "git": git,
    "message": message,
}


def main():
    with open("sources.json") as f:
        data = json.load(f)
        root = data["_settings"].get("folder", "datasets")
        os.makedirs(root, exist_ok=True)
        for dataset in data:
            if dataset.startswith("_"):
                continue
            actions = data[dataset]
            droot = f"{root}/{dataset}"
            if os.path.exists(droot):
                print(f"==> {dataset} already exists, skipping...")
                continue
            print("==> Working on", dataset)
            os.makedirs(droot, exist_ok=True)
            try:
                for action in actions:
                    key, value = [*action.items()][0]
                    handlers.get(key, unknown)(key, value, droot, dataset)
            except Exception as e:
                print(f"==> Can't continue on {dataset}, removing...")
                print(e)
                shutil.rmtree(droot, ignore_errors=True)
            if os.path.exists(droot) and not os.listdir(droot):
                shutil.rmtree(droot, ignore_errors=True)
            input(f"==> Done with {dataset}, press [Return] to continue...")
            print()
        print("==> All done")


def test_requirements():
    print("==> Testing requirements")
    try:
        subprocess.run(["git", "help"], stdout=subprocess.DEVNULL)
    except FileNotFoundError:
        print("==> Fatal error: Missing git executable.")
        print(
            {
                "darwin": "Install it with: brew install git",
                "linux": "Install it with: sudo apt-get install git",
            }.get(sys.platform, "Consult your administrator on how to install git")
        )
        sys.exit(1)
    try:
        subprocess.run(["less"], stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("==> Fatal error: Missing less executable.")
        print("We suggest installing the Windows Subsystem for Linux: https://docs.microsoft.com/windows/wsl/")
        sys.exit(1)
    print("==> All requirements met.")


if __name__ == "__main__":
    test_requirements()
    main()

