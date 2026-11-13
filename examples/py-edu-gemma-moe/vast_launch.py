#!/usr/bin/env python3
"""Launch one py-edu-gemma-moe cell on Vast.ai and destroy it when it finishes.

SkyPilot's Vast backend calls ``/api/v0/instances`` to list instances, which Vast
retired (410 Gone), so ``sky`` cannot see, reuse or destroy what it rents. This
script talks to the vastai SDK directly instead: it pins 1x RTX 3090, prints the
offer and its price before renting, verifies the GPU that was actually
allocated, and tears the instance down when training ends.

    python vast_launch.py configs/moe.yaml --dry-run   # show the offer, rent nothing
    python vast_launch.py configs/moe.yaml             # confirm, then launch
    python vast_launch.py --list                       # running instances and their price
    python vast_launch.py --ssh 12345678               # print the SSH command for one
    python vast_launch.py --logs 12345678              # follow that instance's training log
    python vast_launch.py --destroy 12345678           # clean up an orphan

The job writes ``train.log`` and ``.done`` on the instance. Once the job exits,
the instance destroys itself 15 minutes later even if this process dies.
Normally this script sees ``.done`` first, streams the log into
``outputs/vast/<name>/``, and destroys the instance immediately.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from vastai import VastAI

ROOT = Path(__file__).resolve().parent
IMAGE = "vastai/base-image:@vastai-automatic-tag"
GPU = "RTX_3090"
GPU_NAME = "RTX 3090"
DISK_GB = 50
REMOTE_DIR = "~/py-edu-gemma-moe"
REMOTE_TIMEOUT = 22 * 3600  # kill a hung job so the instance can go away
WAIT_HOURS = 23
SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "LogLevel=ERROR",
]
DONE_MARKER = "__VAST_DONE_9f3c__"


def read_api_key() -> str:
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        return key
    for path in (Path.home() / ".config/vastai/vast_api_key", Path.home() / ".vast_api_key"):
        if path.is_file() and (key := path.read_text().strip()):
            return key
    sys.exit("No Vast API key: set VAST_API_KEY or write ~/.config/vastai/vast_api_key")


def get_instance(api: VastAI, instance_id: int) -> dict:
    response = api.client.get(f"/api/v0/instances/{instance_id}/", query_args={"owner": "me"})
    response.raise_for_status()
    instance = response.json().get("instances")
    if isinstance(instance, list):
        instance = instance[0] if instance else None
    if not instance:
        raise RuntimeError(f"instance {instance_id} is gone")
    return instance


def list_instances(api: VastAI) -> list[dict]:
    # The SDK's show_instances() still calls the retired v0 listing endpoint.
    response = api.client.get("/api/v1/instances/", query_args={})
    response.raise_for_status()
    return response.json().get("instances", [])


def ssh_key() -> Path:
    key = ROOT / "outputs" / "vast" / "id_ed25519"
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    return key


def ssh_target(instance: dict, key: Path) -> list[str]:
    host = instance.get("ssh_host") or instance.get("public_ipaddr")
    port = instance.get("ssh_port")
    if not port:
        mapping = (instance.get("ports") or {}).get("22/tcp") or []
        port = mapping[0].get("HostPort") if mapping else None
    if not host or not port:
        raise RuntimeError("instance has no SSH endpoint")
    return ["ssh", "-i", str(key), "-p", str(port), *SSH_OPTS, f"root@{host}"]


def run_ssh(ssh: list[str], command: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(ssh + [command], **kwargs)


def pick_offer(api: VastAI, max_price: float) -> dict:
    query = (
        f"gpu_name={GPU} num_gpus=1 rentable=true rented=false dph_total<={max_price:.4f} "
        f"disk_space>={DISK_GB} reliability>0.98 inet_down>=500 "
        "cpu_ram>=32 cpu_cores_effective>=8 cuda_max_good>=13.0"
    )
    offers = api.search_offers(query=query, storage=DISK_GB)
    matches = [
        offer
        for offer in (offers if isinstance(offers, list) else [])
        if offer.get("gpu_name") == GPU_NAME and offer.get("num_gpus") == 1
    ]
    if not matches:
        sys.exit(f"No 1x {GPU_NAME} offer under ${max_price:.2f}/h matches the filters")
    return min(matches, key=lambda offer: offer["dph_total"])


def describe(offer: dict) -> str:
    return (
        f"offer {offer['id']}: 1x {offer['gpu_name']}, {offer['cpu_cores_effective']:.0f} vCPU, "
        f"{offer['cpu_ram'] / 1024:.0f} GB RAM, {offer['disk_space']:.0f} GB disk, "
        f"{offer['geolocation']}, {offer['inet_down']:.0f} Mbps, ${offer['dph_total']:.3f}/h"
    )


def create_instance(api: VastAI, offer: dict, name: str) -> int:
    result = api.create_instance(
        id=offer["id"],
        image=IMAGE,
        disk=DISK_GB,
        label=name,
        cancel_unavail=True,
    )
    return int(result["new_contract"])


def wait_until_running(api: VastAI, instance_id: int) -> dict:
    deadline = time.time() + 15 * 60
    while time.time() < deadline:
        instance = get_instance(api, instance_id)
        status = instance.get("actual_status")
        if status == "running":
            return instance
        if status in {"exited", "unknown", "offline"}:
            raise RuntimeError(f"instance {instance_id} entered {status}: {instance.get('status_msg')}")
        time.sleep(10)
    raise TimeoutError(f"instance {instance_id} did not reach running in 15 min")


def attach_key(api: VastAI, instance_id: int, key: Path) -> None:
    # Docker instances ignore keys written by onstart, so register them via the API.
    api.attach_ssh(instance_id, key.with_suffix(".pub").read_text().strip())


def verify(instance: dict) -> None:
    if instance.get("gpu_name") != GPU_NAME or instance.get("num_gpus") != 1:
        raise RuntimeError(f"rented {instance.get('gpu_name')} x{instance.get('num_gpus')} instead of 1x {GPU_NAME}")


def wait_for_ssh(api: VastAI, instance_id: int, key: Path) -> list[str]:
    deadline = time.time() + 10 * 60
    while time.time() < deadline:
        try:
            ssh = ssh_target(get_instance(api, instance_id), key)
            run_ssh(ssh, "true", capture_output=True, timeout=20, check=True)
            return ssh
        except (RuntimeError, subprocess.SubprocessError, subprocess.CalledProcessError):
            time.sleep(10)
    raise TimeoutError("SSH did not come up")


def _stream_tar(ssh: list[str], args: list[str], extract_cmd: str) -> None:
    producer = subprocess.Popen(args, stdout=subprocess.PIPE)
    assert producer.stdout is not None
    run_ssh(ssh, extract_cmd, stdin=producer.stdout, check=True)
    producer.stdout.close()
    if producer.wait() != 0:
        raise RuntimeError("tar failed while uploading")


def upload(ssh: list[str]) -> None:
    excludes = [".venv", "outputs", "__pycache__", "data", ".git", ".pytest_cache", "*.egg-info"]
    tar = ["tar", "-czf", "-", "-C", str(ROOT), *[f"--exclude={pattern}" for pattern in excludes], "."]
    _stream_tar(ssh, tar, f"mkdir -p {REMOTE_DIR} && tar -xzf - -C {REMOTE_DIR}")

    # The fitted tokenizer is keyed by dataset, split and vocab size only, so it is
    # uploaded as-is instead of costing a 10-25 min refit on every instance.
    tokenizers = sorted((ROOT / "outputs" / "tokenizers").glob("*.json"))
    if tokenizers:
        files = [str(path.relative_to(ROOT)) for path in tokenizers]
        _stream_tar(ssh, ["tar", "-czf", "-", "-C", str(ROOT), *files], f"tar -xzf - -C {REMOTE_DIR}")


def copy_clearml(ssh: list[str]) -> None:
    conf = Path.home() / "clearml.conf"
    if not conf.is_file():
        raise FileNotFoundError("~/clearml.conf is required by logger: clearml")
    with conf.open("rb") as handle:
        run_ssh(ssh, "cat > ~/clearml.conf", stdin=handle, check=True)


def launch(ssh: list[str], instance_id: int, config: Path, keep: bool, api_key: str) -> None:
    inner = (
        f"timeout {REMOTE_TIMEOUT} uv run python main.py {shlex.quote(str(config))} > train.log 2>&1; echo $? > .done"
    )
    if not keep:
        inner += (
            "; sleep 900; "
            f"curl -s -X DELETE -H {shlex.quote(f'Authorization: Bearer {api_key}')} "
            f"https://console.vast.ai/api/v0/instances/{instance_id}/ >/dev/null"
        )
    command = (
        f"cd {REMOTE_DIR} && rm -f train.log .done && "
        'export PATH="$HOME/.local/bin:$PATH" && '
        # The braces matter: `cd DIR && nohup X &` backgrounds the whole list, and that
        # subshell holds the SSH channel open until X finishes, blocking stream().
        f"{{ nohup bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & }} && echo started"
    )
    run_ssh(ssh, command, check=True)


def stream(ssh: list[str], api: VastAI, instance_id: int, name: str) -> None:
    log_dir = ROOT / "outputs" / "vast" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    failures = 0
    deadline = time.time() + WAIT_HOURS * 3600
    # Written as it streams, so a log survives even if the instance disappears first.
    with (log_dir / "train.log").open("wb") as local_log:
        while time.time() < deadline:
            try:
                result = run_ssh(
                    ssh,
                    f"tail -c +{offset + 1} {REMOTE_DIR}/train.log 2>/dev/null; "
                    f"test -f {REMOTE_DIR}/.done && echo {DONE_MARKER}; true",
                    capture_output=True,
                    timeout=120,
                    check=True,
                )
                failures = 0
            except subprocess.SubprocessError:
                failures += 1
                if failures >= 5:
                    status = get_instance(api, instance_id).get("actual_status")
                    if status not in {"running", "loading"}:
                        raise RuntimeError(f"instance {instance_id} is {status}") from None
                    failures = 0
                time.sleep(30)
                continue
            text = result.stdout.decode(errors="replace")
            chunk = text.split(DONE_MARKER)[0]
            if chunk:
                print(chunk, end="", flush=True)
                local_log.write(chunk.encode())
                local_log.flush()
                offset += len(chunk.encode())
            if DONE_MARKER in text:
                break
            time.sleep(20)
        else:
            raise TimeoutError(f"still running after {WAIT_HOURS} h")

    code = run_ssh(ssh, f"cat {REMOTE_DIR}/.done", capture_output=True, check=True).stdout.decode().strip()
    print(f"training exited with code {code}; log in {log_dir / 'train.log'}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="training config, e.g. configs/moe.yaml")
    parser.add_argument("--name", help="instance label (default: py-edu-<config stem>)")
    parser.add_argument("--max-price", type=float, default=0.30, help="max $/h for the offer (default: 0.30)")
    parser.add_argument("--dry-run", action="store_true", help="print the offer and exit")
    parser.add_argument("--yes", action="store_true", help="skip the launch confirmation")
    parser.add_argument("--keep", action="store_true", help="leave the instance running after training")
    parser.add_argument("--list", action="store_true", help="list instances")
    parser.add_argument("--ssh", dest="ssh_id", type=int, metavar="ID", help="print the SSH command for one instance")
    parser.add_argument("--logs", dest="logs_id", type=int, metavar="ID", help="follow one instance's training log")
    parser.add_argument("--destroy", type=int, metavar="ID", help="destroy one instance")
    args = parser.parse_args()

    api_key = read_api_key()
    api = VastAI(api_key=api_key)

    if args.list:
        instances = list_instances(api)
        for instance in sorted(instances, key=lambda item: item.get("start_date", 0)):
            print(
                f"{instance['id']}  {instance.get('actual_status', '?'):8s} "
                f"{instance.get('gpu_name', '?')} x{instance.get('num_gpus', 1)}  "
                f"${instance.get('dph_total', 0):.3f}/h  {instance.get('label') or '-'}  "
                f"{instance.get('geolocation', '')}"
            )
        if not instances:
            print("no instances")
        return
    if args.destroy is not None:
        api.destroy_instance(id=args.destroy)
        print(f"destroyed {args.destroy}")
        return
    if args.ssh_id is not None:
        print(" ".join(ssh_target(get_instance(api, args.ssh_id), ssh_key())))
        return
    if args.logs_id is not None:
        ssh = ssh_target(get_instance(api, args.logs_id), ssh_key())
        sys.exit(subprocess.run(ssh + [f"tail -n 40 -F {REMOTE_DIR}/train.log"]).returncode)
    if not args.config:
        parser.error("a config is required unless --list, --ssh, --logs or --destroy is used")

    config = (ROOT / args.config).resolve()
    if not config.is_file() or not config.is_relative_to(ROOT):
        parser.error(f"{args.config} is not a config inside {ROOT}")
    name = args.name or f"py-edu-{config.stem}"

    offer = pick_offer(api, args.max_price)
    print(describe(offer))
    if args.dry_run:
        return
    if not args.yes:
        answer = input(f"Launch {config.name} on it? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            sys.exit("aborted")

    key = ssh_key()
    instance_id = None
    ssh: list[str] | None = None
    started = False
    finished = False
    try:
        instance_id = create_instance(api, offer, name)
        print(f"instance {instance_id} created; waiting for it to run")
        instance = wait_until_running(api, instance_id)
        verify(instance)
        print(f"running: 1x {instance['gpu_name']} for ${instance.get('dph_total', 0):.3f}/h")
        attach_key(api, instance_id, key)
        ssh = wait_for_ssh(api, instance_id, key)
        upload(ssh)
        if "logger: clearml" in config.read_text():
            copy_clearml(ssh)
        print("installing dependencies")
        run_ssh(
            ssh,
            f"cd {REMOTE_DIR} && {{ command -v uv >/dev/null 2>&1 || "
            "curl -LsSf https://astral.sh/uv/install.sh | sh; } && "
            'export PATH="$HOME/.local/bin:$PATH" && uv sync',
            check=True,
        )
        launch(ssh, instance_id, config.relative_to(ROOT), args.keep, api_key)
        started = True
        stream(ssh, api, instance_id, name)
        finished = True
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if instance_id is not None:
            if args.keep:
                print(f"kept instance {instance_id}")
                if ssh:
                    print("attach with: " + " ".join(ssh))
            elif started and not finished:
                print(
                    f"training continues on {instance_id}; it will destroy itself 15 min after it exits.\n"
                    f"stop it now with: python vast_launch.py --destroy {instance_id}"
                )
            else:
                try:
                    api.destroy_instance(id=instance_id)
                    print(f"destroyed {instance_id}")
                except Exception as error:
                    print(f"could not destroy {instance_id}: {error}", file=sys.stderr)
                    print(f"run: python vast_launch.py --destroy {instance_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
