#!/usr/bin/env python3
"""
NCCL Scout for Kubernetes (Nebius-friendly, TCP-only, no OCI/Slurm deps)

Fix included:
- run.sh now uses a dedicated workdir (/tmp/nccl-scout) and creates ordered_hostfile there
  BEFORE using it, so you won't see:
    /scripts/run.sh: line XX: ordered_hostfile: No such file or directory
"""

import os
import sys
import time
import json
import shlex
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from textwrap import dedent

# ----------------------------
# Settings / env overrides
# ----------------------------
KUBECTL = os.environ.get("KUBECTL", "kubectl")

NAMESPACE = os.environ.get("NCCL_SCOUT_NAMESPACE", "nccl-scout")
SECRET_NAME = "ssh-auth"
SCRIPTS_CM = "nccl-scout-scripts"
MPI_FILES_CM = "mpi-files"
LAUNCHER_POD = "nccl-scout-launcher"

SSH_PORT = int(os.environ.get("NCCL_SSH_PORT", "2222"))
IMAGE = os.environ.get("NCCL_SCOUT_IMAGE", "nvcr.io/nvidia/pytorch:24.01-py3")

RUNS = int(os.environ.get("NCCL_RUNS", "1"))
EXEC = os.environ.get("NCCL_EXEC", "all_reduce_perf")

TEST_MIN_BYTES = os.environ.get("NCCL_MIN_BYTES", "1G")
TEST_MAX_BYTES = os.environ.get("NCCL_MAX_BYTES", "10G")
TEST_ITERS = os.environ.get("NCCL_ITERS", "20")

CLEAN = os.environ.get("NCCL_CLEAN", "1") == "1"


# ----------------------------
# Helpers
# ----------------------------
def run(cmd, check=True, capture=False, text=True, **kwargs):
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    if capture:
        p = subprocess.run(
            cmd_list,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=text,
            **kwargs,
        )
        return p.stdout
    return subprocess.run(cmd_list, check=check, text=text, **kwargs)


def info(msg): print(f"[INFO] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def die(msg, code=1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def kubectl(*args, check=True, capture=False, **kwargs):
    return run([KUBECTL, *args], check=check, capture=capture, **kwargs)


def ns_exists(name: str) -> bool:
    rc = subprocess.run([KUBECTL, "get", "ns", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rc.returncode == 0


def delete_namespace(name: str):
    if ns_exists(name):
        info(f"Cleaning previous namespace '{name}' (if any)...")
        kubectl("delete", "ns", name, "--wait=true", "--timeout=180s", check=False)
        for _ in range(240):
            if not ns_exists(name):
                return
            time.sleep(1)
        warn(f"Namespace '{name}' still exists after timeout; continuing.")


def ensure_namespace(name: str):
    if not ns_exists(name):
        kubectl("create", "ns", name)


def discover_gpu_nodes():
    info("Discovering GPU nodes...")
    out = kubectl("get", "nodes", "-o", "json", capture=True)
    data = json.loads(out)
    nodes = []
    for item in data.get("items", []):
        name = item["metadata"]["name"]
        alloc = item.get("status", {}).get("allocatable", {}) or {}
        g = alloc.get("nvidia.com/gpu") or alloc.get("nvidia.com/gpu.shared")
        if not g:
            continue
        try:
            gpus = int(g)
        except Exception:
            continue
        if gpus <= 0:
            continue

        labels = item.get("metadata", {}).get("labels", {}) or {}
        if "node-role.kubernetes.io/control-plane" in labels or "node-role.kubernetes.io/master" in labels:
            continue

        nodes.append({"name": name, "gpus": gpus, "labels": labels})

    if not nodes:
        die("No GPU nodes found (allocatable nvidia.com/gpu missing).")

    def rack_key(n):
        labels = n.get("labels", {})
        for k in [
            "topology.kubernetes.io/zone",
            "failure-domain.beta.kubernetes.io/zone",
            "topology.kubernetes.io/region",
            "nebius.ai/rack",
            "rack",
        ]:
            if k in labels:
                return labels.get(k, "")
        return ""

    nodes_sorted = sorted(nodes, key=lambda n: (rack_key(n), n["name"]))

    max_nodes = os.environ.get("NCCL_MAX_NODES")
    if max_nodes:
        try:
            nodes_sorted = nodes_sorted[: int(max_nodes)]
        except Exception:
            pass

    return [{"name": n["name"], "gpus": n["gpus"]} for n in nodes_sorted]


def create_or_update_secret_from_ssh(namespace: str):
    priv = Path.home() / ".ssh" / "id_rsa"
    pub = Path.home() / ".ssh" / "id_rsa.pub"
    if not priv.exists() or not pub.exists():
        die(f"Missing SSH keypair: {priv} and/or {pub}. Generate with: ssh-keygen -t rsa -b 4096")

    info("Creating SSH secret...")
    ensure_namespace(namespace)
    kubectl("-n", namespace, "delete", "secret", SECRET_NAME, check=False)
    kubectl(
        "-n", namespace, "create", "secret", "generic", SECRET_NAME,
        f"--from-file=id_rsa={str(priv)}",
        f"--from-file=authorized_keys={str(pub)}",
    )


def build_hostfile(selected_nodes):
    # one host per line
    return "\n".join([f"nccl-scout-worker-{i}.{NAMESPACE}.svc.cluster.local" for i in range(len(selected_nodes))]) + "\n"


def build_rankfile(hostfile: str, gpus_per_node: int):
    lines = []
    hosts = [h.strip() for h in hostfile.splitlines() if h.strip()]
    for idx, host in enumerate(hosts):
        for gpu in range(gpus_per_node):
            lines.append(f"rank {idx*gpus_per_node+gpu}={host} slot={gpu}")
    return "\n".join(lines) + "\n"


def build_run_sh(gpus_per_node: int) -> str:
    """
    Key fix:
    - All artifacts (ordered_hostfile, rankfile_pod_dns, logfile) live in /tmp/nccl-scout
    - We 'cd' there first and always reference absolute or local names in that dir.
    """
    return dedent(rf"""\
        #!/usr/bin/env bash
        set -euo pipefail

        max_runs="{RUNS}"
        SSH_PORT="{SSH_PORT}"

        # Mounted from ConfigMap
        IN_HOSTFILE="/etc/mpi/hostfile"
        IN_RANKFILE="/etc/mpi/rankfile"

        # Dedicated workdir so relative files always exist where we expect
        WORKDIR="/tmp/nccl-scout"
        mkdir -p "${{WORKDIR}}"
        cd "${{WORKDIR}}"

        ORDEREDHOSTFILE="${{WORKDIR}}/ordered_hostfile"
        RANKFILE_POD_DNS="${{WORKDIR}}/rankfile_pod_dns"
        LOGFILE="${{WORKDIR}}/nccl_run_allreduce.sh.log"

        echo "INPUTFILE"
        cat "${{IN_HOSTFILE}}"

        # Create ordered hostfile (no-op ordering for now)
        cp "${{IN_HOSTFILE}}" "${{ORDEREDHOSTFILE}}"

        # Prefer provided rankfile if present; else generate
        if [[ -f "${{IN_RANKFILE}}" ]]; then
          cp "${{IN_RANKFILE}}" "${{RANKFILE_POD_DNS}}"
        else
          awk '{{for(i=0;i<{gpus_per_node};i++) print "rank " ((NR-1)*{gpus_per_node}+i) "=" $1 " slot=" i}}' "${{ORDEREDHOSTFILE}}" > "${{RANKFILE_POD_DNS}}"
        fi

        echo "ORDEREDHOSTFILE"
        cat "${{ORDEREDHOSTFILE}}"
        echo "RANKFILE"
        cat "${{RANKFILE_POD_DNS}}"

        np=$(( $(wc -l < "${{ORDEREDHOSTFILE}}") * {gpus_per_node} ))

        OMPI_SSH_OPTS="-p ${{SSH_PORT}} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        export OMPI_MCA_plm_rsh_agent="ssh ${{OMPI_SSH_OPTS}}"

        # TCP-only (Nebius safe)
        export NCCL_DEBUG="${{NCCL_DEBUG:-WARN}}"
        export NCCL_IB_DISABLE=1
        export NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-eth0}}"
        export UCX_TLS="${{UCX_TLS:-tcp}}"
        export UCX_NET_DEVICES="${{UCX_NET_DEVICES:-eth0}}"

        echo "Detected shape on $(head -n 1 "${{ORDEREDHOSTFILE}}"): UNKNOWN"
        echo "Requested MODE: tcp"
        echo "NCCL NET plugin: $(ls /opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so 2>/dev/null || echo none)"

        echo "[INFO] Installing deps..."
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
          openssh-client openssh-server openmpi-bin libopenmpi-dev git jq make g++ iproute2 netcat-traditional

        # Ensure nccl-tests exists (prefer /opt/oci-hpc if it exists, else build in /tmp)
        EXEC_CMD=""
        if [[ -x "/opt/oci-hpc/nccl-test/build/${{EXEC:-%s}}" ]]; then
          EXEC_CMD="/opt/oci-hpc/nccl-test/build/${{EXEC:-%s}}"
        else
          cd /tmp
          if [[ ! -d nccl-tests ]]; then
            git clone -q https://github.com/NVIDIA/nccl-tests.git
          fi
          cd /tmp/nccl-tests
          make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda -j"$(nproc)"
          EXEC_CMD="/tmp/nccl-tests/build/${{EXEC:-%s}}"
        fi

        if [[ ! -x "${{EXEC_CMD}}" ]]; then
          echo "ERROR: Test executable not found/executable: ${{EXEC_CMD}}"
          exit 1
        fi

        echo ""
        echo "Testing TCP ${{SSH_PORT}} reachability..."
        while read -r host; do
          [[ -z "${{host:-}}" ]] && continue
          echo "== $host =="
          nc -vz -w3 "$host" "${{SSH_PORT}}" || true
        done < "${{ORDEREDHOSTFILE}}"

        echo ""
        echo "Testing SSH..."
        while read -r host; do
          [[ -z "${{host:-}}" ]] && continue
          echo -n "  $host: "
          ssh ${{OMPI_SSH_OPTS}} -o ConnectTimeout=10 -i /root/.ssh/id_rsa root@"$host" hostname && echo " OK" || echo " FAIL"
        done < "${{ORDEREDHOSTFILE}}"

        # If we built /tmp/nccl-tests, sync to workers so mpirun can exec remotely
        if [[ "${{EXEC_CMD}}" == /tmp/nccl-tests/* ]]; then
          echo ""
          echo "[INFO] Syncing /tmp/nccl-tests to workers..."
          while read -r host; do
            [[ -z "${{host:-}}" ]] && continue
            echo "  -> syncing to $host"
            tar -C /tmp -cf - nccl-tests | \
              ssh ${{OMPI_SSH_OPTS}} -o ConnectTimeout=10 -i /root/.ssh/id_rsa root@"$host" "mkdir -p /tmp && tar -C /tmp -xf -" || true
          done < "${{ORDEREDHOSTFILE}}"
        fi

        # Run loop
        for x in $(seq 1 1 "${{max_runs}}"); do
          echo "${{x}}"
          echo "${{x}}" >> "${{LOGFILE}}"
          date >> "${{LOGFILE}}"

          echo ""
          echo "[RUN ${{x}}] mpirun allreduce..."
          mpirun --mca pml ucx \
            --allow-run-as-root \
            -mca plm_rsh_args "${{OMPI_SSH_OPTS}}" \
            --bind-to none \
            --mca coll ^hcoll \
            -x NCCL_DEBUG \
            -x NCCL_IB_DISABLE \
            -x NCCL_SOCKET_IFNAME \
            -x UCX_TLS \
            -x UCX_NET_DEVICES \
            --np "${{np}}" --rankfile "${{RANKFILE_POD_DNS}}" "${{EXEC_CMD}}" \
              -b{TEST_MIN_BYTES} -e{TEST_MAX_BYTES} -f 2 -g 1 -n {TEST_ITERS} \
              2>&1 | tee -a "${{LOGFILE}}"

          echo ""
          echo "----- Tail (last 80 lines) -----"
          tail -n 80 "${{LOGFILE}}" || true
          echo "--------------------------------"
        done

        echo ""
        echo "[DONE] NCCL all_reduce finished. Full log is in ${{LOGFILE}}"
    """ % (EXEC, EXEC, EXEC))


def apply_configmaps(namespace: str, run_sh: str, hostfile: str, rankfile: str):
    info("Creating/updating configmaps (run.sh + mpi-files)...")
    kubectl("-n", namespace, "delete", "configmap", SCRIPTS_CM, check=False)
    kubectl("-n", namespace, "delete", "configmap", MPI_FILES_CM, check=False)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "run.sh").write_text(run_sh, encoding="utf-8")
        (td / "hostfile").write_text(hostfile, encoding="utf-8")
        (td / "rankfile").write_text(rankfile, encoding="utf-8")

        kubectl("-n", namespace, "create", "configmap", SCRIPTS_CM, f"--from-file=run.sh={td/'run.sh'}")
        kubectl("-n", namespace, "create", "configmap", MPI_FILES_CM,
                f"--from-file=hostfile={td/'hostfile'}",
                f"--from-file=rankfile={td/'rankfile'}")


def wait_workers_ready(namespace: str, count: int, timeout_s: int = 900):
    info(f"Waiting for {count} worker pods to be Ready...")
    start = time.time()
    while True:
        out = kubectl("-n", namespace, "get", "pods", "-l", "app=nccl-scout-worker", "-o", "json", capture=True)
        data = json.loads(out)
        pods = data.get("items", [])
        ready = 0
        for p in pods:
            conds = {c["type"]: c["status"] for c in p.get("status", {}).get("conditions", [])}
            if conds.get("Ready") == "True":
                ready += 1
        if ready >= count:
            info("All worker pods are Ready.")
            return True
        if time.time() - start > timeout_s:
            warn("Timed out waiting for worker pods Ready. Dumping events:")
            ev = kubectl("-n", namespace, "get", "events", "--sort-by=.lastTimestamp", capture=True, check=False)
            print(ev)
            return False
        time.sleep(3)


def wait_launcher_phase(namespace: str, timeout_s: int = 900):
    info("Waiting for launcher pod to start...")
    start = time.time()
    last_phase = None
    while True:
        out = kubectl("-n", namespace, "get", "pod", LAUNCHER_POD, "-o", "json", capture=True, check=False)
        if out:
            podj = json.loads(out)
            phase = podj.get("status", {}).get("phase", "")
            if phase != last_phase:
                info(f"Launcher pod phase: {phase}")
                last_phase = phase
            if phase in ("Running", "Succeeded", "Failed"):
                return phase
        if time.time() - start > timeout_s:
            warn("Timed out waiting for launcher start. Showing describe:")
            desc = kubectl("-n", namespace, "describe", "pod", LAUNCHER_POD, capture=True, check=False)
            print(desc)
            return "Timeout"
        time.sleep(2)


def wait_pod_done(namespace: str, pod: str, timeout_s: int = 3600):
    info(f"Waiting for pod/{pod} to complete...")
    start = time.time()
    last_phase = None
    while True:
        out = kubectl("-n", namespace, "get", "pod", pod, "-o", "json", capture=True, check=False)
        if out:
            podj = json.loads(out)
            phase = podj.get("status", {}).get("phase", "")
            if phase != last_phase:
                info(f"Pod phase: {phase}")
                last_phase = phase
            if phase in ("Succeeded", "Failed"):
                return phase
        if time.time() - start > timeout_s:
            return "Timeout"
        time.sleep(3)


def stream_logs(namespace: str, pod: str, container: str = "launcher"):
    info("Streaming NCCL logs...\n")
    subprocess.call([KUBECTL, "-n", namespace, "logs", "-f", pod, "-c", container])


def build_manifest(selected_nodes):
    gpus_per_node = selected_nodes[0]["gpus"]
    hostfile = build_hostfile(selected_nodes)
    rankfile = build_rankfile(hostfile, gpus_per_node=gpus_per_node)
    run_sh = build_run_sh(gpus_per_node=gpus_per_node)

    pieces = []
    pieces.append(dedent(f"""\
        apiVersion: v1
        kind: Namespace
        metadata:
          name: {NAMESPACE}
    """))

    for i, n in enumerate(selected_nodes):
        node_name = n["name"]
        gpu_count = n["gpus"]

        pieces.append(dedent(f"""\
            ---
            apiVersion: v1
            kind: Service
            metadata:
              name: nccl-scout-worker-{i}
              namespace: {NAMESPACE}
            spec:
              selector:
                app: nccl-scout-worker
                worker-index: "{i}"
              ports:
                - name: ssh
                  port: {SSH_PORT}
                  targetPort: {SSH_PORT}
        """))

        pieces.append(dedent(f"""\
            ---
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: nccl-scout-worker-{i}
              namespace: {NAMESPACE}
            spec:
              replicas: 1
              selector:
                matchLabels:
                  app: nccl-scout-worker
                  worker-index: "{i}"
              template:
                metadata:
                  labels:
                    app: nccl-scout-worker
                    worker-index: "{i}"
                spec:
                  nodeName: {node_name}
                  restartPolicy: Always
                  volumes:
                    - name: ssh-home
                      emptyDir: {{}}
                    - name: ssh-secret
                      secret:
                        secretName: {SECRET_NAME}
                    - name: sshd-run
                      emptyDir: {{}}
                  initContainers:
                    - name: init-ssh
                      image: busybox:1.36
                      command: ["sh","-lc"]
                      args:
                        - |
                          set -e
                          test -f /ssh-secret/authorized_keys
                          mkdir -p /ssh-home
                          chmod 700 /ssh-home
                          cp /ssh-secret/authorized_keys /ssh-home/authorized_keys
                          chmod 600 /ssh-home/authorized_keys
                      volumeMounts:
                        - name: ssh-home
                          mountPath: /ssh-home
                        - name: ssh-secret
                          mountPath: /ssh-secret
                          readOnly: true
                  containers:
                    - name: worker
                      image: {IMAGE}
                      securityContext:
                        privileged: true
                      command: ["/bin/bash","-lc"]
                      args:
                        - |
                          set -euo pipefail
                          apt-get update -qq
                          DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server
                          mkdir -p /run/sshd
                          mkdir -p /root/.ssh
                          chmod 700 /root/.ssh
                          cp /ssh-home/authorized_keys /root/.ssh/authorized_keys
                          chmod 600 /root/.ssh/authorized_keys
                          cat >/etc/ssh/sshd_config <<'EOF'
                          Port {SSH_PORT}
                          ListenAddress 0.0.0.0
                          PermitRootLogin yes
                          PasswordAuthentication no
                          PubkeyAuthentication yes
                          AuthorizedKeysFile .ssh/authorized_keys
                          UsePAM no
                          ChallengeResponseAuthentication no
                          X11Forwarding no
                          PrintMotd no
                          ClientAliveInterval 30
                          ClientAliveCountMax 6
                          EOF
                          /usr/sbin/sshd -D -e
                      ports:
                        - containerPort: {SSH_PORT}
                      resources:
                        limits:
                          nvidia.com/gpu: {gpu_count}
                      volumeMounts:
                        - name: ssh-home
                          mountPath: /ssh-home
                        - name: ssh-secret
                          mountPath: /ssh-secret
                          readOnly: true
                        - name: sshd-run
                          mountPath: /run
        """))

    pieces.append(dedent(f"""\
        ---
        apiVersion: v1
        kind: Pod
        metadata:
          name: {LAUNCHER_POD}
          namespace: {NAMESPACE}
        spec:
          nodeName: {selected_nodes[0]["name"]}
          restartPolicy: Never
          volumes:
            - name: scripts
              configMap:
                name: {SCRIPTS_CM}
                defaultMode: 0755
            - name: ssh-auth
              secret:
                secretName: {SECRET_NAME}
                defaultMode: 0600
            - name: mpi-files
              configMap:
                name: {MPI_FILES_CM}
                items:
                  - key: hostfile
                    path: hostfile
                  - key: rankfile
                    path: rankfile
          initContainers:
            - name: init-ssh
              image: busybox:1.36
              command: ["sh","-lc"]
              args:
                - |
                  set -e
                  mkdir -p /root/.ssh
                  chmod 700 /root/.ssh
                  cp /ssh/id_rsa /root/.ssh/id_rsa
                  chmod 600 /root/.ssh/id_rsa
              volumeMounts:
                - name: ssh-auth
                  mountPath: /ssh
                  readOnly: true
          containers:
            - name: launcher
              image: {IMAGE}
              securityContext:
                privileged: true
              command: ["/bin/bash","-lc"]
              args: ["/scripts/run.sh"]
              volumeMounts:
                - name: scripts
                  mountPath: /scripts
                - name: ssh-auth
                  mountPath: /root/.ssh
                  readOnly: true
                - name: mpi-files
                  mountPath: /etc/mpi
    """))

    return "\n".join(pieces), run_sh, hostfile, rankfile


def main():
    if CLEAN:
        delete_namespace(NAMESPACE)

    ensure_namespace(NAMESPACE)
    nodes = discover_gpu_nodes()

    info("Selected nodes (in order):")
    for i, n in enumerate(nodes):
        print(f"  {i:02d}. {n['name']} gpus={n['gpus']}")

    create_or_update_secret_from_ssh(NAMESPACE)

    manifest, run_sh, hostfile, rankfile = build_manifest(nodes)
    apply_configmaps(NAMESPACE, run_sh, hostfile, rankfile)

    with tempfile.TemporaryDirectory(prefix="nccl-scout-") as td:
        td = Path(td)
        ypath = td / "nccl-scout.yaml"
        ypath.write_text(manifest, encoding="utf-8")
        kubectl("apply", "-f", str(ypath))

    wait_workers_ready(NAMESPACE, count=len(nodes), timeout_s=900)

    phase = wait_launcher_phase(NAMESPACE, timeout_s=900)
    if phase == "Timeout":
        die("Launcher did not start in time.")

    # Stream logs (so you see the nccl-tests table live)
    try:
        stream_logs(NAMESPACE, LAUNCHER_POD, "launcher")
    except KeyboardInterrupt:
        warn("Stopped streaming (Ctrl+C). Job continues on cluster.")

    final_phase = wait_pod_done(NAMESPACE, LAUNCHER_POD, timeout_s=3600)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path.cwd() / f"nccl_scout_{ts}.log"

    info("Fetching NCCL logs...")
    out = kubectl("-n", NAMESPACE, "logs", LAUNCHER_POD, "-c", "launcher", capture=True, check=False) or ""
    print(out)
    log_path.write_text(out, encoding="utf-8")
    info(f"Saved log → {log_path}")
    info(f"Launcher phase was: {final_phase}")


if __name__ == "__main__":
    main()
