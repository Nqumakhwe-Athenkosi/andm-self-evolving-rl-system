#!/usr/bin/env python3

# Capture baseline or ANDM-inspired Mininet measurements into two CSV files.

import argparse
import csv
import json
import os
import pwd
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from mininet.clean import cleanup
from mininet.link import TCLink
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.topo import Topo


# Keep the offered workload equal so the network condition is the main change.
PHASES = [
    {
        "name": "Morning",
        "flows": [("matatiele", 5.0)],
        "cpu_workers": 0,
    },
    {
        "name": "Afternoon",
        "flows": [("matatiele", 15.0), ("winnie", 15.0)],
        "cpu_workers": 0,
    },
    {
        "name": "Night",
        "flows": [("umzimvubu", 8.0)],
        "cpu_workers": 0,
    },
]


# These are controlled test settings and not measured ANDM values.
SCENARIOS = {
    "baseline": {
        "research_name": "baseline",
        "core_bw_mbps": 100.0,
        "branch_bw_mbps": 100.0,
        "wan_delay_ms": 1.0,
        "wan_loss_pct": 0.0,
    },
    "rural": {
        "research_name": "andm_inspired_rural",
        "core_bw_mbps": 20.0,
        "branch_bw_mbps": 20.0,
        "wan_delay_ms": 50.0,
        "wan_loss_pct": 2.0,
    },
}


# The four branches are rotated during latency and packet-loss sampling.
BRANCHES = ["matatiele", "umzimvubu", "ntabankulu", "winnie"]


# Exclude traffic-change activity from the measured period samples.
PHASE_COOLDOWN_SECONDS = 5
PHASE_SETTLE_SECONDS = 5


# Build an ANDM-inspired five-site topology without claiming it is the real network.
class ANDMTopo(Topo):
    def build(self, config):
        hq = self.addHost("hq", ip="10.0.0.1/24")
        matatiele = self.addHost("matatiele", ip="10.0.0.2/24")
        umzimvubu = self.addHost("umzimvubu", ip="10.0.0.3/24")
        ntabankulu = self.addHost("ntabankulu", ip="10.0.0.4/24")
        winnie = self.addHost("winnie", ip="10.0.0.5/24")

        s1 = self.addSwitch("s1", protocols="OpenFlow13")
        s2 = self.addSwitch("s2", protocols="OpenFlow13")
        s3 = self.addSwitch("s3", protocols="OpenFlow13")
        s4 = self.addSwitch("s4", protocols="OpenFlow13")
        s5 = self.addSwitch("s5", protocols="OpenFlow13")

        self.addLink(
            hq,
            s1,
            cls=TCLink,
            bw=config["core_bw_mbps"],
            delay="1ms",
            loss=0,
            use_tbf=True,
        )

        branch_pairs = [
            (matatiele, s2),
            (umzimvubu, s3),
            (ntabankulu, s4),
            (winnie, s5),
        ]

        for host, switch in branch_pairs:
            self.addLink(
                host,
                switch,
                cls=TCLink,
                bw=100,
                delay="1ms",
                loss=0,
                use_tbf=True,
            )
            self.addLink(
                switch,
                s1,
                cls=TCLink,
                bw=config["branch_bw_mbps"],
                delay=f'{config["wan_delay_ms"]}ms',
                loss=config["wan_loss_pct"],
                use_tbf=True,
            )


# Read the Linux CPU counters used to calculate interval percentages.
def read_cpu_counters():
    first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    values = [int(value) for value in first_line.split()[1:]]
    while len(values) < 8:
        values.append(0)
    return values


# Convert two CPU-counter readings into research CSV values.
def calculate_cpu_percentages(previous, current):
    differences = [max(0, new - old) for old, new in zip(previous, current)]
    total = sum(differences)
    if total <= 0:
        return 0.0, 0.0, 100.0, 0.0, 0.0
    user = 100.0 * (differences[0] + differences[1]) / total
    system = 100.0 * (differences[2] + differences[5] + differences[6]) / total
    idle = 100.0 * differences[3] / total
    iowait = 100.0 * differences[4] / total
    used = 100.0 - idle
    return user, system, idle, used, iowait


# Parse average latency and packet loss from one quiet ping command.
def parse_ping(text):
    loss_match = re.search(r"([0-9.]+)% packet loss", text)
    latency_match = re.search(
        r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = [0-9.]+/([0-9.]+)",
        text,
    )
    loss = float(loss_match.group(1)) if loss_match else 100.0
    latency = float(latency_match.group(1)) if latency_match else ""
    return latency, loss


# Read the receiving byte counter on the district-office host interface.
def read_hq_rx_bytes(hq):
    interface_name = str(hq.defaultIntf())
    value = hq.cmd(
        f"cat /sys/class/net/{interface_name}/statistics/rx_bytes"
    ).strip()
    return int(value)


# Decode an Open vSwitch JSON map into a normal Python dictionary.
def decode_ovs_map(value):
    if isinstance(value, list) and len(value) == 2 and value[0] == "map":
        return dict(value[1])
    return {}


# Capture cumulative counters for every interface on switches s1 to s5.
def read_ovs_counters():
    command = [
        "ovs-vsctl",
        "--format=json",
        "--columns=name,statistics",
        "list",
        "Interface",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    payload = json.loads(completed.stdout)
    headings = payload["headings"]
    rows = []

    for raw_row in payload["data"]:
        item = dict(zip(headings, raw_row))
        interface_name = item.get("name", "")
        if not re.fullmatch(r"s[1-5](?:-eth[0-9]+)?", interface_name):
            continue
        switch = interface_name.split("-")[0]
        port = "LOCAL:" if interface_name == switch else interface_name.split("eth")[-1] + ":"
        statistics = decode_ovs_map(item.get("statistics"))
        rows.append(
            {
                "switch": switch,
                "port": port,
                "rx_packets": int(statistics.get("rx_packets", 0)),
                "tx_packets": int(statistics.get("tx_packets", 0)),
                "rx_bytes": int(statistics.get("rx_bytes", 0)),
                "tx_bytes": int(statistics.get("tx_bytes", 0)),
            }
        )

    return sorted(rows, key=lambda row: (row["switch"], row["port"]))


# Start one iperf3 server and client for each phase flow.
def start_traffic(net, phase, duration):
    hq = net.get("hq")
    processes = []
    traffic_duration = duration * 3 + PHASE_SETTLE_SECONDS + 15
    for index, (source_name, rate_mbps) in enumerate(phase["flows"], start=1):
        port = 5200 + index
        server_output = f"/tmp/andm_iperf_server_{port}.log"
        client_output = f"/tmp/andm_iperf_client_{source_name}_{port}.log"
        server_text = hq.cmd(
            f"iperf3 -s -p {port} -1 > {server_output} 2>&1 & echo $!"
        )
        server_pid = int(server_text.strip().splitlines()[-1])
        time.sleep(0.25)
        source = net.get(source_name)
        client_text = source.cmd(
            f"iperf3 -c {hq.IP()} -p {port} -u -b {rate_mbps}M "
            f"-t {traffic_duration} -i 0 > {client_output} 2>&1 & echo $!"
        )
        client_pid = int(client_text.strip().splitlines()[-1])
        processes.append((source, client_pid))
        processes.append((hq, server_pid))
    return processes


# Stop only the processes created by this script.
def stop_processes(processes):
    for host, process_id in processes:
        host.cmd(f"kill {int(process_id)} 2>/dev/null || true")


# Count traffic processes that ended before all phase samples were recorded.
def count_stopped_processes(processes):
    stopped = 0
    for host, process_id in processes:
        output = host.cmd(
            f"kill -0 {int(process_id)} >/dev/null 2>&1; echo $?"
        ).strip()
        if not output or output.splitlines()[-1] != "0":
            stopped += 1
    return stopped


# Confirm the local ONOS OpenFlow port is accepting connections.
def check_controller(ip_address, port):
    try:
        with socket.create_connection((ip_address, port), timeout=3):
            return
    except OSError as error:
        raise RuntimeError(
            f"ONOS is not reachable at {ip_address}:{port}. Start the ONOS container first."
        ) from error


# Refuse to overwrite evidence unless the user explicitly selects force.
def check_output_paths(paths, force):
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output already exists: {names}. Choose another folder or use --force."
        )


# Write a CSV through a temporary file before replacing its final name.
def write_csv_safely(path, fieldnames, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


# Return files created with sudo to the user who started the command.
def restore_file_owner(paths):
    user_id = os.environ.get("SUDO_UID")
    group_id = os.environ.get("SUDO_GID")
    if not user_id or not group_id:
        return
    for path in paths:
        os.chown(path, int(user_id), int(group_id))


# Print phase averages so each capture can be checked before it is accepted.
def print_capture_summary(cpu_rows, network_rows):
    selected_network_rows = [
        row
        for row in network_rows
        if row["switch"] == "s1" and row["port"] == "1:"
    ]
    print("\nCapture summary")
    for phase in PHASES:
        period = phase["name"]
        cpu_values = [
            float(row["cpu_used"])
            for row in cpu_rows
            if row["simulated_time"] == period
        ]
        period_network_rows = [
            row
            for row in selected_network_rows
            if row["simulated_time"] == period
        ]
        rate_values = [
            float(row["delivered_rate_mbps"])
            for row in period_network_rows
        ]
        latency_values = [
            float(row["latency_ms"])
            for row in period_network_rows
            if row["latency_ms"] != ""
        ]
        loss_values = [
            float(row["packet_loss_pct"])
            for row in period_network_rows
        ]
        latency_text = (
            f"{mean(latency_values):.2f} ms"
            if latency_values
            else "unavailable"
        )
        print(
            f"{period}: CPU mean {mean(cpu_values):.2f}% | "
            f"rate mean {mean(rate_values):.2f} Mbps | "
            f"latency mean {latency_text} | "
            f"loss mean {mean(loss_values):.2f}%"
        )


# Run one scenario and return its CPU and network rows.
def capture_scenario(args, config):
    topology = ANDMTopo(config=config)
    network = Mininet(
        topo=topology,
        controller=None,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        build=False,
    )
    network.addController(
        "c0",
        controller=RemoteController,
        ip=args.controller_ip,
        port=args.controller_port,
    )

    cpu_rows = []
    network_rows = []
    active_processes = []

    try:
        network.build()
        network.start()
        network.waitConnected(timeout=15)
        time.sleep(3)

        warmup_loss_limit = 0.0 if config["wan_loss_pct"] == 0 else 30.0
        packet_loss = network.pingAll(timeout="2")
        if packet_loss > warmup_loss_limit:
            time.sleep(5)
            packet_loss = network.pingAll(timeout="2")
        if packet_loss > warmup_loss_limit:
            raise RuntimeError(
                f"Topology warm-up failed with {packet_loss:.1f}% packet loss."
            )
        if packet_loss > 0:
            print(
                f"Warm-up loss {packet_loss:.1f}% is accepted for the configured rural profile."
            )

        hq = network.get("hq")
        global_sample = 0

        for phase_index, phase in enumerate(PHASES):
            print(f'\nStarting {phase["name"]} ({args.duration_per_period} samples)')
            active_processes = start_traffic(
                network,
                phase,
                args.duration_per_period,
            )
            print(f"  settling for {PHASE_SETTLE_SECONDS} unrecorded seconds")
            time.sleep(PHASE_SETTLE_SECONDS)

            previous_cpu = read_cpu_counters()
            previous_bytes = read_hq_rx_bytes(hq)
            previous_time = time.monotonic()
            phase_start = previous_time

            for phase_sample in range(args.duration_per_period):
                source_name = BRANCHES[global_sample % len(BRANCHES)]
                source = network.get(source_name)
                ping_text = source.cmd(
                    f"ping -n -q -c 3 -i 0.1 -W 1 {hq.IP()}"
                )
                latency_ms, packet_loss_pct = parse_ping(ping_text)

                target_time = phase_start + phase_sample + 1
                remaining = target_time - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

                current_time = time.monotonic()
                current_cpu = read_cpu_counters()
                current_bytes = read_hq_rx_bytes(hq)
                elapsed = max(current_time - previous_time, 0.001)
                delivered_rate = max(0, current_bytes - previous_bytes) * 8.0 / (1_000_000.0 * elapsed)
                cpu_user, cpu_system, cpu_idle, cpu_used, cpu_iowait = calculate_cpu_percentages(
                    previous_cpu,
                    current_cpu,
                )
                timestamp = datetime.now().isoformat(sep=" ", timespec="microseconds")
                offered_load = sum(flow[1] for flow in phase["flows"])

                cpu_rows.append(
                    {
                        "timestamp": timestamp,
                        "cpu_user": round(cpu_user, 4),
                        "cpu_system": round(cpu_system, 4),
                        "cpu_idle": round(cpu_idle, 4),
                        "cpu_used": round(cpu_used, 4),
                        "scenario": config["research_name"],
                        "sample": global_sample,
                        "simulated_time": phase["name"],
                        "cpu_iowait": round(cpu_iowait, 4),
                        "cpu_workers": phase["cpu_workers"],
                    }
                )

                for counter in read_ovs_counters():
                    network_rows.append(
                        {
                            "timestamp": timestamp,
                            **counter,
                            "sample": global_sample,
                            "simulated_time": phase["name"],
                            "scenario": config["research_name"],
                            "source_site": source_name,
                            "latency_ms": latency_ms,
                            "packet_loss_pct": round(packet_loss_pct, 4),
                            "delivered_rate_mbps": round(delivered_rate, 6),
                            "offered_load_mbps": offered_load,
                            "active_flows": len(phase["flows"]),
                            "configured_core_bw_mbps": config["core_bw_mbps"],
                            "configured_branch_bw_mbps": config["branch_bw_mbps"],
                            "configured_wan_delay_ms": config["wan_delay_ms"],
                            "configured_wan_loss_pct": config["wan_loss_pct"],
                        }
                    )

                previous_cpu = current_cpu
                previous_bytes = current_bytes
                previous_time = current_time
                global_sample += 1

                if phase_sample == 0 or (phase_sample + 1) % 10 == 0:
                    latency_text = "unavailable" if latency_ms == "" else f"{latency_ms:.2f} ms"
                    print(
                        f"  sample {global_sample:3d}: CPU {cpu_used:5.1f}% | "
                        f"rate {delivered_rate:6.2f} Mbps | latency {latency_text}"
                    )

            stopped_processes = count_stopped_processes(active_processes)
            if stopped_processes > 0:
                raise RuntimeError(
                    f"{stopped_processes} traffic process(es) ended before "
                    f'the {phase["name"]} samples were complete.'
                )
            stop_processes(active_processes)
            active_processes = []
            if phase_index < len(PHASES) - 1:
                print(f"  cooling down for {PHASE_COOLDOWN_SECONDS} unrecorded seconds")
                time.sleep(PHASE_COOLDOWN_SECONDS)

    finally:
        stop_processes(active_processes)
        network.stop()
        cleanup()

    return cpu_rows, network_rows


# Check software, collect the scenario and save the two research files.
def main():
    parser = argparse.ArgumentParser(
        description="Capture an ANDM-inspired ONOS and Mininet scenario."
    )
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--duration-per-period", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise PermissionError("Run this script with sudo python3.")
    if args.duration_per_period < 5:
        raise ValueError("Use at least 5 samples per simulated time period.")
    for program in ["iperf3", "ovs-vsctl"]:
        if shutil.which(program) is None:
            raise FileNotFoundError(f"Required program is missing: {program}")

    check_controller(args.controller_ip, args.controller_port)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SCENARIOS[args.scenario]
    cpu_path = args.output_dir / f"mininet_{args.scenario}_cpu.csv"
    network_path = args.output_dir / f"mininet_{args.scenario}_network_raw.csv"
    check_output_paths([cpu_path, network_path], args.force)

    print("\nANDM-inspired Mininet capture")
    print(f"Scenario: {config['research_name']}")
    print(f"Periods: Morning, Afternoon and Night")
    print(f"Samples: {args.duration_per_period * len(PHASES)}")
    print("Boundary: this is a controlled testbed, not real ANDM telemetry.")

    cleanup()
    cpu_rows, network_rows = capture_scenario(args, config)

    cpu_fields = [
        "timestamp",
        "cpu_user",
        "cpu_system",
        "cpu_idle",
        "cpu_used",
        "scenario",
        "sample",
        "simulated_time",
        "cpu_iowait",
        "cpu_workers",
    ]
    network_fields = [
        "timestamp",
        "switch",
        "port",
        "rx_packets",
        "tx_packets",
        "rx_bytes",
        "tx_bytes",
        "sample",
        "simulated_time",
        "scenario",
        "source_site",
        "latency_ms",
        "packet_loss_pct",
        "delivered_rate_mbps",
        "offered_load_mbps",
        "active_flows",
        "configured_core_bw_mbps",
        "configured_branch_bw_mbps",
        "configured_wan_delay_ms",
        "configured_wan_loss_pct",
    ]

    write_csv_safely(cpu_path, cpu_fields, cpu_rows)
    write_csv_safely(network_path, network_fields, network_rows)
    restore_file_owner([args.output_dir, cpu_path, network_path])

    print("\nCapture complete.")
    print(f"CPU rows: {len(cpu_rows)} -> {cpu_path}")
    print(f"Network rows: {len(network_rows)} -> {network_path}")
    print_capture_summary(cpu_rows, network_rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCapture stopped by the user. No final CSV files were written.")
        sys.exit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
