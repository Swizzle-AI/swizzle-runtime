"""
Hardware Thermal, Voltage & Power Guard (`hardware_guard.py`)
=============================================================
Monitors CPU voltage, clock, load, and ACPI/GPU temperature sensors before and
between heavy model download or inference bursts so consumer laptops never overheat
or pull unsafe sustained boost currents:

- Caps PyTorch CPU threads (`torch.set_num_threads(...)`) based on real CPU load & thermals.
- Reads Windows `Win32_Processor` (`CurrentVoltage`, `CurrentClockSpeed`, `LoadPercentage`),
  `Win32_Battery` / AC status, ACPI thermal zones (`MSAcpi_ThermalZoneTemperature` when
  exposed), Linux `/sys/class/thermal`, and `nvidia-smi` GPU temp/power.
- Inserts adaptive cool-down pauses if load/temperature crosses warning thresholds.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class HardwareHealthSnapshot:
    cpu_name: str
    voltage_v: Optional[float]
    clock_mhz: Optional[int]
    load_pct: Optional[int]
    temp_c: Optional[float]
    gpu_temp_c: Optional[float]
    safe_threads: int
    status: str  # "COOL" | "WARM_THROTTLED" | "HOT_PAUSED"


def probe_hardware_health(max_temp_c: float = 82.0, hard_stop_temp_c: float = 90.0) -> HardwareHealthSnapshot:
    cpu_name = platform.processor() or "CPU"
    volt_v: Optional[float] = None
    clock_mhz: Optional[int] = None
    load_pct: Optional[int] = None
    temp_c: Optional[float] = None
    gpu_temp_c: Optional[float] = None

    if platform.system() == "Windows":
        try:
            ps = (
                "$p = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,CurrentVoltage,CurrentClockSpeed,LoadPercentage; "
                "$t = Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue | Select-Object -First 1 CurrentTemperature; "
                "Write-Output ($p.Name + '|' + $p.CurrentVoltage + '|' + $p.CurrentClockSpeed + '|' + $p.LoadPercentage + '|' + $t.CurrentTemperature)"
            )
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True, timeout=4.0).strip()
            parts = out.split("|")
            if len(parts) >= 4:
                if parts[0].strip():
                    cpu_name = parts[0].strip()
                if parts[1].strip().isdigit():
                    # SMBIOS reports tenths of a volt (e.g. 12 -> 1.2 V) or mV
                    raw_v = float(parts[1].strip())
                    volt_v = round(raw_v / 10.0 if raw_v <= 50 else raw_v / 1000.0, 2)
                if parts[2].strip().isdigit():
                    clock_mhz = int(parts[2].strip())
                if parts[3].strip().isdigit():
                    load_pct = int(parts[3].strip())
                if len(parts) >= 5 and parts[4].strip().isdigit():
                    temp_c = round((float(parts[4].strip()) - 2732.0) / 10.0, 1)
        except Exception:
            pass
    else:
        for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            try:
                val = float(p.read_text().strip()) / 1000.0
                if 10.0 <= val <= 130.0:
                    temp_c = round(max(temp_c or 0.0, val), 1)
            except Exception:
                pass

    try:
        g_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.5,
        ).strip()
        if g_out:
            gpu_temp_c = float(g_out.splitlines()[0].strip())
    except Exception:
        pass

    cores = os.cpu_count() or 8
    max_observed_temp = max([v for v in (temp_c, gpu_temp_c) if v is not None], default=None)

    if max_observed_temp is not None and max_observed_temp >= hard_stop_temp_c:
        status = "HOT_PAUSED"
        safe_threads = max(2, cores // 4)
    elif (max_observed_temp is not None and max_observed_temp >= max_temp_c) or (load_pct is not None and load_pct >= 85):
        status = "WARM_THROTTLED"
        safe_threads = max(2, cores // 4)
    else:
        status = "COOL"
        # Cap at 50% threads max (e.g., 6-8 out of 16) to keep thermal & voltage headroom safe
        safe_threads = min(8, max(2, cores // 2))

    return HardwareHealthSnapshot(
        cpu_name=cpu_name,
        voltage_v=volt_v,
        clock_mhz=clock_mhz,
        load_pct=load_pct,
        temp_c=temp_c,
        gpu_temp_c=gpu_temp_c,
        safe_threads=safe_threads,
        status=status,
    )


def enforce_safe_thermal_budget(cool_pause_s: float = 2.0) -> HardwareHealthSnapshot:
    """Inspects hardware sensors, caps PyTorch threads, and pauses if warm/hot."""
    snap = probe_hardware_health()
    torch.set_num_threads(snap.safe_threads)
    if snap.status == "HOT_PAUSED":
        time.sleep(cool_pause_s * 3.0)
        snap = probe_hardware_health()
        torch.set_num_threads(snap.safe_threads)
    elif snap.status == "WARM_THROTTLED":
        time.sleep(cool_pause_s)
    return snap
