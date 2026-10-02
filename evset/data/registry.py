"""File registry for the three XJTU-DV subsets.

Data root: environment variable EVSET_DATA (default: <repository>/data/XJTU-DV) with the sub-folders
Rotor/, Pump/ and Beam/ holding the .dat recordings (see docs/DATA.md). The laser-vibrometer text files of the
Beam subset are read from EVSET_LDV (default: <data root>/Beam_LDV).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("EVSET_DATA", os.path.join(REPO, "data", "XJTU-DV"))

ROTOR_CLASSES = ["Healthy", "Inner", "Outer", "Ball"]
PUMP_CLASSES = ["Healthy", "Inner", "Outer", "Ball", "Cav", "Misalignment"]
PUMP_CODE = {"H": "Healthy", "I": "Inner", "O": "Outer", "b": "Ball", "cav": "Cav", "MA": "Misalignment"}


@dataclass
class Recording:
    subset: str          # rotor / pump / beam
    path: str
    label: str           # class name
    view: str            # angle1..3 (rotor) / -30,0,30 (pump) / set1..3 (beam)
    speed: float         # nominal shaft frequency in Hz (rotor: rpm/60; pump: drive Hz)
    extra: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return f"{self.view}@{self.speed:g}Hz"

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


def rotor_files(root: str = ROOT) -> list[Recording]:
    out = []
    d = os.path.join(root, "Rotor")
    for fn in sorted(os.listdir(d)):
        m = re.match(r"(angle\d)_(Healthy|Inner|Outer|Ball)_(\d+)\.dat", fn)
        if not m:
            continue
        rpm = int(m.group(3))
        out.append(Recording("rotor", os.path.join(d, fn), m.group(2), m.group(1), rpm / 60.0, dict(rpm=rpm)))
    return out


def pump_entries(zip_path: str | None = None) -> list[dict]:
    """List the Pump recordings inside the original XJTU-DV-Pump.zip archive (parsed metadata, no extraction)."""
    import zipfile
    zip_path = zip_path or os.path.join(ROOT, "XJTU-DV-Pump.zip")
    z = zipfile.ZipFile(zip_path)
    out = []
    for i in z.infolist():
        if not i.filename.endswith(".dat"):
            continue
        fn = os.path.basename(i.filename)
        m = re.match(r"(\d+)H[Zz]_(H|I|MA|O|b|cav)_(\d+)_(-?\d+)\.dat", fn)
        if not m:
            raise ValueError(fn)
        out.append(dict(member=i.filename, name=fn[:-4], size=i.file_size, speed=float(m.group(1)),
                        label=PUMP_CODE[m.group(2)], setting=int(m.group(3)), view=m.group(4)))
    return out


def beam_files(root: str = ROOT) -> list[Recording]:
    out = []
    d = os.path.join(root, "Beam")
    ldv_dir = os.environ.get("EVSET_LDV", os.path.join(root, "Beam_LDV"))
    for fn in sorted(os.listdir(d)):
        m = re.match(r"(On|Off)_set(\d)_trai?l?(\d)\.dat", fn)
        if not m:
            continue
        ldv = os.path.join(ldv_dir, f"{m.group(1)}_set{m.group(2)}_trail{m.group(3)}.txt")
        out.append(Recording("beam", os.path.join(d, fn), m.group(1), f"set{m.group(2)}", 0.0,
                             dict(trial=int(m.group(3)), ldv=os.path.normpath(ldv))))
    return out


def pump_files(root: str = ROOT) -> list[Recording]:
    out = []
    d = os.path.join(root, "Pump")
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        m = re.match(r"(\d+)H[Zz]_(H|I|MA|O|b|cav)_(\d+)_(-?\d+)\.dat", fn)
        if not m:
            continue
        out.append(Recording("pump", os.path.join(d, fn), PUMP_CODE[m.group(2)], m.group(4), float(m.group(1)),
                             dict(setting=int(m.group(3)))))
    return out
