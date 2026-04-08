from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


_FILE_RE = re.compile(r"^S11_(Mag|Pha)_(.+)\.csv$", re.IGNORECASE)
_DATA_MODE_PATTERNS = {
    "all": None,
    "raw": re.compile(r"^SVR_CYCLE_\d+$", re.IGNORECASE),
    "socip0p1": re.compile(r"^SVR_SOCIP0p1_CYCLE_\d+$", re.IGNORECASE),
    "socip0p5": re.compile(r"^SVR_SOCIP0p5_CYCLE_\d+$", re.IGNORECASE),
    "socip1p0": re.compile(r"^SVR_SOCIP1p0_CYCLE_\d+$", re.IGNORECASE),
}


@dataclass(frozen=True)
class S11Record:
    day: str
    series_name: str
    mag_path: Path | None = None
    pha_path: Path | None = None

    @property
    def record_id(self) -> str:
        return f"{self.day}/{self.series_name}"

    @property
    def has_phase(self) -> bool:
        return self.pha_path is not None


def discover_s11_records(
    root: str | Path,
    name_contains: str | None = None,
    require_phase: bool = False,
    data_mode: str = "all",
) -> list[S11Record]:
    root = Path(root)
    mode_key = data_mode.lower()
    if mode_key not in _DATA_MODE_PATTERNS:
        raise ValueError(f"Unsupported data_mode={data_mode!r}. Expected one of: {', '.join(_DATA_MODE_PATTERNS)}")
    mode_pattern = _DATA_MODE_PATTERNS[mode_key]

    grouped: dict[tuple[str, str], dict[str, Path | str | None]] = {}

    for csv_path in root.rglob("*.csv"):
        match = _FILE_RE.match(csv_path.name)
        if not match:
            continue
        modality, series_name = match.groups()
        key = (csv_path.parent.name, series_name)
        bucket = grouped.setdefault(
            key,
            {
                "day": csv_path.parent.name,
                "series_name": series_name,
                "mag_path": None,
                "pha_path": None,
            },
        )
        if modality.lower() == "mag":
            bucket["mag_path"] = csv_path
        else:
            bucket["pha_path"] = csv_path

    records = []
    for entry in grouped.values():
        if entry["mag_path"] is None:
            continue
        if require_phase and entry["pha_path"] is None:
            continue
        series_name = str(entry["series_name"])
        if mode_pattern is not None and not mode_pattern.match(series_name):
            continue
        if name_contains and name_contains.lower() not in series_name.lower():
            continue
        records.append(
            S11Record(
                day=str(entry["day"]),
                series_name=series_name,
                mag_path=Path(entry["mag_path"]),
                pha_path=Path(entry["pha_path"] ) if entry["pha_path"] else None,
            )
        )

    records.sort(key=lambda item: (item.day, item.series_name))
    return records


def record_ids(records: Iterable[S11Record]) -> list[str]:
    return [record.record_id for record in records]
