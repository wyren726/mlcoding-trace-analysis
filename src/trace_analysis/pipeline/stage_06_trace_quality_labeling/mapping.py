"""Read the user-owned CSV without relabeling its anchored atomic abilities."""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Subrequirement:
    requirement_name: str
    subrequirement_name: str
    source_row: int
    definition: str
    inputs: str
    outputs: str
    boundary: str
    atomic_capabilities: tuple[str, ...]
    stage_number: str
    stage_name: str
    consideration: str

    def prompt_record(self) -> dict[str, Any]:
        # Atoms are deliberately not a model decision or a generated taxonomy ID.
        return {
            "taxonomy_row": self.source_row,
            "requirement_name": self.requirement_name,
            "subrequirement_name": self.subrequirement_name,
            "definition": self.definition,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "boundary": self.boundary,
        }


@dataclass
class CapabilityTaxonomy:
    path: Path
    sha256: str
    entries: dict[tuple[str, str], Subrequirement]
    skipped_rows: list[int]
    warnings: list[dict[str, Any]]

    @property
    def requirements(self) -> list[str]:
        return list(dict.fromkeys(key[0] for key in self.entries))

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "sha256": self.sha256,
            "requirement_count": len(self.requirements),
            "subrequirement_count": len(self.entries),
            "skipped_rows": self.skipped_rows, "warnings": self.warnings,
        }

    def prompt_records(self) -> list[dict[str, Any]]:
        return [entry.prompt_record() for entry in self.entries.values()]

    def resolve_rows(self, selections: Any) -> list[dict[str, Any]]:
        """Model boundary: only existing integer CSV rows, never free-text names."""
        if not isinstance(selections, list):
            raise ValueError("capability_mapping must be a list of taxonomy_row objects")
        valid_rows = {entry.source_row for entry in self.entries.values()}
        for choice in selections:
            if not isinstance(choice, dict) or set(choice) != {"taxonomy_row"}:
                raise ValueError("Model mapping must contain only taxonomy_row")
            if type(choice["taxonomy_row"]) is not int or choice["taxonomy_row"] not in valid_rows:
                raise ValueError(f"Invalid taxonomy_row: {choice['taxonomy_row']!r}")
        return self.resolve(selections)

    def resolve(self, selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate exact parent/child choices; attach atoms by table lookup only.

        A null child means only the parent was identified. An empty selection
        does not mean 'no requirements'; the calling labeler must track unknown.
        """
        if not isinstance(selections, list):
            raise ValueError("capability_mapping must be a list")
        result = []
        seen = set()
        for choice in selections:
            if not isinstance(choice, dict):
                raise ValueError("Each capability selection must be an object")
            if any(key.startswith("atomic_") for key in choice):
                raise ValueError("Atomic abilities must be derived, not model supplied")
            if "taxonomy_row" in choice:
                row = choice["taxonomy_row"]
                if type(row) is not int:
                    raise ValueError("taxonomy_row must be an integer")
                matches = [entry for entry in self.entries.values() if entry.source_row == row]
                if len(matches) != 1:
                    raise ValueError(f"Unknown taxonomy row: {row!r}")
                entry = matches[0]
                key = (entry.requirement_name, entry.subrequirement_name)
                if key in seen:
                    continue
                seen.add(key)
                result.append({"requirement_name": entry.requirement_name, "subrequirement_name": entry.subrequirement_name, "atomic_capabilities": list(entry.atomic_capabilities), "mapping_status": "mapped", "taxonomy_row": row})
                continue
            raw_parent = choice.get("requirement_name")
            parent = self._canonical_name(raw_parent, self.requirements)
            if parent is None:
                raise ValueError(f"Unknown requirement: {raw_parent!r}")
            child = choice.get("subrequirement_name")
            if child is not None:
                raw_child = child
                child = self._canonical_name(child, [key[1] for key in self.entries if key[0] == parent])
                if child is None:
                    raise ValueError(f"Unknown subrequirement: {raw_child!r}")
            if child is not None and not isinstance(child, str):
                raise ValueError("subrequirement_name must be an exact name or null")
            key = (parent, child)
            if key in seen:
                continue
            seen.add(key)
            if child is None:
                result.append({
                    "requirement_name": parent, "subrequirement_name": None,
                    "atomic_capabilities": None, "mapping_status": "partial",
                })
                continue
            entry = self.entries.get(key)
            if entry is None:
                raise ValueError(f"Invalid requirement/subrequirement pair: {key!r}")
            result.append({
                "requirement_name": parent, "subrequirement_name": child,
                "atomic_capabilities": list(entry.atomic_capabilities),
                "mapping_status": "mapped", "taxonomy_row": entry.source_row,
            })
        return result

    @staticmethod
    def _canonical_name(value: Any, options: list[str]) -> str | None:
        if not isinstance(value, str):
            return value if value is None else None
        if value in options:
            return value
        compact = re.sub(r"\s+", "", value)
        matches = [option for option in options if re.sub(r"\s+", "", option) == compact]
        return matches[0] if len(matches) == 1 else None


def load_taxonomy(path: Path) -> CapabilityTaxonomy:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    required = {"需求", "子需求", "定义", "边界", "原子能力"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing taxonomy columns: {sorted(required - set(reader.fieldnames or []))}")
    entries: dict[tuple[str, str], Subrequirement] = {}
    warnings: list[dict[str, Any]] = []
    skipped = []
    parent = ""
    for row_number, row in enumerate(reader, 2):
        def cell(name: str) -> str:
            return (row.get(name) or "").strip()

        if cell("需求"):
            parent = cell("需求")
        child = cell("子需求")
        if not child:
            if any(cell(key) for key in ("定义", "原子能力", "输入", "输出", "边界")):
                raise ValueError(f"Taxonomy row {row_number}: content without a subrequirement")
            skipped.append(row_number)
            continue
        if not parent:
            raise ValueError(f"Taxonomy row {row_number}: no parent requirement")
        key = (parent, child)
        if key in entries:
            raise ValueError(f"Duplicate taxonomy pair at row {row_number}: {key!r}")
        atoms = tuple(dict.fromkeys(
            part.strip() for part in re.split(r"[;；\r\n]+", cell("原子能力")) if part.strip()
        ))
        if not atoms:
            raise ValueError(f"Taxonomy row {row_number}: missing anchored atomic abilities")
        if not cell("定义"):
            raise ValueError(f"Taxonomy row {row_number}: missing definition")
        missing = [name for name in ("输入", "输出", "边界", "阶段编号", "阶段") if not cell(name)]
        if missing:
            warnings.append({"row": row_number, "missing_fields": missing})
        entries[key] = Subrequirement(
            parent, child, row_number, cell("定义"), cell("输入"), cell("输出"),
            cell("边界"), atoms, cell("阶段编号"), cell("阶段"), cell("是否考虑"),
        )
    if not entries:
        raise ValueError("Taxonomy has no subrequirements")
    return CapabilityTaxonomy(path, hashlib.sha256(raw).hexdigest(), entries, skipped, warnings)
