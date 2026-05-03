"""MIF (Memory Interchange Format) exporter/importer for NeuralMem."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neuralmem.core.types import Memory


class MIFExporter:
    """Export and import NeuralMem memories in MIF (Memory Interchange Format)."""

    def __init__(self, version: str = "0.2") -> None:
        self.version = version

    def export_memory(
        self, memory: Memory, relations: list[dict] | None = None
    ) -> dict[str, Any]:
        """Export a single Memory to MIF dict format.

        Args:
            memory: The Memory object to export.
            relations: Optional list of relation dicts (currently unused placeholder).

        Returns:
            A dict representing the memory in MIF format.
        """
        mif_entry: dict[str, Any] = {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type.value
            if hasattr(memory.memory_type, "value")
            else str(memory.memory_type),
            "importance": memory.importance,
            "confidence": 1.0,  # default; no confidence field on Memory yet
            "source_refs": [memory.source] if memory.source else [],
            "provenance": {
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
                "extractor": "rule-based",
                "embedder": "local",
            },
            "tags": list(memory.tags),
            "user_id": memory.user_id or "",
            "supersedes": list(memory.supersedes),
            "contradicts": [],  # not tracked on Memory model yet
            "validity": {
                "valid_from": memory.created_at.isoformat(),
                "valid_until": memory.expires_at.isoformat()
                if memory.expires_at
                else None,
            },
            "metadata": {},
        }

        if relations is not None:
            mif_entry["relations"] = relations

        return mif_entry

    def export_json(
        self,
        memories: list[Memory],
        output_path: str | None = None,
    ) -> str:
        """Export a list of memories as a MIF JSON string.

        Args:
            memories: List of Memory objects.
            output_path: Optional file path to write the JSON to.

        Returns:
            The JSON string.
        """
        payload = {
            "version": self.version,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(memories),
            "memories": [self.export_memory(m) for m in memories],
        }

        json_str = json.dumps(payload, indent=2, ensure_ascii=False)

        if output_path is not None:
            Path(output_path).write_text(json_str, encoding="utf-8")

        return json_str

    def export_markdown(
        self,
        memories: list[Memory],
        output_path: str | None = None,
    ) -> str:
        """Export memories as readable Markdown.

        Args:
            memories: List of Memory objects.
            output_path: Optional file path to write the Markdown to.

        Returns:
            The Markdown string.
        """
        lines: list[str] = []
        lines.append("# NeuralMem Memory Export")
        lines.append("")
        lines.append(f"**Version:** {self.version}  ")
        lines.append(f"**Exported at:** {datetime.now(timezone.utc).isoformat()}  ")
        lines.append(f"**Count:** {len(memories)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for idx, mem in enumerate(memories, 1):
            lines.append(f"## Memory {idx}")
            lines.append("")
            lines.append(f"- **ID:** `{mem.id}`")
            lines.append(f"- **Type:** {mem.memory_type.value}")
            lines.append(f"- **Importance:** {mem.importance}")
            lines.append(f"- **Created:** {mem.created_at.isoformat()}")
            lines.append(f"- **Updated:** {mem.updated_at.isoformat()}")
            if mem.user_id:
                lines.append(f"- **User ID:** {mem.user_id}")
            if mem.tags:
                lines.append(f"- **Tags:** {', '.join(mem.tags)}")
            if mem.source:
                lines.append(f"- **Source:** {mem.source}")
            lines.append("")
            lines.append("### Content")
            lines.append("")
            lines.append(mem.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        md_str = "\n".join(lines)

        if output_path is not None:
            Path(output_path).write_text(md_str, encoding="utf-8")

        return md_str

    def import_json(self, json_str: str) -> list[dict[str, Any]]:
        """Parse MIF JSON string back to a list of dicts with validation.

        Args:
            json_str: A MIF JSON string.

        Returns:
            List of memory dicts.

        Raises:
            ValueError: If the JSON is invalid.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("MIF JSON root must be an object")

        memories = data.get("memories", [])

        errors_all: list[str] = []
        for entry in memories:
            is_valid, errors = self.validate(entry)
            if not is_valid:
                errors_all.extend(errors)

        if errors_all:
            raise ValueError(
                f"MIF validation failed with {len(errors_all)} error(s): "
                + "; ".join(errors_all)
            )

        return memories

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a single MIF entry.

        Args:
            data: A dict representing one MIF memory entry.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        # Required top-level fields
        if not data.get("id"):
            errors.append("Missing required field: id")
        if not data.get("content"):
            errors.append("Missing required field: content")

        # Type checks
        if "importance" in data:
            imp = data["importance"]
            if not isinstance(imp, (int, float)):
                errors.append("importance must be a number")
            elif not (0.0 <= imp <= 1.0):
                errors.append("importance must be between 0.0 and 1.0")

        if "confidence" in data:
            conf = data["confidence"]
            if not isinstance(conf, (int, float)):
                errors.append("confidence must be a number")

        if "source_refs" in data and not isinstance(data["source_refs"], list):
            errors.append("source_refs must be a list")

        if "tags" in data and not isinstance(data["tags"], list):
            errors.append("tags must be a list")

        if "provenance" in data:
            prov = data["provenance"]
            if not isinstance(prov, dict):
                errors.append("provenance must be a dict")

        if "supersedes" in data and not isinstance(data["supersedes"], list):
            errors.append("supersedes must be a list")

        if "contradicts" in data and not isinstance(data["contradicts"], list):
            errors.append("contradicts must be a list")

        if "validity" in data:
            val = data["validity"]
            if not isinstance(val, dict):
                errors.append("validity must be a dict")

        if "metadata" in data and not isinstance(data["metadata"], dict):
            errors.append("metadata must be a dict")

        is_valid = len(errors) == 0
        return is_valid, errors
