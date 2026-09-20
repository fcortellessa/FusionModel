#!/usr/bin/env python3
"""
Generate config variants from a template config and a YAML variant spec.

The spec lets you choose exactly which config paths vary without changing the
script itself.


If both `grid` and `variants` are given, every variant is generated once per
grid point (a full cross product), e.g. 3 variants x 10 grid combos = 30
files, each named `<variant_name>__<grid_overrides>__<hash>.yml`. If only one
of `grid`/`variants` is given, that set alone determines the output (grid
combos get an auto-generated name from the varying grid keys; explicit
variants use their own `name`).

By default the cross product is ordered variant-major: all grid combos for
one variant are emitted before moving to the next variant. Pass
`--grid-major` to flip this to grid-major order: all variants for one grid
combo are emitted before moving to the next grid combo. This matters when a
grid dimension is expensive to change between runs — e.g. `training.seed`
triggers a new data split (`--force_preprocess`) — since it lets you launch
generated configs in file order and only pay that cost once per seed instead
of once per variant.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml


DEFAULT_SPEC_PATH = Path("configs/gated_2d.yml")
DEFAULT_TEMPLATE_CONFIG = Path("configs/fusions_schedules_variants_spec.yml")
DEFAULT_OUTPUT_DIR = Path("configs/generated/fusions_schedules_variants")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def normalize_scalar_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def set_nested_value(config: Dict[str, Any], dotted_path: str, value: Any) -> None:
    current = config
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid config path: {dotted_path!r}")

    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            raise TypeError(
                f"Cannot set {dotted_path!r}: {part!r} already exists and is not a mapping"
            )
        current = current[part]

    current[parts[-1]] = value


def apply_overrides(template: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    config = deepcopy(dict(template))
    for dotted_path, value in overrides.items():
        set_nested_value(config, dotted_path, value)
    return config


def stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "none"
    if isinstance(value, float):
        return format(value, "g")
    if isinstance(value, (int, str)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "-".join(stringify_value(item) for item in value)
    if isinstance(value, dict):
        return "obj"
    return str(value)


def sanitize_filename(name: str, max_length: int = 180) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "variant"
    if len(cleaned) <= max_length:
        return cleaned

    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    prefix = cleaned[: max_length - 9].rstrip("._-")
    if not prefix:
        prefix = "variant"
    return f"{prefix}_{digest}"


def build_override_name_parts(overrides: Mapping[str, Any]) -> List[str]:
    parts: List[str] = []
    for dotted_path, value in overrides.items():
        leaf_name = dotted_path.split(".")[-1]
        parts.append(f"{leaf_name}-{stringify_value(value)}")
    return parts


def build_auto_variant_name(overrides: Mapping[str, Any]) -> str:
    if not overrides:
        return "variant"

    parts = build_override_name_parts(overrides)
    digest_input = repr(sorted(overrides.items())).encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()[:6]
    return sanitize_filename("__".join(parts) + f"__{digest}")


def expand_grid(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not grid:
        return [dict()]

    keys = list(grid.keys())
    value_lists = [normalize_scalar_list(grid[key]) for key in keys]
    combinations: List[Dict[str, Any]] = []

    for values in itertools.product(*value_lists):
        combinations.append({key: value for key, value in zip(keys, values)})

    return combinations


def load_variant_entries(spec: Mapping[str, Any], grid_major: bool = False) -> List[Dict[str, Any]]:
    fixed_overrides = spec.get("fixed_overrides", {})
    if fixed_overrides and not isinstance(fixed_overrides, Mapping):
        raise TypeError("spec.fixed_overrides must be a mapping")

    grid = spec.get("grid", {})
    if grid and not isinstance(grid, Mapping):
        raise TypeError("spec.grid must be a mapping")

    explicit_variants = spec.get("variants", [])
    if explicit_variants and not isinstance(explicit_variants, Sequence):
        raise TypeError("spec.variants must be a sequence")

    prefix = spec.get("name_prefix")
    if prefix is not None and not isinstance(prefix, str):
        raise TypeError("spec.name_prefix must be a string when provided")

    entries: List[Dict[str, Any]] = []
    varying_grid_keys = [key for key, values in grid.items() if len(normalize_scalar_list(values)) > 1]
    grid_combos = expand_grid(grid) if grid else [dict()]

    if explicit_variants:
        # Cross product: every explicit variant is generated once per grid
        # point, so seed/grid sweeps apply to LR-schedule-style variants too
        # instead of producing two disjoint, independently-named sets.
        #
        # Iteration order is controlled by `grid_major`:
        #   - variant-major (default): all grid combos for one variant are
        #     emitted before moving to the next variant.
        #   - grid-major: all variants for one grid combo (e.g. one data
        #     split seed) are emitted before moving to the next grid combo.
        #     Useful when a grid dimension (like `training.seed`) is
        #     expensive to change (triggers a new data split), so runs that
        #     share a seed should be launched back-to-back.
        parsed_variants: List[Dict[str, Any]] = []
        for variant in explicit_variants:
            if not isinstance(variant, Mapping):
                raise TypeError("Each entry in spec.variants must be a mapping")
            variant_overrides = variant.get("overrides", {})
            if not isinstance(variant_overrides, Mapping):
                raise TypeError("Each spec.variants[*].overrides must be a mapping")
            variant_name = variant.get("name")
            if variant_name is not None and not isinstance(variant_name, str):
                raise TypeError("Each spec.variants[*].name must be a string when provided")
            parsed_variants.append({"name": variant_name, "overrides": variant_overrides})

        def make_entry(variant: Mapping[str, Any], combo: Mapping[str, Any]) -> Dict[str, Any]:
            overrides = dict(fixed_overrides)
            overrides.update(combo)
            overrides.update(variant["overrides"])
            grid_name_overrides = {key: combo[key] for key in varying_grid_keys}
            return {
                "prefix": prefix,
                "name": variant["name"],
                "overrides": overrides,
                "name_overrides": variant["overrides"],
                "grid_name_overrides": grid_name_overrides,
            }

        if grid_major:
            for combo in grid_combos:
                for variant in parsed_variants:
                    entries.append(make_entry(variant, combo))
        else:
            for variant in parsed_variants:
                for combo in grid_combos:
                    entries.append(make_entry(variant, combo))
    elif grid:
        for combo in grid_combos:
            overrides = dict(fixed_overrides)
            overrides.update(combo)
            name_overrides = {key: combo[key] for key in varying_grid_keys}
            entries.append({"prefix": prefix, "overrides": overrides, "name_overrides": name_overrides})
    else:
        entries.append({"prefix": prefix, "overrides": dict(fixed_overrides), "name_overrides": dict(fixed_overrides)})

    return entries


def build_variant_filename(variant: Mapping[str, Any]) -> str:
    name_overrides = dict(variant.get("name_overrides") or variant["overrides"])
    grid_name_overrides = variant.get("grid_name_overrides") or {}
    combined_name_overrides = {**grid_name_overrides, **name_overrides}

    explicit_name = variant.get("name")
    if explicit_name:
        parts = [explicit_name] + build_override_name_parts(grid_name_overrides)
        digest_input = repr(sorted(combined_name_overrides.items())).encode("utf-8")
        digest = hashlib.sha1(digest_input).hexdigest()[:6]
        variant_name = sanitize_filename("__".join(parts) + f"__{digest}")
    else:
        variant_name = build_auto_variant_name(combined_name_overrides)

    prefix = variant.get("prefix")
    if prefix:
        variant_name = f"{prefix}_{variant_name}"
    return sanitize_filename(variant_name)


def write_config_variants(
    template_config_path: Path,
    spec_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    grid_major: bool = False,
) -> List[Path]:
    template_config = load_yaml(template_config_path)
    spec = load_yaml(spec_path)

    variants = load_variant_entries(spec, grid_major=grid_major)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: List[Path] = []
    for index, variant in enumerate(variants, start=1):
        overrides = variant["overrides"]
        config = apply_overrides(template_config, overrides)

        variant_filename = f"{index:03d}_{build_variant_filename(variant)}.yml"
        output_path = output_dir / variant_filename
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )

        if not dry_run:
            with output_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)

        written_paths.append(output_path)

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate config variants from a template YAML and a variant spec"
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=DEFAULT_TEMPLATE_CONFIG,
        help="Template YAML config to copy and modify",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="YAML file that defines fixed_overrides, grid, and explicit variants",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated config files will be written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )
    parser.add_argument(
        "--grid-major",
        action="store_true",
        help=(
            "Order output so every explicit variant is generated for one grid "
            "combo before moving to the next grid combo (default: every grid "
            "combo is generated for one variant before moving to the next "
            "variant). Use this when a grid dimension is expensive to change "
            "across runs (e.g. training.seed triggers a new data split), so "
            "runs sharing that value are grouped together."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.template_config.exists():
        raise FileNotFoundError(f"Template config not found: {args.template_config}")
    if not args.spec.exists():
        raise FileNotFoundError(f"Spec file not found: {args.spec}")

    spec = load_yaml(args.spec)
    output_dir = args.output_dir
    if output_dir == DEFAULT_OUTPUT_DIR and isinstance(spec.get("output_dir"), str):
        output_dir = Path(spec["output_dir"])

    written_paths = write_config_variants(
        template_config_path=args.template_config,
        spec_path=args.spec,
        output_dir=output_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        grid_major=args.grid_major,
    )

    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {len(written_paths)} config files to {output_dir}")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()