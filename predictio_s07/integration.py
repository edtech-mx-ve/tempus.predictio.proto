"""Carga y validación de artefactos S01–S06; solo lectura, sin llamadas externas."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

SYMBOLS = ("JNJ", "MSFT", "JPM", "WMT", "XOM")
HORIZONS = (1, 5)
REPORTS = {
    2: "diagnosticos.json",
    3: "estructura_temporal.json",
    4: "modelos_s04.json",
    5: "evaluacion_s05.json",
    6: "calibracion_s06.json",
}
SECTIONS = (
    "01 · Normalidad",
    "02 · Valores atípicos",
    "03 · Medias móviles",
    "04 · Descomposición temporal",
    "05 · Regresión lineal y tendencia",
    "06 · Fourier y periodograma",
    "07 · Modelos predictivos",
    "08 · Evaluación e incertidumbre",
)

class ValidationError(ValueError):
    """Archivo o informe no satisface el contrato de lectura."""


def safe_path(root: Path, relative: str) -> Path:
    """Impide traversal y escapes del directorio local de resultados."""
    if not isinstance(relative, str) or not relative:
        raise ValidationError("Ruta relativa inválida")
    base = Path(root).resolve(strict=True)
    path = (base / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(base):
        raise ValidationError("Ruta fuera de resultados o archivo inválido")
    return path


def read_json(root: Path, relative: str) -> dict[str, Any]:
    path = safe_path(root, relative)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValidationError("El JSON no es un objeto")
    return value


def read_csv(root: Path, relative: str, columns: tuple[str, ...],
             min_rows: int = 1) -> pd.DataFrame:
    path = safe_path(root, relative)
    result = pd.read_csv(path)
    if not set(columns).issubset(result.columns) or len(result) < min_rows:
        raise ValidationError(f"CSV incompleto: {relative}")
    return result


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_reports(reports: dict[int, dict[str, Any]]) -> str:
    """Asegura fechas, activos y procedencia comunes entre los seis experimentos."""
    if set(reports) != set(REPORTS):
        raise ValidationError("No están disponibles todos los reportes S02–S06")
    for sprint, report in reports.items():
        if report.get("sprint") != sprint or report.get("status") != "PASS":
            raise ValidationError(f"Sprint {sprint} no está aprobado")
        if set(report.get("symbols", {})) != set(SYMBOLS):
            raise ValidationError(f"Sprint {sprint}: faltan activos")
    source_sha = reports[2].get("data_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValidationError("Hash contractual no disponible en S02")
    if reports[3].get("data_sha256") != source_sha:
        raise ValidationError("S02 y S03 utilizan históricos diferentes")
    s02_audit = reports[2].get("source_audit", {})
    if (s02_audit.get("summary") != "PASS"
        or s02_audit.get("csv_parquet_equivalence", {}).get("status") != "PASS"):
        raise ValidationError("La auditoría integral del Sprint 01 no está aprobada")
    if reports[4].get("audit") != "PASS" or reports[4].get("scope") != "BACKTEST_ONLY_NO_REAL_TIME_FORECAST":
        raise ValidationError("El Sprint 04 no tiene procedencia de backtest aprobada")
    for sprint in (5, 6):
        if reports[sprint].get("scope") != "HISTORICAL_BACKTEST_ONLY":
            raise ValidationError(f"Sprint {sprint} no es un backtest histórico")
        if reports[sprint].get("source_audit") != "PASS":
            raise ValidationError(f"Sprint {sprint}: auditoría fallida")
    dataset_hash = reports[4].get("sha256_dataset_zip")
    source_hash = reports[4].get("sha256_source_zip")
    if (reports[5].get("dataset_sha256") != dataset_hash
        or reports[5].get("source_zip_sha256") != source_hash
        or reports[6].get("dataset_sha256") != dataset_hash
        or reports[6].get("source_zip_sha256") != source_hash):
        raise ValidationError("Cambió la procedencia entre S04, S05 y S06")
    for asset in SYMBOLS:
        for h in HORIZONS:
            if str(h) not in reports[4]["symbols"][asset] or str(h) not in reports[5]["symbols"][asset] or str(h) not in reports[6]["symbols"][asset]:
                raise ValidationError(f"Falta {asset}/h={h}")
    return source_sha


def hash_compatible(path: Path, esperado: str) -> bool:
    if not isinstance(esperado, str) or len(esperado) != 64:
        return False

    datos = path.read_bytes()
    lf = datos.replace(b"\r\n", b"\n")

    return esperado in (
        hashlib.sha256(datos).hexdigest(),
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    )


def select_latest_complete(results_root: Path) -> tuple[dict[int, Path],dict[int, dict[str,Any]]]:
    """Usa ejecuciones completas existentes; no recalcula ni cambia ningún sprint."""
    root = Path(results_root).resolve(strict=True)
    paths: dict[int,Path] = {}
    objects: dict[int,dict[str,Any]] = {}
    for sprint, filename in REPORTS.items():
        options = sorted(root.glob(f"sprint{sprint:02d}_*/{filename}"), reverse=True)
        for option in options:
            try:
                data = read_json(root, str(option.relative_to(root)))
                if data.get("status") != "PASS":
                    continue
                paths[sprint] = option.parent
                objects[sprint] = data
                break
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        if sprint not in paths:
            raise FileNotFoundError(f"Falta reporte aprobado S{sprint:02d}: {filename}. Ejecute antes ese sprint.")
    validate_reports(objects)
    # Evita mezclar una ejecución antigua de S04 o S05 con la nueva S06.
    if not (
        hash_compatible(paths[4] / REPORTS[4], objects[5].get("s04_report_sha256"))
        and hash_compatible(paths[4] / REPORTS[4], objects[6].get("s04_report_sha256"))
        and hash_compatible(paths[5] / REPORTS[5], objects[6].get("s05_report_sha256"))
    ):
        raise ValidationError("S04/S05/S06 no proceden de las mismas ejecuciones")
    return paths, objects


def validate_artifacts(results_root: Path, folders: dict[int,Path]) -> int:
    """Comprueba todos los archivos que la interfaz necesita sin modificar ninguno."""
    count = 0
    root = Path(results_root).resolve(strict=True)
    for symbol in SYMBOLS:
        files = {
            2: (f"{symbol}_transformado.csv",f"{symbol}_normalidad.png",f"{symbol}_suavizado.png"),
            3: (f"{symbol}_componentes.csv",f"{symbol}_espectro.csv",f"{symbol}_descomposicion.png",f"{symbol}_tendencia.png",f"{symbol}_periodograma.png"),
            4: (f"{symbol}_h1_test.csv",f"{symbol}_h5_test.csv"),
            5: (f"{symbol}_h1_evaluacion.csv",f"{symbol}_h5_evaluacion.csv"),
            6: (f"{symbol}_h1_calibracion.csv",f"{symbol}_h5_calibracion.csv"),
        }
        for sprint, names in files.items():
            for name in names:
                file = safe_path(root, str((folders[sprint] / name).relative_to(root)))
                if file.stat().st_size == 0:
                    raise ValidationError(f"Archivo vacío: {file}")
                count += 1
    return count


def academic_summary(reports: dict[int,dict[str,Any]], symbol: str, h: int) -> dict[str,Any]:
    """Reúne resultados sin transformarlos en afirmaciones de capacidad predictiva."""
    if symbol not in SYMBOLS or h not in HORIZONS:
        raise ValidationError("Activo u horizonte inválidos")
    s2 = reports[2]["symbols"][symbol]
    s3 = reports[3]["symbols"][symbol]
    s4 = reports[4]["symbols"][symbol][str(h)]
    s5 = reports[5]["symbols"][symbol][str(h)]
    s6 = reports[6]["symbols"][symbol][str(h)]
    return {
        "normality":s2["normality_logreturns"],
        "outliers":int(s2["outliers_mad_logreturns"]),
        "trend":s3["trend"],
        "stl":s3["stl"],
        "peaks":s3["spectrum"]["top_peaks_exploratory"],
        "selected_model":s4["selected_model"],
        "mae":float(s5["selected_metrics"]["mae"]),
        "naive_mae":float(s5["naive_metrics"]["mae"]),
        "coverage_fixed":float(s6["static_s05_coverage"]),
        "coverage_rolling":float(s6["rolling_coverage"]),
        "nominal":float(s6["nominal"]),
    }
