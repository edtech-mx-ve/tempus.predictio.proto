"""Exportación auditada de evidencias y PDF académico, sin acceso a la red.

No incluye información del sistema operativo, archivos de configuración,
datos externos no auditados ni el entorno virtual. Todas las rutas exportadas
son relativas y verificadas dentro de la carpeta de resultados.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from predictio_s07.integration import (
    HORIZONS, REPORTS, SYMBOLS, ValidationError, academic_summary,
    safe_path, select_latest_complete, validate_artifacts,
)
from predictio_s08.reporting import pdf_report, md_report, summary_matrix
from predictio_s09.normality import histogram_figure, qq_figure
from predictio_visual07.charts import (
    validation_chart, test_predictions_chart, error_chart,
    prepare_test_predictions, prepare_error_series,
)

LOG = logging.getLogger("tempus.entregables")
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_TOTAL_BYTES = 250 * 1024 * 1024
ALLOWED_EXTENSIONS = frozenset((".csv", ".png", ".json", ".md", ".pdf"))
ZIP_ROOT = "TEMPUS_PREDICTIO_PRACTICA_2"
EXPECTED_AUDIT_COUNT = 70


def audit_input(root: Path) -> tuple[dict[int, Path], dict[int, dict[str, Any]], int]:
    """Selecciona ejecuciones coherentes y comprueba las evidencias antes de exportar."""
    folders, reports = select_latest_complete(root)
    count = validate_artifacts(root, folders)
    if count != EXPECTED_AUDIT_COUNT:
        raise ValidationError(f"Se esperaban {EXPECTED_AUDIT_COUNT} artefactos; disponibles: {count}")
    return folders, reports, count


def _png_bytes(fig: Any) -> bytes:
    try:
        out = io.BytesIO()
        fig.savefig(out, format="png", dpi=120, bbox_inches="tight")
        data = out.getvalue()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValidationError("La gráfica no es un PNG válido")
        return data
    finally:
        plt.close(fig)


def additional_charts(folders: dict[int, Path],
                      reports: dict[int, dict[str, Any]]) -> dict[str, bytes]:
    """Reproduce solo gráficos de datos históricos guardados; no inventa predicciones."""
    result: dict[str, bytes] = {}
    for symbol in SYMBOLS:
        csv_path = folders[2] / f"{symbol}_transformado.csv"
        source = pd.read_csv(csv_path, usecols=["LogReturn"])
        returns = pd.to_numeric(source["LogReturn"], errors="raise").to_numpy()
        result[f"graficas_adicionales/01_normalidad/{symbol}_histograma.png"] = _png_bytes(
            histogram_figure(returns, symbol))
        result[f"graficas_adicionales/01_normalidad/{symbol}_QQ.png"] = _png_bytes(
            qq_figure(returns, symbol))
        for horizon in HORIZONS:
            item = reports[4]["symbols"][symbol][str(horizon)]
            selected = academic_summary(reports, symbol, horizon)["selected_model"]
            validation = pd.DataFrame([
                {"Modelo": name, "MAE validación": float(values["mae"])}
                for name, values in item["validation_metrics"].items()
            ])
            result[f"graficas_adicionales/07_modelos/{symbol}_h{horizon}_MAE_validacion.png"] = _png_bytes(
                validation_chart(validation, selected))
            test_file = folders[5] / f"{symbol}_h{horizon}_evaluacion.csv"
            frame = pd.read_csv(test_file)
            predictions = prepare_test_predictions(frame, horizon, selected)
            result[f"graficas_adicionales/07_modelos/{symbol}_h{horizon}_pronosticos.png"] = _png_bytes(
                test_predictions_chart(predictions, symbol, horizon))
            result[f"datos_generados/07_modelos/{symbol}_h{horizon}_pronosticos.csv"] = (
                predictions.to_csv(index=False).encode("utf-8-sig"))
            calibration = pd.read_csv(folders[6] / f"{symbol}_h{horizon}_calibracion.csv")
            errors = prepare_error_series(calibration, horizon)
            result[f"graficas_adicionales/08_evaluacion/{symbol}_h{horizon}_error_absoluto.png"] = _png_bytes(
                error_chart(errors, symbol, horizon))
            result[f"datos_generados/08_evaluacion/{symbol}_h{horizon}_errores.csv"] = (
                errors.to_csv(index=False).encode("utf-8-sig"))
    return result


def _collect_generated(root: Path, folders: dict[int, Path]) -> list[tuple[Path, str]]:
    """Recoge TODOS los resultados de ejecuciones auditadas, sin incluir rutas privadas."""
    result: list[tuple[Path, str]] = []
    count_bytes = 0
    base = root.resolve(strict=True)
    for stage, folder in sorted(folders.items()):
        if not folder.resolve(strict=True).is_relative_to(base):
            raise ValidationError("La carpeta de resultados escapa del directorio permitido")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValidationError("No se permiten enlaces simbólicos en los entregables")
            if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            rel = path.relative_to(folder)
            # Ignorar carpetas ocultas y datos ajenos a los resultados del procedimiento.
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            verified = safe_path(base, str(path.relative_to(base)))
            size = verified.stat().st_size
            if not (0 < size <= MAX_FILE_BYTES):
                raise ValidationError(f"Archivo inválido o excesivo: {path.name}")
            count_bytes += size
            if count_bytes > MAX_TOTAL_BYTES:
                raise ValidationError("Volumen total excede el límite de exportación")
            arcname = f"datos_generados/procedimiento_{stage:02d}/{rel.as_posix()}"
            result.append((verified, arcname))
    return result


def _pdf_bytes(reports: dict[int, dict[str, Any]], folders: dict[int, Path],
               rows: list[dict[str, Any]], count: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="tempus_informe_") as temp:
        path = Path(temp) / "INFORME_FINAL_PRACTICA_2.pdf"
        pdf_report(path, reports, folders, rows, count)
        data = path.read_bytes()
        if len(data) < 1024 or not data.startswith(b"%PDF"):
            raise ValidationError("El informe PDF generado es inválido")
        from predictio_economia.pdf09 import add_historical_economic_page
        data = add_historical_economic_page(data, rows)
        from predictio_academico.protocolo import formalizar_pdf
        return formalizar_pdf(data, "practica2")


def build_deliverables(results_root: Path) -> tuple[bytes, bytes, dict[str, Any]]:
    """Construye ZIP y PDF en memoria, siempre desde los mismos resultados validados.

    Retorna (zip, pdf, metadatos). No altera ni sobrescribe datos históricos.
    """
    root = Path(results_root).resolve(strict=True)
    folders, reports, count = audit_input(root)
    rows = summary_matrix(reports)
    if len(rows) != len(SYMBOLS) * len(HORIZONS):
        raise ValidationError("No hay diez experimentos históricos completos")

    pdf = _pdf_bytes(reports, folders, rows, count)
    from predictio_economia.lectura import historical_economic_reading
    historial = historical_economic_reading(rows)
    appendix = [
        "",
        "## Lectura económica de resultados históricos",
        "",
        "Se presenta después de los resultados técnicos; no es una predicción, "
        "señal de compra ni cálculo de ganancias personales.",
        "",
        "| Empresa | h | MAE (USD) | Ancho móvil (USD) | Cobertura (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in historial:
        appendix.append(
            f'| {item["activo"]} | {item["horizonte"]} | '
            f'{item["mae_usd"]:.3f} | {item["ancho_intervalo_usd"]:.2f} | '
            f'{item["cobertura_pct"]:.1f} |'
        )
    appendix.extend([
        "",
        "El MAE es un error medio por acción, no una pérdida monetaria realizada. "
        "La cobertura y amplitud de los intervalos describen incertidumbre "
        "histórica; no demuestran rentabilidad.",
        "",
        "No se dispone de posiciones, precios de compra ni una serie auditada "
        "de dividendos para calcular beneficios realizados.",
    ])
    md = (md_report(reports, folders, rows, count) + "\n".join(appendix)).encode("utf-8")
    csv_data = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
    extra = additional_charts(folders, reports)
    collected = _collect_generated(root, folders)

    generated: dict[str, bytes] = {
        "informe/INFORME_FINAL_PRACTICA_2.pdf": pdf,
        "informe/INFORME_FINAL_PRACTICA_2.md": md,
        "informe/RESUMEN_RESULTADOS.csv": csv_data,
        "LEEME.txt": (
            "TEMPUS PREDICTIO PROTO - PRACTICA 2\\n"
            "Evidencias del analisis estadistico y backtesting historico.\\n"
            "Incluye los resultados auditados de los procedimientos 02 a 06,\\n"
            "graficas adicionales, tablas de los diez experimentos y el informe final.\\n"
            "JNJ, MSFT, JPM, WMT, XOM | 2016-10-31 a 2026-10-30.\\n"
            "No es una herramienta de pronostico en tiempo real ni asesoramiento financiero.\\n"
            "El contenido no incluye el entorno virtual ni archivos de configuracion privados.\\n"
        ).encode("utf-8"),
        **extra,
    }
    if len(set(name for _, name in collected)) != len(collected):
        raise ValidationError("Colisión de nombres de archivos de evidencia")
    if any(name in generated for _, name in collected):
        raise ValidationError("Colisión entre resultados generados y originales auditados")
    manifest = {
        "contrato": reports[2]["data_contract"],
        "sha256_fuente_csv": reports[2]["data_sha256"],
        "activos": list(SYMBOLS),
        "horizontes_sesiones": list(HORIZONS),
        "n_artefactos_auditados": count,
        "n_archivos_resultados": len(collected),
        "n_graficas_y_csv_adicionales": len(extra),
        "archivos": {},
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        for path, rel in collected:
            sha = hashlib.sha256()
            with path.open("rb") as source, z.open(f"{ZIP_ROOT}/{rel}", "w") as dest:
                for piece in iter(lambda: source.read(1024 * 1024), b""):
                    sha.update(piece)
                    dest.write(piece)
            manifest["archivos"][rel] = sha.hexdigest()
        for rel, data in sorted(generated.items()):
            if len(data) > MAX_FILE_BYTES:
                raise ValidationError(f"Archivo generado excesivo: {rel}")
            z.writestr(f"{ZIP_ROOT}/{rel}", data)
            manifest["archivos"][rel] = hashlib.sha256(data).hexdigest()
        z.writestr(f"{ZIP_ROOT}/MANIFIESTO_SHA256.json",
                   json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    zip_bytes = archive.getvalue()
    if not zip_bytes:
        raise ValidationError("El archivo ZIP está vacío")
    LOG.info("Entregables preparados: %d originales, %d adicionales, %d bytes",
             len(collected), len(extra), len(zip_bytes))
    return zip_bytes, pdf, {
        "archivos": len(manifest["archivos"])+1,
        "artefactos_auditados": count,
        "experimentos": len(rows),
        "zip_bytes": len(zip_bytes),
        "pdf_bytes": len(pdf),
    }
