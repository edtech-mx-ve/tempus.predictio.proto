"""Gráficas comparativas estrictamente históricas del procedimiento 07 y error del 08.

Las trayectorias de test proceden de datos persistidos y auditados.
Se comparan todos los candidatos por MAE de validación (sin consultar test).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger("tempus.prediccion.visual")


class ChartDataError(ValueError):
    """Datos de gráfica incompletos o inconsistentes."""


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...], positive: bool = False) -> pd.DataFrame:
    if frame.empty or not set(columns).issubset(frame.columns):
        raise ChartDataError(f"Columnas faltantes: {columns}")
    clean = frame.copy()
    for name in columns:
        clean[name] = pd.to_numeric(clean[name], errors="raise")
        if not np.isfinite(clean[name].to_numpy(dtype=float)).all():
            raise ChartDataError(f"Valores no finitos: {name}")
        if positive and (clean[name] <= 0).any():
            raise ChartDataError(f"Valores no positivos: {name}")
    return clean


def _dates(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if date_column not in frame.columns:
        raise ChartDataError(f"Falta la fecha {date_column}")
    clean = frame.copy()
    dates = pd.to_datetime(clean[date_column], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ChartDataError("Fechas incompletas, duplicadas o desordenadas")
    clean[date_column] = dates
    return clean


def prepare_test_predictions(frame: pd.DataFrame, horizon: int, selected_model: str) -> pd.DataFrame:
    """Valida el CSV de evaluación; nunca reconstruye predicciones con información futura."""
    if horizon not in (1, 5) or not isinstance(selected_model, str) or not selected_model:
        raise ChartDataError("Horizonte o modelo inválido")
    required = ("actual_close", "predicted_close", "naive_predicted_close")
    clean = _numeric(frame, required, positive=True)
    for column in ("origin_date", "target_date", "selected_model"):
        if column not in clean.columns:
            raise ChartDataError(f"Falta {column}")
    clean = _dates(clean, "target_date")
    origin = pd.to_datetime(clean["origin_date"], errors="raise")
    if origin.isna().any() or not (origin < clean["target_date"]).all():
        raise ChartDataError("El origen debe anteceder al objetivo")
    if not clean["selected_model"].eq(selected_model).all():
        raise ChartDataError("La identidad del modelo difiere de la selección validada")
    if "horizon_sessions" in clean and not clean["horizon_sessions"].eq(horizon).all():
        raise ChartDataError("Horizonte discordante")
    if len(clean) < 30:
        raise ChartDataError("Evaluación histórica insuficiente")
    return clean[["target_date", *required]].copy()


def validation_chart(frame: pd.DataFrame, selected_model: str) -> plt.Figure:
    """Compara todos los candidatos solo mediante el MAE de validación."""
    if "Modelo" not in frame.columns or "MAE validación" not in frame.columns:
        raise ChartDataError("La tabla de validación carece de columnas")
    clean = _numeric(frame, ("MAE validación",))
    if clean["Modelo"].duplicated().any() or selected_model not in clean["Modelo"].to_numpy():
        raise ChartDataError("Modelo elegido ausente o candidatos repetidos")
    if (clean["MAE validación"] < 0).any():
        raise ChartDataError("MAE negativo")
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    labels = clean["Modelo"].astype(str).to_list()
    values = clean["MAE validación"].to_numpy(dtype=float)
    bars = ax.barh(labels, values, color=["#176c57" if x == selected_model else "#8fa6bc" for x in labels])
    ax.invert_yaxis()
    ax.set(xlabel="MAE en validación (USD por acción)", title="Comparación de modelos: menor MAE es mejor")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlim(0, max(values.max() * 1.22, 0.01))
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    return fig


def test_predictions_chart(frame: pd.DataFrame, symbol: str, horizon: int) -> plt.Figure:
    """Muestra las series de prueba reales, seleccionada e ingenua."""
    clean = _dates(_numeric(frame, ("actual_close", "predicted_close", "naive_predicted_close"), positive=True), "target_date")
    if len(clean) < 30:
        raise ChartDataError("Prueba insuficiente")
    fig, ax = plt.subplots(figsize=(11, 4.7))
    x = clean["target_date"]
    ax.plot(x, clean["actual_close"], linewidth=1.55, label="Precio observado")
    ax.plot(x, clean["predicted_close"], linewidth=1.05, alpha=0.90, label="Modelo seleccionado")
    ax.plot(x, clean["naive_predicted_close"], linewidth=0.95, alpha=0.76, linestyle="--", label="Modelo ingenuo")
    ax.set(title=f"{symbol} · Pronósticos históricos · horizonte {horizon}", xlabel="Fecha objetivo", ylabel="Cierre (USD por acción)")
    ax.legend(loc="best")
    ax.grid(alpha=0.17)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def prepare_error_series(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Extrae errores de un conjunto test sin modificar los datos originales."""
    if horizon not in (1, 5):
        raise ChartDataError("Horizonte inválido")
    clean = _dates(_numeric(frame, ("actual_close", "predicted_close"), positive=True), "target_date")
    if "horizon_sessions" in clean and not clean["horizon_sessions"].eq(horizon).all():
        raise ChartDataError("Horizonte inconsistente")
    if len(clean) < 30:
        raise ChartDataError("Muestra insuficiente")
    return pd.DataFrame({
        "target_date": clean["target_date"],
        "absolute_error_usd": (clean["predicted_close"] - clean["actual_close"]).abs(),
    })


def error_chart(frame: pd.DataFrame, symbol: str, horizon: int) -> plt.Figure:
    """Evolución temporal del error absoluto del modelo elegido."""
    clean = _dates(_numeric(frame, ("absolute_error_usd",)), "target_date")
    if (clean["absolute_error_usd"] < 0).any():
        raise ChartDataError("Error absoluto negativo")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(clean["target_date"], clean["absolute_error_usd"], linewidth=1.0, label="Error absoluto")
    ax.set(title=f"{symbol} · Error de pronóstico histórico · horizonte {horizon}", xlabel="Fecha objetivo", ylabel="Error absoluto (USD)")
    ax.grid(alpha=.18)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
