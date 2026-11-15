"""Visualizaciones autónomas y verificables del test de normalidad (Sprint 09).

No elimina atípicos, no aplica suavizado para forzar normalidad, no modifica
el archivo de origen. Las dos figuras se obtienen de los log-retornos S02.
"""
from __future__ import annotations
import io
import math
from typing import Sequence

import numpy as np
from scipy import stats

def valid_returns(values: Sequence[float]) -> np.ndarray:
    """Valida y devuelve log-retornos finitos, respetando el orden original."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("Los retornos deben ser un vector")
    if data.size and np.isnan(data[0]):
        data = data[1:]  # primera diferencia sin predecesor
    if data.size < 20 or not np.isfinite(data).all():
        raise ValueError("Se requieren >=20 retornos finitos")
    if float(data.std(ddof=1)) <= 0:
        raise ValueError("La varianza de la serie debe ser positiva")
    return data

def normality_summary(values: Sequence[float]) -> dict:
    """Cálculos puros para documentar la referencia normal y el contraste."""
    data = valid_returns(values)
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)),
        "dagostino_p": float(stats.normaltest(data).pvalue),
        "jarque_bera_p": float(stats.jarque_bera(data).pvalue),
    }

def histogram_figure(values: Sequence[float], symbol: str):
    """Histograma de densidad y PDF normal estimada, separados del Q-Q."""
    import matplotlib.pyplot as plt
    data = valid_returns(values)
    if symbol not in ("JNJ", "MSFT", "JPM", "WMT", "XOM"):
        raise ValueError("Activo no autorizado")
    mu, std = float(data.mean()), float(data.std(ddof=1))
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.hist(data, bins="fd", density=True, color="#346A96",
            edgecolor="#F6F8FA", alpha=0.8, label="Densidad empírica")
    x = np.linspace(float(data.min()), float(data.max()), 400)
    ax.plot(x, stats.norm.pdf(x, mu, std), color="#B74736", linewidth=2.2,
            label="Normal teórica (media y desviación estimadas)")
    ax.set(title=f"{symbol} | Histograma de log-retornos y curva normal",
           xlabel="Log-retorno por sesión", ylabel="Densidad (no frecuencia)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.15)
    return fig

def qq_figure(values: Sequence[float], symbol: str):
    """Q-Q de log-retornos contra cuantiles normales; gráfico complementario."""
    import matplotlib.pyplot as plt
    data = valid_returns(values)
    if symbol not in ("JNJ", "MSFT", "JPM", "WMT", "XOM"):
        raise ValueError("Activo no autorizado")
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    stats.probplot(data, dist="norm", plot=ax)
    ax.set(title=f"{symbol} | Q-Q frente a normal", xlabel="Cuantiles teóricos normales",
           ylabel="Cuantiles observados de log-retornos")
    ax.grid(alpha=0.15)
    return fig
