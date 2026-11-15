"""Transformaciones puras de resultados históricos en interpretaciones económicas.

No estima dividendos, decisiones de inversión ni ganancias realizadas.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

TICKERS = ("JNJ", "MSFT", "JPM", "WMT", "XOM")
HORIZONS = (1, 5)


def validate_nonnegative(value: Any, name: str) -> float:
    """Valida un indicador no negativo y finito."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: valor no numérico") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name}: valor fuera de rango")
    return result


def historical_economic_reading(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lee únicamente resultados observados en prueba del módulo 09."""
    if len(rows) != 10:
        raise ValueError("Se requieren diez escenarios históricos")
    seen: set[tuple[str, int]] = set()
    result: list[dict[str, Any]] = []
    for item in rows:
        ticker = str(item["activo"])
        h = int(item["horizonte"])
        key = (ticker, h)
        if ticker not in TICKERS or h not in HORIZONS or key in seen:
            raise ValueError("Activos, horizontes o duplicados inválidos")
        seen.add(key)
        mae = validate_nonnegative(item["mae"], "mae")
        width = validate_nonnegative(item["ancho_movil_usd"], "ancho_movil_usd")
        coverage = validate_nonnegative(item["cobertura_movil_pct"], "cobertura_movil_pct")
        nominal = validate_nonnegative(item["nominal_pct"], "nominal_pct")
        if coverage > 100 or nominal > 100:
            raise ValueError("Coberturas fuera de 0-100")
        result.append({
            "activo":ticker, "horizonte":h, "mae_usd":mae,
            "ancho_intervalo_usd":width, "cobertura_pct":coverage,
            "cobertura_nominal_pct":nominal,
            "brecha_cobertura_pp":coverage - nominal,
        })
    if seen != {(t,h) for t in TICKERS for h in HORIZONS}:
        raise ValueError("Faltan escenarios")
    return result


def prediction_economic_reading(
    raw: pd.DataFrame,
    selection: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Obtiene del último origen de prueba una comparación precio previsto vs real.

    La diferencia es ERROR de predicción por acción, NUNCA rentabilidad realizada.
    """
    required = {"ticker", "particion", "modelo", "horizonte", "fecha_origen",
                "fecha_objetivo", "precio_predicho", "precio_real"}
    if not required.issubset(raw.columns):
        raise ValueError("Faltan campos de predicción verificable")
    expected = {(t,h) for t in TICKERS for h in HORIZONS}
    if not {"ticker","horizonte","modelo"}.issubset(selection.columns):
        raise ValueError("Selección incompleta")
    if len(selection) != 10 or set(zip(selection.ticker,selection.horizonte)) != expected:
        raise ValueError("Faltan diez selecciones académicas")
    result=[]
    for t in TICKERS:
        for h in HORIZONS:
            model = str(selection.loc[
                (selection.ticker == t) & (selection.horizonte == h), "modelo"
            ].iloc[0])
            if model not in ("ARIMA(1,1,0)", "Holt-Winters aditivo sin estacionalidad"):
                raise ValueError("Modelo principal inválido")
            d = raw.loc[
                (raw.ticker == t) & (raw.horizonte == h)
                & (raw.particion == "prueba") & (raw.modelo == model)
            ].copy()
            if d.empty:
                raise ValueError(f"Sin prueba: {t}, h={h}")
            for col in ("fecha_objetivo","fecha_origen"):
                d[col] = pd.to_datetime(d[col], errors="raise")
            d = d.sort_values(["fecha_origen","fecha_objetivo"])
            last = d.iloc[-1]
            predicted = float(last["precio_predicho"])
            realized = float(last["precio_real"])
            if not all(map(math.isfinite,(predicted,realized))) or min(predicted,realized)<=0:
                raise ValueError("Precios ausentes o fuera de rango")
            result.append({
                "activo":t, "horizonte":h,
                "modelo":"ARIMA" if model.startswith("ARIMA") else "Holt-Winters",
                "origen":last["fecha_origen"].date().isoformat(),
                "objetivo":last["fecha_objetivo"].date().isoformat(),
                "predicho_usd":predicted,
                "real_usd":realized,
                "error_firmado_usd":predicted-realized,
                "error_absoluto_usd":abs(predicted-realized),
            })
    return result


def historical_company_interpretations(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Interpreta, por empresa, error, incertidumbre y cobertura YA observados."""
    validated = historical_economic_reading(rows)
    by_key = {(r["activo"], r["horizonte"]): r for r in validated}
    report: list[dict[str, str]] = []
    for ticker in TICKERS:
        one, five = (by_key[(ticker, h)] for h in HORIZONS)
        delta = five["mae_usd"] - one["mae_usd"]
        direction = "aumentó" if delta > 0 else ("disminuyó" if delta < 0 else "no cambió")
        compare = "mayor" if delta > 0 else ("menor" if delta < 0 else "igual")
        nominal = one["cobertura_nominal_pct"]
        coverage_1 = ("cumplió" if one["cobertura_pct"] >= nominal else "quedó por debajo de")
        coverage_5 = ("cumplió" if five["cobertura_pct"] >= five["cobertura_nominal_pct"]
                      else "quedó por debajo de")
        description = (
            f"El error histórico medio pasó de {one['mae_usd']:.3f} USD por acción "
            f"(1 sesión) a {five['mae_usd']:.3f} USD (5 sesiones); "
            f"{direction} {abs(delta):.3f} USD. "
            f"El ancho del intervalo pasó de {one['ancho_intervalo_usd']:.2f} "
            f"a {five['ancho_intervalo_usd']:.2f} USD, señalando una amplitud "
            f"histórica {compare} al ampliar el horizonte. "
            f"La cobertura observada fue {one['cobertura_pct']:.1f}% y "
            f"{five['cobertura_pct']:.1f}%: a 1 sesión {coverage_1} la referencia "
            f"nominal de {nominal:.0f}% y a 5 sesiones {coverage_5} "
            f"la referencia nominal de {five['cobertura_nominal_pct']:.0f}%. "
        )
        report.append({"activo": ticker, "texto": description})
    return report


def prediction_company_interpretations(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Sintetiza cinco empresas sin confundir error de estimación con rentabilidad."""
    if len(rows) != len(TICKERS) * len(HORIZONS):
        raise ValueError("Se requieren diez predicciones de prueba")
    data = {(str(r["activo"]), int(r["horizonte"])): r for r in rows}
    if set(data) != {(t, h) for t in TICKERS for h in HORIZONS}:
        raise ValueError("Escenarios por empresa y horizonte incompletos")
    result: list[dict[str, str]] = []
    for ticker in TICKERS:
        a, b = (data[(ticker,h)] for h in HORIZONS)
        fragments = []
        for h, item in [(1,a),(5,b)]:
            expected = float(item["predicho_usd"])
            observed = float(item["real_usd"])
            if not math.isfinite(expected) or not math.isfinite(observed) or expected <= 0 or observed <= 0:
                raise ValueError("Precios inválidos")
            diff = observed - expected
            if abs(diff) < 0.005:
                part = "coincidió prácticamente con el precio observado"
            elif diff > 0:
                part = f"subestimó el precio observado en {diff:.2f} USD por acción"
            else:
                part = f"sobreestimó el precio observado en {-diff:.2f} USD por acción"
            fragments.append(f"A {h} sesión{'es' if h != 1 else ''}, {item['modelo']} {part}")
        delta1 = float(a["real_usd"]) - float(a["predicho_usd"])
        delta5 = float(b["real_usd"]) - float(b["predicho_usd"])
        if delta1 > 0 and delta5 > 0:
            conclusion = "Los dos pronósticos quedaron por debajo de las cotizaciones efectivamente observadas."
        elif delta1 < 0 and delta5 < 0:
            conclusion = "Los dos pronósticos quedaron por encima de las cotizaciones efectivamente observadas."
        elif delta1 * delta5 < 0:
            conclusion = "El sentido del error cambió entre ambos horizontes: una advertencia sobre la incertidumbre predictiva."
        else:
            conclusion = "Las diferencias evidencian incertidumbre en ambos horizontes."
        text = f"{fragments[0]}; {fragments[1]}. {conclusion}"

        result.append({"activo": ticker, "texto": text})
    return result
