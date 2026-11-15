"""Lectura y exportación segura de experimentos de predicción temporal.

Los resultados se calculan previamente; este módulo nunca ejecuta entrenamiento
durante la visualización y no utiliza datos de prueba para seleccionar modelos.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger("tempus.predictio.integracion")
TICKERS = ("JNJ", "MSFT", "JPM", "WMT", "XOM")
HORIZONS = (1, 5)
MODEL_NAMES = ("Ingenuo", "ARIMA(1,1,0)", "Holt-Winters aditivo sin estacionalidad")
STUDY_MODELS = ("ARIMA(1,1,0)", "Holt-Winters aditivo sin estacionalidad")
DATA_COLUMNS = (
    "ticker", "particion", "horizonte", "modelo", "fecha_origen",
    "fecha_objetivo", "precio_real", "precio_predicho", "error", "error_absoluto",
)
METRIC_COLUMNS = ("ticker", "particion", "horizonte", "modelo", "n", "MAE", "RMSE", "sesgo")
ARTIFACTS = ("resumen_modelos.csv", "seleccion_validacion.csv",
             "evaluacion_seleccionados.csv", "metodologia.json",
             *tuple(f"{t}_predicciones.csv" for t in TICKERS),
             *tuple(f"{t}_h{h}_comparacion.png" for t in TICKERS for h in HORIZONS))
MAX_FILE_SIZE = 30 * 1024 * 1024
MAX_ARCHIVE_SIZE = 120 * 1024 * 1024


class DataValidationError(ValueError):
    """Inconsistencia de datos o integridad de resultados."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root_check(folder: Path, results_root: Path) -> Path:
    base = results_root.resolve(strict=True)
    candidate = folder.resolve(strict=True)
    if not candidate.is_relative_to(base) or candidate == base:
        raise DataValidationError("Carpeta fuera de resultados permitidos")
    if folder.is_symlink():
        raise DataValidationError("No se permiten enlaces simbólicos")
    return candidate


def select_run(results_root: Path) -> Path:
    """Selecciona la ejecución más reciente, sin sustituir fallas por informes viejos."""
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de resultados: {root}")
    dirs = sorted((p for p in root.glob("predictio_proto10_*") if p.is_dir()),
                  key=lambda p: p.name, reverse=True)
    if not dirs:
        raise FileNotFoundError(
            "No hay resultados ARIMA/Holt-Winters. Ejecute ejecutar_predictio_proto10.ps1.")
    result = _root_check(dirs[0], root)
    return result


def verify_run(run: Path, results_root: Path) -> dict[str, str]:
    folder = _root_check(Path(run), Path(results_root))
    manifest_path = folder / "manifest_sha256.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DataValidationError("Falta manifiesto de integridad de predicción")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DataValidationError("Manifiesto inválido")
    missing = set(ARTIFACTS) - set(manifest)
    if missing:
        raise DataValidationError(f"Faltan entradas del manifiesto: {sorted(missing)}")
    total = 0
    for filename in ARTIFACTS:
        path = folder / filename
        if not path.is_file() or path.is_symlink():
            raise DataValidationError(f"Falta archivo de la ejecución: {filename}")
        if path.resolve(strict=True).parent != folder:
            raise DataValidationError("Ruta de resultado fuera del experimento")
        size = path.stat().st_size
        if size < 1 or size > MAX_FILE_SIZE:
            raise DataValidationError(f"Tamaño de resultado inválido: {filename}")
        total += size
        if total > MAX_ARCHIVE_SIZE:
            raise DataValidationError("Los resultados exceden el límite del paquete")
        esperado = manifest[filename]
        contenido = path.read_bytes()

        coincide = (
            isinstance(esperado, str)
            and digest(contenido) == esperado
        )

        if (
            not coincide
            and isinstance(esperado, str)
            and path.suffix.lower() in {".csv", ".json"}
        ):
            lf = contenido.replace(b"\r\n", b"\n")
            coincide = (
                digest(lf) == esperado
                or digest(lf.replace(b"\n", b"\r\n")) == esperado
            )

        if not coincide:
            raise DataValidationError(
                f"SHA-256 distinto en {filename}"
            )
    return manifest


def _assert_text_column(frame: pd.DataFrame, column: str, choices: tuple[str,...]) -> None:
    if frame[column].isna().any() or not frame[column].isin(choices).all():
        raise DataValidationError(f"Valores inválidos en {column}")


def _numeric(frame: pd.DataFrame, names: tuple[str,...]) -> None:
    import numpy as np
    for name in names:
        vals = pd.to_numeric(frame[name], errors="raise")
        if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
            raise DataValidationError(f"Columna no finita: {name}")
        frame[name] = vals


def load_run(run: Path, results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str,Any]]:
    verify_run(run, results_root)
    folder = Path(run)
    summary = pd.read_csv(folder / "resumen_modelos.csv")
    selection = pd.read_csv(folder / "seleccion_validacion.csv")
    metadata = json.loads((folder / "metodologia.json").read_text(encoding="utf-8"))
    if metadata.get("contrato") != "predictio-proto-arima-holt/1.0":
        raise DataValidationError("Contrato de resultados incompatible")
    if metadata.get("no_es_pronostico_operativo") is not True:
        raise DataValidationError("Los resultados deben declararse retrospectivos")
    if not set(METRIC_COLUMNS).issubset(summary.columns):
        raise DataValidationError("Faltan columnas de resumen")
    if not {"ticker","horizonte","modelo","MAE_validacion"}.issubset(selection.columns):
        raise DataValidationError("Faltan columnas de selección")
    for df in (summary, selection):
        _assert_text_column(df,"ticker",TICKERS)
        _assert_text_column(df,"modelo",MODEL_NAMES)
        _numeric(df, ("horizonte",))
        if not df["horizonte"].isin(HORIZONS).all():
            raise DataValidationError("Horizonte de predicción no admitido")
    _assert_text_column(summary,"particion",("validacion","prueba"))
    _numeric(summary,("n","MAE","RMSE","sesgo"))
    _numeric(selection,("MAE_validacion",))
    if (summary[["n","MAE","RMSE"]] < 0).any().any() or (summary["n"] < 1).any():
        raise DataValidationError("Métricas de error inválidas")
    if summary.duplicated(["ticker","horizonte","particion","modelo"]).any():
        raise DataValidationError("Duplicados en resumen")
    expected = {(t,h,p,m) for t in TICKERS for h in HORIZONS for p in ("validacion","prueba") for m in MODEL_NAMES}
    actual = set(summary[["ticker","horizonte","particion","modelo"]].itertuples(index=False,name=None))
    if actual != expected:
        raise DataValidationError("No se encontraron las 60 combinaciones esperadas")
    if len(selection) != 10 or selection.duplicated(["ticker","horizonte"]).any():
        raise DataValidationError("Se requieren exactamente diez selecciones")
    expected_selection={(t,h) for t in TICKERS for h in HORIZONS}
    if set(selection[["ticker","horizonte"]].itertuples(index=False,name=None)) != expected_selection:
        raise DataValidationError("Selección incompleta")
    # Comprobar que el modelo seleccionado fue elegido sin consultar prueba.
    validation = summary.query("particion == 'validacion'")
    for row in selection.itertuples(index=False):
        item=validation.loc[(validation["ticker"]==row.ticker)&(validation["horizonte"]==row.horizonte)]
        best=item.sort_values(["MAE","modelo"]).iloc[0]
        if row.modelo != best["modelo"] or not math.isclose(row.MAE_validacion, best["MAE"], rel_tol=1e-8,abs_tol=1e-8):
            raise DataValidationError("Selección no coincide con validación")
    # La selección original entre tres métodos permanece inmutable y auditada.
    # La selección académica se deriva únicamente de los dos modelos estudiados.
    selection = select_study_models(summary)
    raw_frames=[]
    for ticker in TICKERS:
        df=pd.read_csv(folder/f"{ticker}_predicciones.csv")
        if not set(DATA_COLUMNS).issubset(df.columns):
            raise DataValidationError(f"Faltan columnas del activo {ticker}")
        _assert_text_column(df,"ticker",(ticker,))
        _assert_text_column(df,"particion",("validacion","prueba"))
        _assert_text_column(df,"modelo",MODEL_NAMES)
        _numeric(df,("horizonte","precio_real","precio_predicho","error","error_absoluto"))
        if not df["horizonte"].isin(HORIZONS).all():
            raise DataValidationError("Horizonte fuera de contrato")
        if (df[["precio_real","precio_predicho"]] <= 0).any().any():
            raise DataValidationError("El precio de cierre debe ser positivo")
        origin=pd.to_datetime(df["fecha_origen"],errors="raise")
        target=pd.to_datetime(df["fecha_objetivo"],errors="raise")
        if origin.isna().any() or target.isna().any() or not (origin < target).all():
            raise DataValidationError("Se encontró un objetivo no posterior al origen")
        if df.duplicated(["particion","horizonte","modelo","fecha_objetivo"]).any():
            raise DataValidationError("Fechas objetivo duplicadas")
        if not ((df["precio_real"] - df["precio_predicho"] - df["error"]).abs() <= 0.00001).all():
            raise DataValidationError("Error inconsistente con precios")
        if not ((df["error"].abs()-df["error_absoluto"]).abs() <= 0.00001).all():
            raise DataValidationError("Error absoluto inconsistente")
        raw_frames.append(df)
    raw=pd.concat(raw_frames,ignore_index=True)
    grouped=raw.groupby(["ticker","particion","horizonte","modelo"])
    for key, rows in grouped:
        metric=summary.loc[(summary["ticker"]==key[0])&(summary["particion"]==key[1])&
            (summary["horizonte"]==key[2])&(summary["modelo"]==key[3])].iloc[0]
        import numpy as np
        checks=(
          (metric["n"],len(rows)),
          (metric["MAE"],rows["error_absoluto"].mean()),
          (metric["RMSE"],float(np.sqrt(np.mean(np.square(rows["error"]))))),
          (metric["sesgo"],rows["error"].mean()),
        )
        if any(not math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-7) for a,b in checks):
            raise DataValidationError(f"Las métricas no reproducen los pronósticos: {key}")
    if len(grouped) != 60:
        raise DataValidationError("Faltan series de pronóstico")
    return raw, summary, selection, metadata


def select_study_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Selecciona ARIMA/Holt-Winters por MAE de validación, sin usar prueba."""
    required = {"ticker", "particion", "horizonte", "modelo", "MAE"}
    if not required.issubset(summary.columns):
        raise DataValidationError("Faltan métricas para selección académica")
    validation = summary.loc[
        (summary["particion"] == "validacion") &
        (summary["modelo"].isin(STUDY_MODELS))
    ].copy()
    expected = {(t,h,m) for t in TICKERS for h in HORIZONS for m in STUDY_MODELS}
    actual = set(validation[["ticker","horizonte","modelo"]].itertuples(index=False,name=None))
    if actual != expected or len(validation) != len(expected):
        raise DataValidationError("Validación académica incompleta o duplicada")
    import numpy as np
    values=pd.to_numeric(validation["MAE"],errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any():
        raise DataValidationError("MAE de validación inválido")
    output = (validation.sort_values(["ticker","horizonte","MAE","modelo"],kind="stable")
              .drop_duplicates(["ticker","horizonte"])
              [["ticker","horizonte","modelo","MAE"]]
              .rename(columns={"MAE":"MAE_validacion"})
              .reset_index(drop=True))
    if output["modelo"].isin(("Ingenuo",)).any() or len(output) != 10:
        raise DataValidationError("Selección académica inválida")
    return output


def study_chart(raw:pd.DataFrame, ticker:str, horizon:int):
    """Crea la gráfica principal exclusivamente con los dos modelos de estudio."""
    import matplotlib.pyplot as plt
    frame=plot_frame(raw,ticker,horizon)
    fig,ax=plt.subplots(figsize=(9,3.9),layout="constrained")
    for name,color in (("Precio real","#111827"),
                       (STUDY_MODELS[0],"#2563eb"),
                       (STUDY_MODELS[1],"#e07a2b")):
        ax.plot(pd.to_datetime(frame.index),frame[name],label=name,
                color=color,linewidth=1.6 if name!="Precio real" else 2)
    ax.set(title=f"{ticker} · Horizonte {horizon} sesión(es) · prueba retrospectiva",
           xlabel="Fecha objetivo",ylabel="Cierre (USD)")
    ax.legend(loc="best",fontsize=8)
    ax.grid(alpha=0.22)
    return fig


def plot_frame(raw:pd.DataFrame, ticker:str, horizon:int) -> pd.DataFrame:
    if ticker not in TICKERS or horizon not in HORIZONS:
        raise DataValidationError("Activo u horizonte inválidos")
    sample=raw.loc[(raw.ticker==ticker)&(raw.particion=="prueba")&(raw.horizonte==horizon)]
    if sample.empty:
        raise DataValidationError("Sin resultados de prueba")
    sample=sample.loc[sample.modelo.isin(STUDY_MODELS)]
    wide=sample.pivot(index="fecha_objetivo",columns="modelo",values="precio_predicho")
    real=sample.groupby("fecha_objetivo")["precio_real"].agg(["min","max"])
    if not (real["min"]-real["max"]).abs().le(1e-8).all():
        raise DataValidationError("El cierre real difiere entre modelos para una fecha")
    wide.insert(0,"Precio real",real["min"])
    if set(wide.columns) != {"Precio real",*STUDY_MODELS}:
        raise DataValidationError("Faltan modelos principales")
    return wide.sort_index()


def integral_prediction_analysis(summary: pd.DataFrame,
                                 selection: pd.DataFrame) -> dict[str, Any]:
    """Sintetiza únicamente los dos métodos de estudio y la prueba retrospectiva.

    No combina errores de activos con escalas diferentes, ni vuelve a seleccionar
    modelos a partir de la partición de prueba.
    """
    expected = select_study_models(summary)
    if not selection.reset_index(drop=True).equals(expected):
        raise DataValidationError("La síntesis requiere selección académica verificada")
    rows: list[dict[str, Any]] = []
    counts = {model: 0 for model in STUDY_MODELS}
    longer_error = 0
    for ticker in TICKERS:
        per_horizon: dict[int, dict[str, Any]] = {}
        for h in HORIZONS:
            chosen = selection.loc[(selection.ticker == ticker) &
                                   (selection.horizonte == h)].iloc[0]
            test = summary.loc[(summary.ticker == ticker) &
                               (summary.horizonte == h) &
                               (summary.particion == "prueba") &
                               (summary.modelo == chosen.modelo)]
            if len(test) != 1:
                raise DataValidationError(f"Prueba ausente o duplicada: {ticker} h={h}")
            mae = float(test.iloc[0]["MAE"])
            if not math.isfinite(mae) or mae < 0:
                raise DataValidationError(f"MAE de prueba inválido: {ticker} h={h}")
            counts[str(chosen.modelo)] += 1
            per_horizon[h] = {"modelo": str(chosen.modelo), "MAE": mae}
        if per_horizon[5]["MAE"] > per_horizon[1]["MAE"]:
            longer_error += 1
        rows.append({"ticker": ticker, "h1": per_horizon[1], "h5": per_horizon[5]})
    if len(rows) != 5 or sum(counts.values()) != 10:
        raise DataValidationError("Cobertura insuficiente para análisis integral")
    return {"empresas": rows, "selecciones": counts,
            "error_mayor_h5": longer_error, "escenarios": 10}


def export_all(run:Path, results_root:Path,
               raw:pd.DataFrame, summary:pd.DataFrame, selection:pd.DataFrame,
               metadata:dict[str,Any]) -> tuple[bytes,bytes]:
    """Genera desde una ejecución validada el ZIP y el Informe de Predicción."""
    verify_run(run,results_root)
    expected_selection=select_study_models(summary)
    if not selection.reset_index(drop=True).equals(expected_selection):
        raise DataValidationError("La selección académica no coincide con las métricas originales")
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
    from reportlab.lib.units import cm
    styles=getSampleStyleSheet()
    pdfio=io.BytesIO()
    pdf=SimpleDocTemplate(pdfio,pagesize=A4,topMargin=1.7*cm,bottomMargin=1.6*cm,
                          leftMargin=1.7*cm,rightMargin=1.7*cm,
                          title="TEMPUS PREDICTIO PROTO - Informe de Prediccion")
    story=[
        Paragraph("TEMPUS PREDICTIO PROTO",styles["Title"]),
        Paragraph("10 · Informe de Predicción",styles["Heading1"]),
        Paragraph("Práctica 2 | Box-Jenkins (ARIMA) y Holt-Winters",styles["Normal"]),
        Spacer(1,12),
        Paragraph("Objetivo y alcance",styles["Heading2"]),
        Paragraph("Comparar pronósticos históricos de cinco series bursátiles entre 2016-10-31 y "
                  "2026-10-30, usando ARIMA(1,1,0), suavizado exponencial con tendencia amortiguada "
                  "y un pronóstico ingenuo como referencia.",styles["BodyText"]),
        Spacer(1,8),
        Paragraph("Metodología",styles["Heading2"]),
        Paragraph("Particiones cronológicas 70/15/15; predicciones a 1 y 5 sesiones desde hasta "
                  "24 orígenes por partición. En cada origen se usan exclusivamente datos previos "
                  "o contemporáneos. El menor MAE de validación selecciona únicamente entre ARIMA "
                  "y Holt-Winters. El ingenuo es una línea base no elegible; la prueba es independiente.",
                  styles["BodyText"]),
        Spacer(1,8),
        Paragraph("Advertencia de convergencia y limitaciones",styles["Heading2"]),
        Paragraph("Algunos ajustes ARIMA pueden emitir avisos de no convergencia. El experimento "
                  "original no guarda el estado del optimizador por origen; por ello, NO se puede "
                  "certificar convergencia de cada ajuste con estos resultados. "
                  "Los orígenes están muestreados, no representan todas las sesiones. "
                  "Holt-Winters se emplea sin estacionalidad forzada. Los resultados no son "
                  "pronósticos operativos ni recomendaciones de inversión.",styles["BodyText"]),
        Spacer(1,12),
    ]
    headers=["Activo","h","Modelo principal","MAE principal","MAE referencia","RMSE principal"]
    table_data=[headers]
    for t in TICKERS:
        for h in HORIZONS:
            selected=selection.loc[(selection.ticker==t)&(selection.horizonte==h)].iloc[0]
            selected_test=summary.loc[(summary.ticker==t)&(summary.horizonte==h)&
                (summary.particion=="prueba")&(summary.modelo==selected.modelo)].iloc[0]
            naive=summary.loc[(summary.ticker==t)&(summary.horizonte==h)&
                (summary.particion=="prueba")&(summary.modelo=="Ingenuo")].iloc[0]
            label={"ARIMA(1,1,0)":"ARIMA","Holt-Winters aditivo sin estacionalidad":"Holt-Winters"}[selected.modelo]
            table_data.append([t,str(h),label,f"{selected_test.MAE:.3f}",f"{naive.MAE:.3f}",f"{selected_test.RMSE:.3f}"])
    tab=Table(table_data,repeatRows=1,colWidths=[1.55*cm,0.7*cm,3.5*cm,2.6*cm,2.6*cm,2.7*cm])
    tab.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfeaf2")),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#c6d0d6")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f8fa")]),
    ]))
    story.extend([Paragraph("Resultados fuera de muestra",styles["Heading2"]),tab,
                  Spacer(1,12),
                  Paragraph("Interpretación",styles["Heading2"]),
                  Paragraph("Una menor MAE indica menor error promedio en los orígenes evaluados, "
                            "pero no garantiza superioridad estadística ni rentabilidad. "
                            "Las métricas se calculan sobre los mismos objetivos dentro de cada "
                            "activo y horizonte; la comparación con los experimentos previos "
                            "requiere alinear orígenes.",styles["BodyText"]),PageBreak()])
    for ticker in TICKERS:
        story.append(Paragraph(f"{ticker}: comparación gráfica y resultados",styles["Heading1"]))
        for h in HORIZONS:
            fig=study_chart(raw,ticker,h)
            chart_buffer=io.BytesIO()
            fig.savefig(chart_buffer,format="png",dpi=135,bbox_inches="tight")
            plt.close(fig)
            chart_buffer.seek(0)
            figure=Image(chart_buffer,width=16.3*cm,height=7.1*cm)
            story.append(Paragraph(f"Horizonte: {h} sesión(es) bursátil(es)",styles["Heading3"]))
            story.append(figure)
            chosen=selection.loc[(selection.ticker==ticker)&(selection.horizonte==h)].iloc[0]
            subset=summary.loc[(summary.ticker==ticker)&(summary.horizonte==h)&
                               (summary.particion=="prueba")]
            chosen_test=subset.loc[subset.modelo==chosen.modelo].iloc[0]
            baseline=subset.loc[subset.modelo=="Ingenuo"].iloc[0]
            story.append(Paragraph(
                f"Modelo principal elegido por validación: {chosen.modelo}. "
                f"MAE en prueba: {chosen_test.MAE:.3f} USD. "
                f"Referencia técnica ingenua: MAE {baseline.MAE:.3f} USD (no elegible). "
                "Las diferencias no constituyen prueba de superioridad estadística.",
                styles["BodyText"]))
            story.append(Spacer(1,7))
        story.append(PageBreak())
    integral = integral_prediction_analysis(summary, selection)
    def _short_model(name: str) -> str:
        return "ARIMA" if name == "ARIMA(1,1,0)" else "Holt-Winters"

    integrated_rows = [[
        "Empresa", "Modelo · h=1", "MAE h=1 (USD)",
        "Modelo · h=5", "MAE h=5 (USD)"
    ]]
    for result in integral["empresas"]:
        integrated_rows.append([
            result["ticker"],
            _short_model(result["h1"]["modelo"]),
            f'{result["h1"]["MAE"]:.3f}',
            _short_model(result["h5"]["modelo"]),
            f'{result["h5"]["MAE"]:.3f}',
        ])
    integrated_table = Table(
        integrated_rows, repeatRows=1,
        colWidths=[2.1*cm, 3.5*cm, 3.0*cm, 3.5*cm, 3.0*cm]
    )
    integrated_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dfeaf2")),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#c6d0d6")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#f5f8fa")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    count_arima = integral["selecciones"]["ARIMA(1,1,0)"]
    count_holt = integral["selecciones"]["Holt-Winters aditivo sin estacionalidad"]
    story.extend([
        Paragraph("Análisis Integral de Predicción de las Cinco Empresas",
                  styles["Heading1"]),
        Paragraph(
            "Síntesis de los resultados retrospectivos de JNJ, MSFT, JPM, WMT "
            "y XOM. Se comparan únicamente las predicciones de ARIMA(1,1,0) "
            "y Holt-Winters con tendencia amortiguada, para horizontes "
            "de 1 y 5 sesiones bursátiles. Cada modelo principal se eligió "
            "por MAE de validación antes de medir su error en prueba.",
            styles["BodyText"]),
        Spacer(1, 10),
        integrated_table,
        Spacer(1, 13),
        Paragraph("Lectura comparativa", styles["Heading2"]),
        Paragraph(
            f"En los {integral['escenarios']} escenarios evaluados, "
            f"Holt-Winters fue elegido {count_holt} veces y ARIMA "
            f"{count_arima} veces. Estos recuentos describen el criterio "
            "de selección en validación y no constituyen un ranking "
            "de rentabilidad ni una demostración de superioridad estadística.",
            styles["BodyText"]),
        Spacer(1, 7),
        Paragraph(
            f"El MAE de prueba a 5 sesiones supera al de 1 sesión en "
            f"{integral['error_mayor_h5']} de las cinco empresas. "
            "La frecuencia describe únicamente los resultados retrospectivos "
            "de estos activos y no garantiza un patrón fuera de muestra.",
            styles["BodyText"]),
        Spacer(1, 7),
        Paragraph(
            "La magnitud del MAE depende también de la escala de precios "
            "de cada activo. Por ello, no corresponde ordenar las cinco "
            "empresas por MAE en dólares como si compartieran la misma "
            "dificultad predictiva. La comparación más directa es entre "
            "ARIMA y Holt-Winters para una misma empresa y horizonte.",
            styles["BodyText"]),
        Spacer(1, 7),
        Paragraph(
            "En conjunto, las series muestran que ambos métodos pueden "
            "aproximar la evolución histórica del cierre, pero su utilidad "
            "predictiva depende del activo, del horizonte y del período "
            "evaluado. El análisis es retrospectivo y no equivale "
            "a un pronóstico operativo del siguiente cierre.",
            styles["BodyText"]),
        Spacer(1, 17),
    ])
    # La valoración sigue a TODOS los resultados y a su síntesis metodológica.
    # Es retrospectiva: los precios previstos tienen objetivos históricos ya observados.
    from predictio_economia.lectura import prediction_economic_reading
    economic = prediction_economic_reading(raw, selection)
    economic_table = [["Empresa", "h", "Modelo", "Previsto USD", "Real USD", "Error USD"]]
    for item in economic:
        economic_table.append([
            item["activo"], str(item["horizonte"]), item["modelo"],
            f'{item["predicho_usd"]:.2f}', f'{item["real_usd"]:.2f}',
            f'{item["error_firmado_usd"]:+.2f}',
        ])
    tbl = Table(economic_table, repeatRows=1,
                colWidths=[2.1*cm,0.9*cm,3.1*cm,3.1*cm,3.1*cm,3.0*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfeaf2")),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#c6d0d6")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white,colors.HexColor("#f5f8fa")]),
    ]))
    story.extend([
        PageBreak(),
        Paragraph("Valoración económica posterior a los resultados",styles["Heading1"]),
        Paragraph("Lectura de cinco empresas y dos horizontes: último origen de "
                  "PRUEBA retrospectiva por escenario; no son cotizaciones actuales "
                  "ni previsiones posteriores al cierre del conjunto de datos.",
                  styles["BodyText"]),
        Spacer(1,9),
        tbl,
        Spacer(1,11),
        Paragraph("Análisis económico por empresa", styles["Heading2"]),
    ])
    from predictio_economia.lectura import prediction_company_interpretations
    for company in prediction_company_interpretations(economic):
        story.append(Paragraph(
            f"<b>{company['activo']}.</b> {company['texto']}",
            styles["BodyText"]
        ))
        story.append(Spacer(1,5))
    story.extend([
        Spacer(1,5),
        Paragraph("Significado económico de los datos",styles["Heading2"]),
        Paragraph("La columna Error USD equivale a precio previsto menos precio "
                  "real observado, por acción, en el último origen de prueba. "
                  "Un signo positivo indica sobreestimación y uno negativo "
                  "subestimación; NO corresponde a una ganancia o pérdida "
                  "de cartera. El MAE anterior resume la magnitud de errores "
                  "sobre varios orígenes históricos.",
                  styles["BodyText"]),
        Spacer(1,7),
        Paragraph("Óptica de quien ya posee acciones",styles["Heading2"]),
        Paragraph("La plusvalía o minusvalía personal exige el precio de compra "
                  "y las cantidades poseídas; esos datos no están disponibles "
                  "en este experimento. La recepción de dividendos depende de "
                  "la titularidad en las fechas aplicables y de distribuciones "
                  "corporativas verificadas. No hay datos auditados de "
                  "dividendos en estos resultados, por lo que no es posible "
                  "cuantificar importes cobrados o dejados de percibir.",
                  styles["BodyText"]),
        Spacer(1,7),
        Paragraph("Óptica de quien aún no posee acciones",styles["Heading2"]),
        Paragraph("Los pronósticos retrospectivos permiten evaluar errores, "
                  "pero por sí solos no justifican comprar o abstenerse. "
                  "No se conoce un precio de ejecución actual, una "
                  "distribución futura confirmada de dividendos ni "
                  "los costos de negociación. No se calcula ni atribuye "
                  "rentabilidad total, retorno neto ni una recomendación "
                  "operativa para JNJ, MSFT, JPM, WMT o XOM.",
                  styles["BodyText"]),
        Spacer(1,7),
        Paragraph("Alcance",styles["Heading2"]),
        Paragraph("La interpretación económica se deriva exclusivamente de "
                  "los datos de predicción y prueba anteriores. "
                  "ARIMA y Holt-Winters estiman precios de cierre, "
                  "no dividendos, y no demuestran por sí mismos capacidad "
                  "para generar rendimientos positivos.",styles["BodyText"]),
        Spacer(1,17),
        Paragraph("Conclusiones y trazabilidad",styles["Heading2"]),
        Paragraph("El módulo muestra implementaciones retrospectivas de Box-Jenkins y "
                  "Holt-Winters con una referencia ingenua y verificación SHA-256. "
                  "Los resultados reflejan una muestra acotada de orígenes de prueba, "
                  "sin promesa de generalización. La selección se efectuó en validación. "
                  "Consultar el ZIP anexo para los pronósticos por fecha y métricas completas.",
                  styles["BodyText"]),
        Spacer(1,12),
        Paragraph("Dependencia técnica: Python, pandas, NumPy, statsmodels y Matplotlib. "
                  "Documento generado automáticamente a partir de resultados auditados.",
                  styles["BodyText"]),
    ])
    pdf.build(story)
    pdf_data=pdfio.getvalue()
    if not pdf_data.startswith(b"%PDF") or len(pdf_data)<1500:
        raise DataValidationError("No se pudo crear un PDF válido")

    from predictio_academico.protocolo import formalizar_pdf
    pdf_data=formalizar_pdf(pdf_data, "prediccion")

    package=io.BytesIO()
    content={}
    # Evidencia histórica SIN MODIFICACIÓN: conserva selección original entre tres.
    for filename in ARTIFACTS:
        content[f"auditoria_original/{filename}"]=(Path(run)/filename).read_bytes()
    content["auditoria_original/manifest_sha256.json"]=(Path(run)/"manifest_sha256.json").read_bytes()
    content["seleccion_academica/seleccion_ARIMA_HOLT.csv"]=selection.to_csv(index=False).encode("utf-8-sig")
    # Archivo derivado, no sustituye la auditoría de origen.
    content["valoracion_economica/comparacion_pronostico_real.csv"] = (
        pd.DataFrame(economic).to_csv(index=False).encode("utf-8-sig")
    )
    content["seleccion_academica/metodologia_seleccion.json"]=json.dumps({
        "metodo":"MAE mínimo en validación entre ARIMA y Holt-Winters",
        "modelos_elegibles":list(STUDY_MODELS),
        "linea_base_no_elegible":"Ingenuo",
        "no_reentrenamiento":True,
        "datos_historicos_inmutables":True,
    },ensure_ascii=False,indent=2).encode("utf-8")
    for t in TICKERS:
        for h in HORIZONS:
            fig=study_chart(raw,t,h)
            chart_buffer=io.BytesIO()
            fig.savefig(chart_buffer,format="png",dpi=135,bbox_inches="tight")
            plt.close(fig)
            content[f"graficas_modelos_principales/{t}_h{h}.png"]=chart_buffer.getvalue()
    content["INFORME_PREDICCION.pdf"]=pdf_data
    content["LEEME.txt"]=(
        "TEMPUS PREDICTIO PROTO - MODULO 10\n"
        "Resultados historicos ARIMA / Holt-Winters / Ingenuo\n"
        "Cinco activos, horizontes 1 y 5 sesiones. "
        "Incluye CSV, graficas, PDF y hashes.\n"
        "No es un pronostico en tiempo real ni asesoria financiera.\n"
        "Convergencia ARIMA por origen: no registrada en la version del modelo.\n"
    ).encode("utf-8")
    checks={key:digest(value) for key,value in content.items()}
    content["MANIFIESTO_DESCARGA_SHA256.json"]=json.dumps(
        {"contrato":"predictio-proto-entregable/1.0","archivos":checks},
        ensure_ascii=False,indent=2).encode("utf-8")
    with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6,allowZip64=True) as out:
        for filename,value in sorted(content.items()):
            out.writestr(f"TEMPUS_PREDICTIO_PROTO_10/{filename}",value)
    LOGGER.info("Informe de predicción creado: %d bytes; ZIP: %d bytes",
                len(pdf_data),len(package.getvalue()))
    return package.getvalue(),pdf_data
