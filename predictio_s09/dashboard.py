"""Navegación explícita por los procedimientos académicos de la Práctica 2.

La aplicación solo lee artefactos auditados ya existentes. NO entrena, descarga
cotizaciones ni sirve pronósticos futuros.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from predictio_visual07.charts import validation_chart, test_predictions_chart, error_chart, prepare_test_predictions, prepare_error_series
from predictio_s09.normality import histogram_figure, qq_figure, normality_summary
from predictio_entregables.export import build_deliverables
from predictio_s09.estetica import apply_style, render_project_information
from predictio_s07.integration import (
    HORIZONS, SECTIONS, SYMBOLS, ValidationError, academic_summary,
    read_csv, safe_path, select_latest_complete, validate_artifacts,
)

PURPOSES = (
    "comprobar si la distribución de los log-retornos es compatible con la normal teórica "
    "mediante histogramas, curva de referencia y contrastes.",
    "localizar fluctuaciones excepcionalmente grandes sin borrar eventos financieros reales.",
    "visualizar el ruido de corto plazo y distinguirlo de cambios persistentes; "
    "comparar el precio con medias móviles causales.",
    "separar descriptivamente tendencia, oscilación de cinco sesiones y residuo mediante STL.",
    "estimar dirección y magnitud de la tendencia descriptiva mediante mínimos cuadrados.",
    "explorar frecuencias dominantes mediante FFT y periodograma, sin asumir ciclos estables.",
    "comparar modelos con validación temporal y reserva estricta del conjunto test.",
    "evaluar precisión y calibración fuera de muestra frente al modelo ingenuo.",
    "descargar las evidencias completas y un informe académico final para analizar los resultados.",
    "comparar Box-Jenkins (ARIMA) y Holt-Winters con el modelo ingenuo, "
    "y descargar datos e Informe de Predicción.",
)
MENU = SECTIONS + ("09 · Entregables", "10 · Predictio Proto")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados"

st.set_page_config(page_title="TEMPUS PREDICTIO PROTO | Práctica 2", page_icon="📈", layout="wide")
apply_style(st)
st.title("TEMPUS PREDICTIO PROTO")
st.caption("Plataforma TEMPUS DECISIO PROTO · Práctica 2")
st.markdown("**Análisis de series bursátiles y evaluación de predicción temporal.**")

@st.cache_data(show_spinner=False)
def cached_csv(path: str, version: int, required: tuple[str,...]) -> pd.DataFrame:
    """Invalida caché cuando el archivo local cambia."""
    del version
    return pd.read_csv(path, usecols=None).pipe(
        lambda df: df if set(required).issubset(df.columns)
        else (_ for _ in ()).throw(ValidationError("Columnas faltantes"))
    )

@st.cache_data(show_spinner=False)
def cached_image(path: str, version: int) -> bytes:
    del version
    return Path(path).read_bytes()

def table(sprint: int, filename: str, required: tuple[str,...] = ()) -> pd.DataFrame:
    file=safe_path(RESULTS, str((folders[sprint]/filename).relative_to(RESULTS)))
    frame=cached_csv(str(file),file.stat().st_mtime_ns,required)
    if frame.empty:
        raise ValidationError(f"CSV sin registros: {filename}")
    return frame

def plot(sprint: int, filename: str, caption: str) -> None:
    file=safe_path(RESULTS, str((folders[sprint]/filename).relative_to(RESULTS)))
    st.image(cached_image(str(file),file.stat().st_mtime_ns),caption=caption,width="stretch")

def csv_download(frame:pd.DataFrame, filename:str, label:str) -> None:
    st.download_button(label,frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename,mime="text/csv",key=filename)

try:
    folders,reports=select_latest_complete(RESULTS)
    count=validate_artifacts(RESULTS,folders)
except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
    st.error(f"No están disponibles los resultados integrados: {exc}")
    st.info("Genere primero los reportes históricos requeridos en este proyecto, "
            "y después vuelva a iniciar Streamlit.")
    st.stop()

with st.sidebar:
    logo_file = ROOT / "assets" / "tempus-logo.png"
    if logo_file.is_file():
        st.image(str(logo_file), width=155)
    st.markdown("### TEMPUS PREDICTIO PROTO")
    st.caption("TEMPUS DECISIO PROTO · Módulo II")
    st.write("Análisis y predicción de acciones bursátiles como series temporales cuantitativas.")
    st.divider()
    st.caption("SECCIONES")
    main_area = st.radio(
        "Área de navegación",
        ("Análisis financiero", "Información del proyecto"),
        key="predictio_main_area",
        label_visibility="collapsed",
    )
    if main_area == "Análisis financiero":
        st.divider()
        st.caption("ANÁLISIS FINANCIERO")
        section = st.radio("Procedimiento", MENU, index=0, key="predictio_procedure")
        st.divider()
        st.caption("SERIE FINANCIERA")
        symbol = st.selectbox("Empresa", SYMBOLS, index=0)
        horizon = st.selectbox("Horizonte de prueba (sesiones)", HORIZONS, index=0)
        st.caption("Cobertura: 2016-10-31 a 2026-10-30 · 2.514 sesiones por activo")
    else:
        st.divider()
        st.caption("INFORMACIÓN DEL PROYECTO")
        information_page = st.radio(
            "Página informativa",
            ("Acerca de", "Dataset", "Fundamentos", "Ayuda", "Institucional", "Tecnologías"),
            key="predictio_information_page",
            label_visibility="collapsed",
        )
    st.divider()
    st.caption("Práctica 2 · Evaluación histórica reproducible")

if main_area == "Información del proyecto":
    render_project_information(st, information_page)
    st.stop()

st.header("Análisis financiero")
st.caption("Cinco series históricas · Diez procedimientos de análisis y predicción")
st.info("Cobertura oficial: **10 años (2016-10-31 a 2026-10-30)** · "
        "**2.514 sesiones por empresa** · **12.570 registros históricos en total**.")

try:
    r=academic_summary(reports,symbol,horizon)
    st.subheader(section)
    st.markdown("**Propósito del procedimiento:** " + PURPOSES[MENU.index(section)])
    if section==SECTIONS[0]:
        st.write("Se examina la normalidad de los **log-retornos**, no la normalidad de los precios.")
        n=r["normality"]
        a,b,c=st.columns(3)
        a.metric("D'Agostino · p",f"{n['dagostino_p']:.2e}")
        b.metric("Jarque–Bera · p",f"{n['jarque_bera_p']:.2e}")
        c.metric("Curtosis excedente",f"{n['excess_kurtosis']:.2f}")
        st.markdown("### Histograma de log-retornos y curva normal teórica")
        st.write("**¿Para qué sirve?** Permite ver la frecuencia relativa de los cambios "
                 "diarios (densidad) y contrastar visualmente las colas y la asimetría "
                 "con una distribución normal estimada usando los mismos retornos.")
        normal_frame=table(2,f"{symbol}_transformado.csv",("Date","LogReturn"))
        returns=pd.to_numeric(normal_frame["LogReturn"],errors="raise").to_numpy()
        summary=normality_summary(returns)
        if summary["n"] != int(n["n"]):
            raise ValidationError("El histograma no coincide con el número de retornos del reporte S02")
        if abs(summary["dagostino_p"]-float(n["dagostino_p"]))>1e-8 or abs(summary["jarque_bera_p"]-float(n["jarque_bera_p"]))>1e-8:
            raise ValidationError("El histograma no coincide con los contrastes del reporte auditado")
        fig=histogram_figure(returns,symbol)
        st.pyplot(fig,clear_figure=False,width="stretch")
        import matplotlib.pyplot as plt
        plt.close(fig)
        st.caption("Barras azules: densidad empírica. Línea roja: normal teórica. "
                   "Desviaciones importantes en las colas sugieren eventos extremos "
                   "más frecuentes de lo esperado bajo normalidad.")
        st.markdown("### Gráfico Q-Q complementario")
        qq=qq_figure(returns,symbol)
        st.pyplot(qq,clear_figure=False,width="stretch")
        plt.close(qq)
        rejected=bool(float(n["dagostino_p"])<0.05 or float(n["jarque_bera_p"])<0.05)
        st.info(("**Conclusión: se rechaza la normalidad** de los log-retornos " if rejected
                 else "**Conclusión: no se rechaza la normalidad** de los log-retornos ")
                + "al 5 % con al menos uno de los contrastes indicados. "
                "El histograma es descriptivo y no sustituye las pruebas formales. "
                "Rechazar la normalidad no implica imposibilidad de pronóstico.")
        with st.expander("Ver gráfico original auditado"):
            plot(2,f"{symbol}_normalidad.png",
                 "Gráfico histórico conjunto: histograma y Q-Q")

    elif section==SECTIONS[1]:
        frame=table(2,f"{symbol}_transformado.csv",
                    ("Date","Close","LogReturn","OutlierMAD"))
        flagged=frame.loc[frame["OutlierMAD"].astype(str).str.lower().isin(("true","1","1.0"))]
        a,b=st.columns(2)
        a.metric("Retornos señalados (MAD)",str(r["outliers"]))
        b.metric("Proporción",f"{100*r['outliers']/max(len(frame)-1,1):.2f} %")
        st.line_chart(frame.set_index("Date")[["LogReturn"]])
        st.caption("Las fluctuaciones señaladas no se eliminan automáticamente: "
                   "reemplazarlas podría ocultar caídas o subidas reales.")
        st.write("**Fechas con valores extremos detectados**")
        st.dataframe(flagged[["Date","Close","LogReturn"]],hide_index=True,width="stretch")
        st.info("Los atípicos se señalan, pero no se eliminan ni interpolan: "
                "pueden reflejar acontecimientos financieros reales.")
        csv_download(flagged[["Date","Close","LogReturn"]],
                     f"{symbol}_outliers_s07.csv","Descargar observaciones señaladas")

    elif section==SECTIONS[2]:
        frame=table(2,f"{symbol}_transformado.csv",
                    ("Date","Close","MA5","MA20","MA50"))
        plot(2,f"{symbol}_suavizado.png","Cierre y medias móviles retrospectivas")
        st.line_chart(frame.set_index("Date")[["Close","MA5","MA20","MA50"]])
        st.caption("Cuanto mayor es la ventana, más lisa suele verse la curva y mayor es su retardo.")
        st.info("MA5, MA20 y MA50 se calculan con observaciones actuales y anteriores. "
                "Suavizar una serie no hace que sus residuos sean necesariamente normales.")
        csv_download(frame[["Date","Close","MA5","MA20","MA50"]],
                     f"{symbol}_medias_moviles_s07.csv","Descargar datos de suavizado")

    elif section==SECTIONS[3]:
        st.metric("Fuerza periódica STL, 5 sesiones",
                  f"{r['stl']['seasonal_strength_exploratory']:.3f}")
        plot(3,f"{symbol}_descomposicion.png",
             "STL robusta: log-retorno, tendencia, componente periódica y residuo")
        st.info("El período de cinco sesiones es una hipótesis exploratoria; "
                "la descomposición se calculó sobre la muestra completa, sin "
                "uso predictivo. No equivale a probar estacionalidad estable.")
        data=table(3,f"{symbol}_componentes.csv",
                   ("Date","LogReturn","STLTrend","STLSeasonal","STLResidual"))
        csv_download(data,f"{symbol}_STL_s07.csv","Descargar componentes STL")

    elif section==SECTIONS[4]:
        t=r["trend"]
        a,b,c=st.columns(3)
        a.metric("R² descriptivo",f"{t['r_squared']:.3f}")
        b.metric("Pendiente log/sesión",f"{t['slope_log_per_session']:.6f}")
        c.metric("Tendencia log anualizada",f"{t['annualized_log_trend_pct']:.2f} %")
        plot(3,f"{symbol}_tendencia.png",
             "Regresión OLS sobre log-precios de la muestra completa")
        st.info("Una tendencia ascendente y un R² alto en muestra NO "
                "demuestran capacidad predictiva fuera de muestra.")

    elif section==SECTIONS[5]:
        st.write("Transformada rápida de Fourier (FFT) sobre log-retornos "
                 "centrados; ventana Hann y periodograma.")
        plot(3,f"{symbol}_periodograma.png","Potencia espectral por período en sesiones")
        peaks=pd.DataFrame(r["peaks"])
        st.write("**Picos espectrales dominantes (exploratorios)**")
        st.dataframe(peaks,hide_index=True,width="stretch")
        st.info("Los picos espectrales son candidatos a ciclos, no evidencia "
                "suficiente de periodicidad persistente o predictibilidad.")
        csv_download(peaks,f"{symbol}_picos_Fourier_s07.csv","Descargar picos espectrales")

    elif section==SECTIONS[6]:
        item=reports[4]["symbols"][symbol][str(horizon)]
        st.metric("Modelo elegido en VALIDACIÓN",r["selected_model"])
        st.caption("Selección por MAE en validación; test reservado; la evaluación walk-forward aplica embargo h−1.")
        validation=pd.DataFrame([{
            "Modelo":name, "MAE validación":round(float(values["mae"]),4),
            "RMSE":round(float(values["rmse"]),4),
            "sMAPE %":round(float(values["smape_pct"]),3),
        } for name,values in item["validation_metrics"].items()])
        st.dataframe(validation,hide_index=True,width="stretch")
        st.markdown("### Comparación de candidatos en validación")
        st.caption("Cada barra representa el MAE obtenido en VALIDACIÓN; menor es mejor. "
                   "Se incluyen ingenuo, Drift20 y las variantes AR(1). "
                   "El conjunto de prueba no se utiliza para escoger el modelo.")
        vfig=validation_chart(validation,r["selected_model"])
        st.pyplot(vfig,clear_figure=False,width="stretch")
        import matplotlib.pyplot as plt
        plt.close(vfig)
        st.markdown("### Precio real frente a pronósticos históricos")
        st.caption("Solo se grafican trayectorias realmente almacenadas: modelo seleccionado "
                   "y referencia ingenua. Los otros candidatos se comparan en VALIDACIÓN, "
                   "no se inventan pronósticos individuales del conjunto de prueba.")
        test_frame=table(5,f"{symbol}_h{horizon}_evaluacion.csv",
                         ("origin_date","target_date","actual_close",
                          "predicted_close","naive_predicted_close","selected_model"))
        trajectories=prepare_test_predictions(test_frame,horizon,r["selected_model"])
        pfig=test_predictions_chart(trajectories,symbol,horizon)
        st.pyplot(pfig,clear_figure=False,width="stretch")
        plt.close(pfig)
        csv_download(trajectories,
                     f"{symbol}_h{horizon}_comparacion_pronosticos.csv",
                     "Descargar series reales y pronosticadas")
        st.write("**Predicciones históricas del conjunto de prueba**")
        frame=table(4,f"{symbol}_h{horizon}_test.csv")
        st.dataframe(frame.tail(30),hide_index=True,width="stretch")
        csv_download(frame,f"{symbol}_h{horizon}_modelos_s07.csv",
                     "Descargar predicciones históricas")
        st.info("Naive, Drift20 y AR(1) se comparan históricamente. "
                "No son señales operativas de compra o venta.")

    elif section==SECTIONS[7]:
        item5=reports[5]["symbols"][symbol][str(horizon)]
        item6=reports[6]["symbols"][symbol][str(horizon)]
        a,b,c,d=st.columns(4)
        a.metric("MAE walk-forward",f"{r['mae']:.3f}")
        b.metric("MAE ingenuo",f"{r['naive_mae']:.3f}")
        c.metric("Cobertura banda fija",f"{100*r['coverage_fixed']:.2f} %")
        d.metric("Cobertura banda móvil",f"{100*r['coverage_rolling']:.2f} %")
        st.caption("Bandas nominales del 90 %, sin garantía de cobertura futura.")
        frame=table(6,f"{symbol}_h{horizon}_calibracion.csv",
                    ("target_date","actual_close","predicted_close",
                     "lower_close","upper_close","rolling_lower_close",
                     "rolling_upper_close"))
        plot(6,f"{symbol}_h{horizon}_calibracion.png",
             "Datos reales, pronóstico histórico y bandas fija/móvil")
        st.markdown("### Evolución del error absoluto")
        st.caption("Los valores altos identifican sesiones con mayor desviación "
                   "del modelo seleccionado; la serie corresponde al conjunto de prueba.")
        errors=prepare_error_series(frame,horizon)
        efig=error_chart(errors,symbol,horizon)
        st.pyplot(efig,clear_figure=False,width="stretch")
        import matplotlib.pyplot as plt
        plt.close(efig)
        st.write("**Cobertura por subperíodo**")
        by_quarter=pd.DataFrame(item6["rolling_by_quarter"])
        by_quarter["Cobertura móvil (%)"]=(by_quarter["coverage"]*100).round(2)
        st.dataframe(by_quarter[["quarter","start_date","end_date",
                                 "Cobertura móvil (%)"]],hide_index=True,width="stretch")
        st.write("**Comparación de errores con estimador HAC (exploratorio)**")
        st.json(item5["error_comparison_hac_exploratory"])
        csv_download(frame,f"{symbol}_h{horizon}_evaluacion_s07.csv",
                     "Descargar evaluación de incertidumbre")
        st.info("La cobertura retrospectiva puede diferir del valor nominal. "
                "Las comparaciones HAC son exploratorias, con dependencia "
                "temporal y pruebas múltiples.")

    elif section == MENU[8]:
        st.markdown("### Descargas de la Práctica 2")
        st.write("El paquete incluye **todos los CSV, PNG, JSON y documentos generados** "
                 "en las ejecuciones históricas validadas, más histogramas independientes, "
                 "gráficas de modelos, errores y tablas comparativas.")
        st.write("El informe PDF presenta **metodología, resultados de los cinco activos, "
                 "gráficas por procedimiento, evaluación de los diez experimentos, "
                 "interpretación y limitaciones**.")
        @st.cache_data(show_spinner=False)
        def cached_deliverables(path: str, signature: tuple) -> tuple[bytes, bytes, dict]:
            del signature
            return build_deliverables(Path(path))
        # La firma permite invalidar el caché si cambian los reportes o archivos.
        files_for_signature = [
            file for folder in folders.values()
            for file in folder.rglob("*") if file.is_file()
        ]
        signature = tuple(sorted(
            (str(file.relative_to(RESULTS)), file.stat().st_size, file.stat().st_mtime_ns)
            for file in files_for_signature
        ))
        with st.spinner("Preparando entregables con datos históricos auditados..."):
            zip_data, pdf_data, meta = cached_deliverables(str(RESULTS), signature)
        a, b, c = st.columns(3)
        a.metric("Archivos en el ZIP", str(meta["archivos"]))
        b.metric("Evidencias verificadas", str(meta["artefactos_auditados"]))
        c.metric("Experimentos", str(meta["experimentos"]))
        st.download_button(
            "Descargar todos los datos y resultados (ZIP)",
            zip_data,
            file_name="TEMPUS_PREDICTIO_PROTO_PRACTICA_2_ENTREGABLES.zip",
            mime="application/zip",
            width="stretch",
            key="descargar_evidencias",
        )
        st.download_button(
            "Descargar informe académico final (PDF)",
            pdf_data,
            file_name="TEMPUS_PREDICTIO_PROTO_INFORME_FINAL_PRACTICA_2.pdf",
            mime="application/pdf",
            width="stretch",
            key="descargar_informe",
        )
        st.info("Los archivos corresponden a análisis y backtesting retrospectivos. "
                "El informe no constituye una recomendación de inversión.")


    elif section == MENU[9]:
        from predictio_proto10.integration_final import (
            MODEL_NAMES, STUDY_MODELS, select_run, load_run, plot_frame, export_all,
        )
        st.markdown("### Box-Jenkins (ARIMA) y Holt-Winters")
        st.write("**Modelos:** ARIMA(1,1,0) sobre log-precios, "
                 "Holt-Winters con tendencia amortiguada y sin estacionalidad forzada, "
                 "y modelo ingenuo de referencia.")
        st.caption("Se comparan pronósticos retrospectivos de 1 y 5 sesiones. "
                   "Cada origen utiliza solo datos ya disponibles. "
                   "La evaluación usa hasta 24 orígenes muestreados por partición.")
        try:
            run = select_run(RESULTS)
            # Cache controlado por los hashes y marcas de modificación de la ejecución.
            @st.cache_data(show_spinner=False)
            def cached_predictions(folder: str, root: str, signature: tuple):
                del signature
                return load_run(Path(folder), Path(root))
            from predictio_proto10.integration_final import ARTIFACTS
            signature = tuple(sorted(
                (name, (run/name).stat().st_size, (run/name).stat().st_mtime_ns)
                for name in (*ARTIFACTS, "manifest_sha256.json")
            ))
            raw, metricas, elegidos, meta = cached_predictions(
                str(run), str(RESULTS), signature
            )
            selected = elegidos.loc[
                (elegidos.ticker == symbol) & (elegidos.horizonte == horizon)
            ].iloc[0]
            subset = metricas.loc[
                (metricas.ticker == symbol) &
                (metricas.horizonte == horizon) &
                (metricas.particion == "prueba")
            ].copy()
            st.metric("Modelo principal seleccionado en validación", selected.modelo)
            st.markdown("### Precios reales y predicciones en prueba")
            chart = plot_frame(raw, symbol, horizon)
            chart.index = pd.to_datetime(chart.index)
            st.line_chart(chart, x_label="Fecha objetivo",
                          y_label="Precio de cierre (USD)")
            st.caption("Líneas: precio real, ARIMA y Holt-Winters. "
                       "El ingenuo no aparece en la gráfica principal y solo sirve "
                       "como referencia técnica de MAE. No son predicciones en vivo.")
            st.markdown("### Comparación de errores fuera de muestra")
            comparison = subset.loc[subset.modelo.isin(STUDY_MODELS),
                                    ["modelo","n","MAE","RMSE","sesgo"]].sort_values("MAE")
            st.dataframe(comparison, hide_index=True, width="stretch")
            st.bar_chart(comparison.set_index("modelo")[["MAE","RMSE"]])
            test_selected = subset.loc[subset.modelo == selected.modelo].iloc[0]
            naive = subset.loc[subset.modelo == "Ingenuo"].iloc[0]
            c1,c2,c3 = st.columns(3)
            c1.metric("MAE seleccionado", f"{test_selected.MAE:.3f} USD")
            c2.metric("MAE referencia (ingenuo)", f"{naive.MAE:.3f} USD")
            c3.metric("RMSE seleccionado", f"{test_selected.RMSE:.3f} USD")
            st.info(
                "El modelo principal se seleccionó por MAE de VALIDACIÓN únicamente "
                "entre ARIMA y Holt-Winters. El ingenuo es una referencia no elegible. "
                "Una mejora pequeña frente a la referencia no demuestra superioridad "
                "estadística. Las métricas usan 24 orígenes de prueba o menos."
            )
            st.warning(
                "**Avisos de convergencia ARIMA:** durante el ajuste histórico "
                "se observaron advertencias del optimizador. "
                "Los archivos originales no conservan el estado de convergencia "
                "por origen; no se puede certificar qué ajustes convergieron. "
                "Interprete los resultados como prototipo académico."
            )
            st.markdown("### Descargar datos e Informe de Predicción")
            @st.cache_data(show_spinner=False)
            def cached_predictio_exports(folder: str, root: str, signature: tuple):
                del signature
                data, summ, chosen, info = load_run(Path(folder), Path(root))
                return export_all(Path(folder), Path(root), data, summ, chosen, info)
            with st.spinner("Preparando evidencia y reporte desde los archivos auditados..."):
                zip_bytes, pdf_bytes = cached_predictio_exports(
                    str(run), str(RESULTS), signature
                )
            st.download_button(
                "Descargar datos y gráficas de predicción (ZIP)",
                data=zip_bytes, file_name="TEMPUS_PREDICTIO_PROTO_10_DATOS.zip",
                mime="application/zip", key="predictio10_zip", width="stretch"
            )
            st.download_button(
                "Descargar Informe de Predicción (PDF)",
                data=pdf_bytes, file_name="TEMPUS_PREDICTIO_PROTO_10_INFORME.pdf",
                mime="application/pdf", key="predictio10_pdf", width="stretch"
            )
            st.caption("Informe y ZIP elaborados desde la misma ejecución histórica "
                       "con integridad SHA-256 comprobada.")
        except (FileNotFoundError, ValueError, OSError, KeyError,
                TypeError, IndexError, json.JSONDecodeError) as error:
            st.error(f"No están disponibles los resultados de Predictio Proto: {error}")
            st.info("Ejecute antes el modelado clásico con "
                    "ejecutar_predictio_proto10.ps1 y reinicie la aplicación.")

    with st.expander("Procedencia y archivos auditados"):
        st.write("Contrato:",reports[2]["data_contract"])
        st.write("SHA-256 CSV:",reports[2]["data_sha256"])
        st.write("Archivos para todas las secciones verificados:",count)
        st.dataframe(pd.DataFrame([{"Etapa de análisis":n,"Identificador de ejecución":folder.name.split("_",1)[-1]}
                                  for n,folder in folders.items()]),
                     hide_index=True,width="stretch")
except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
    st.error(f"No se pudo representar esta sección: {exc}")
    st.stop()

st.caption("Análisis histórico reproducible. Ninguna selección de modelos se realiza sobre el conjunto de prueba.")
