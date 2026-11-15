"""Informe académico S08: lectura auditada, PDF y Markdown, sin modificar insumos."""
from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from predictio_s07.integration import (
    HORIZONS, SECTIONS, SYMBOLS, REPORTS, ValidationError,
    academic_summary, select_latest_complete, validate_artifacts,
)

LOG = logging.getLogger("tempus.s08")
ROOT = Path(__file__).resolve().parents[1]
PURPOSES = (
    "Histograma de log-retornos, curva normal y pruebas D'Agostino/Jarque-Bera.",
    "Identificación MAD de observaciones extremas, sin borrar datos.",
    "Suavizado causal MA5, MA20 y MA50; comparación con cierres.",
    "Descomposición STL sobre log-retornos: tendencia, componente periódica y residuo.",
    "Regresión OLS sobre log-precios completos para describir tendencia.",
    "FFT y periodograma Hann para explorar frecuencias y períodos dominantes.",
    "Modelos ingenuo, Drift20 y AR(1), seleccionados solamente en validación.",
    "Walk-forward, MAE/RMSE/sMAPE, comparación ingenua e intervalos al 90 %.",
)
REMARKS = (
    "La normalidad es una hipótesis contrastable, no requisito universal de predicción. "
    "El histograma es exploratorio y no sustituye los valores p.",
    "Los saltos de precios pueden corresponder a acontecimientos económicos reales. "
    "No eliminarlos preserva fidelidad histórica.",
    "Las medias móviles son descriptivas y tienen retardo. Solo usan datos conocidos.",
    "El patrón STL a cinco sesiones es una hipótesis de análisis retrospectivo. "
    "No se demuestra estacionalidad robusta.",
    "Un R2 elevado dentro de muestra no constituye prueba de generalización.",
    "Los picos espectrales no demuestran que existan ciclos persistentes o explotables.",
    "La selección ocurre antes del test. Las mejoras pequeñas requieren cautela estadística.",
    "La cobertura móvil puede mejorar respecto de la fija, pero sin garantía futura; "
    "HAC exploratorio, solapamiento temporal y comparaciones múltiples.",
)
PLOTS = {
    0: (2, "{a}_normalidad.png"),
    1: (2, "{a}_suavizado.png"),
    2: (2, "{a}_suavizado.png"),
    3: (3, "{a}_descomposicion.png"),
    4: (3, "{a}_tendencia.png"),
    5: (3, "{a}_periodograma.png"),
    6: (6, "{a}_h1_calibracion.png"),
    7: (6, "{a}_h5_calibracion.png"),
}


def summary_matrix(reports: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Recopila las diez combinaciones verificadas de activos y horizontes."""
    rows: list[dict[str, Any]] = []
    for asset in SYMBOLS:
        for h in HORIZONS:
            r = academic_summary(reports, asset, h)
            s5 = reports[5]["symbols"][asset][str(h)]
            s6 = reports[6]["symbols"][asset][str(h)]
            row = {
                "activo": asset, "horizonte": h,
                "modelo": str(r["selected_model"]),
                "mae": float(r["mae"]),
                "mae_ingenuo": float(r["naive_mae"]),
                "mejora_mae_pct": float(s5["improvement_mae_pct"]),
                "cobertura_fija_pct": 100.0 * float(r["coverage_fixed"]),
                "cobertura_movil_pct": 100.0 * float(r["coverage_rolling"]),
                "nominal_pct": 100.0 * float(r["nominal"]),
                "n_test": int(s5["test_targets_after_embargo"]),
                 "ancho_fijo_usd": float(s6["static_s05_mean_width_usd"]),
                "ancho_movil_usd": float(s6["rolling_mean_width_usd"]),
            }
            for name in ("mae", "mae_ingenuo", "mejora_mae_pct",
                         "cobertura_fija_pct", "cobertura_movil_pct", "nominal_pct", "ancho_fijo_usd", "ancho_movil_usd"):
                if not math.isfinite(row[name]):
                    raise ValidationError(f"Métrica no finita: {asset}/h={h}/{name}")
            if row["n_test"] < 100 or not (0 <= row["cobertura_movil_pct"] <= 100):
                raise ValidationError(f"Métrica fuera de rango: {asset}/h={h}")
            rows.append(row)
    return rows


def md_report(reports: dict[int, dict[str, Any]], folders: dict[int, Path],
              rows: list[dict[str, Any]], artefacts: int) -> str:
    """Composición pura de informe; datos en tablas con procedencia explícita."""
    source = reports[2]
    parts = [
        "# TEMPUS PREDICTIO PROTO - Práctica 2",
        "## Informe académico final generado desde ejecuciones locales auditadas",
        "",
        "**Ámbito:** estudio descriptivo y backtesting histórico, sin predicción en tiempo real.",
        "**Período histórico:** 2016-10-31 a 2026-10-30; cinco activos; 2.514 sesiones por activo.",
        f"**Contrato:** `{source['data_contract']}`.",
        f"**SHA-256 fuente CSV:** `{source['data_sha256']}`.",
        f"**Artefactos auditados para interfaz:** {artefacts}.",
        "",
        "## Planteamiento y metodología",
        "Se estudian JNJ, MSFT, JPM, WMT y XOM, respetando secuencia cronológica XNYS y "
        "separando muestra de entrenamiento, validación y prueba. La selección de modelos "
        "se realiza antes de examinar los objetivos test. Para h=5 se aplica embargo de 4 objetivos.",
        "",
        "## Ocho procedimientos explicados por el Tutor",
    ]
    for i, sec in enumerate(SECTIONS):
        parts += [f"### {sec}", PURPOSES[i], "", "**Interpretación y límites:** " + REMARKS[i], ""]
        if i == 0:
            parts.append("**Hallazgos:** las pruebas de normalidad rechazan la hipótesis "
                         "de una distribución normal de log-retornos en los cinco activos.")
        if i == 1:
            parts.append("**Hallazgos:** se detectaron valores extremos por MAD y se conservaron.")
        if i == 5:
            parts.append("**Hallazgos:** el periodograma identifica picos candidatos; "
                         "no se afirma periodicidad estable.")
        parts.append("")
    parts += [
        "## Resultados de backtesting en prueba",
        "",
        "| Activo | h | Modelo | MAE | MAE ingenuo | Mejora % | Cobertura fija % | Cobertura móvil % | n |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        parts.append(
            f"| {r['activo']} | {r['horizonte']} | {r['modelo']} | "
            f"{r['mae']:.4f} | {r['mae_ingenuo']:.4f} | {r['mejora_mae_pct']:+.2f} | "
            f"{r['cobertura_fija_pct']:.2f} | {r['cobertura_movil_pct']:.2f} | {r['n_test']} |"
        )
    parts += [
        "",
        "### Eficiencia de las bandas: amplitud media en USD",
        "| Activo | h | Ancho banda S05 USD | Ancho banda móvil S06 USD |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        parts.append(f"| {r['activo']} | {r['horizonte']} | "
                     f"{r['ancho_fijo_usd']:.3f} | {r['ancho_movil_usd']:.3f} |")
    parts += [
        "",
        "**Lectura:** una mejora porcentual positiva significa MAE inferior a la línea base. "
        "La cobertura se compara con el 90 % nominal: mayor cobertura no implica, por sí sola, "
        "intervalos más eficientes, porque la amplitud también puede crecer.",
        "",
        "## Limitaciones y ética",
        "- Estudio retrospectivo, sin cotizaciones ni pronósticos operativos.",
        "- No se infiere rentabilidad económica; no incorpora costes transaccionales.",
        "- La normalidad fue rechazada; la volatilidad financiera puede cambiar con el tiempo.",
        "- Las bandas no garantizan cobertura conformal bajo dependencia o no estacionariedad.",
        "- Los test HAC son exploratorios, post-selección y sujetos a comparaciones múltiples.",
        "- La inspección visual manual de los ocho apartados debe registrarse por separado.",
        "",
        "## Reproducibilidad y procedencia",
    ]
    for i, folder in sorted(folders.items()):
        parts.append(f"- S{i:02d}: `{folder.name}/{REPORTS[i]}`")
    parts += ["", "## Conclusión",
              "Se implementaron y auditaron procedimientos para normalidad, atípicos, "
              "suavizado, descomposición, tendencia, Fourier, modelado y evaluación. "
              "La mejora predictiva puntual respecto de la línea base es modesta y "
              "la incertidumbre necesita interpretarse críticamente. "
              "Este proyecto es académico, no asesoría de inversión.", ""]
    return "\n".join(parts)


def pdf_report(path: Path, reports: dict[int, dict[str, Any]],
               folders: dict[int, Path], rows: list[dict[str, Any]],
               artefacts: int) -> None:
    """Informe PDF autónomo con los ocho apartados y evidencia visual por activo."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = [Path("C:/Windows/Fonts/arial.ttf"),
             Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    fam = "Helvetica"
    for font in fonts:
        if font.is_file():
            pdfmetrics.registerFont(TTFont("TempusSans", str(font)))
            fam = "TempusSans"
            break
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(name="TempusTitle", parent=ss["Title"], fontName=fam, fontSize=17, leading=22))
    ss.add(ParagraphStyle(name="TempusHeading", parent=ss["Heading2"], fontName=fam, fontSize=12, spaceBefore=13))
    ss.add(ParagraphStyle(name="TempusBody", parent=ss["BodyText"], fontName=fam, fontSize=9, leading=13, spaceAfter=8))
    ss.add(ParagraphStyle(name="TempusTiny", parent=ss["BodyText"], fontName=fam, fontSize=7, leading=10))
    file_width = A4[0] - 72
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=36, rightMargin=36,
                          topMargin=44, bottomMargin=40, title="TEMPUS PREDICTIO - Práctica 2",
                          author="TEMPUS PREDICTIO PROTO - Informe reproducible")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    def page_canvas(canv: Any, doc_obj: Any) -> None:
        canv.saveState()
        canv.setFont(fam, 8)
        canv.setFillColor(colors.HexColor("#475569"))
        canv.drawString(36, 23, "TEMPUS PREDICTIO PROTO | Práctica 2 | Backtesting histórico")
        canv.drawRightString(A4[0]-36, 23, f"Página {doc_obj.page}")
        canv.restoreState()
    doc.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=page_canvas))

    def para(txt: str, style: str="TempusBody") -> Paragraph:
        from xml.sax.saxutils import escape
        return Paragraph(escape(str(txt)), ss[style])
    story: list[Any] = [
        para("TEMPUS PREDICTIO PROTO", "TempusTitle"),
        para("Práctica 2 - Análisis, tratamientos y predicción de series de tiempo", "TempusHeading"),
        para("Informe generado desde los reportes locales validados de S02 a S06. "
             "No presenta cotizaciones operativas ni consejos de inversión."),
        para("Cobertura: 31/10/2016 al 30/10/2026 | JNJ, MSFT, JPM, WMT, XOM | "
             "2.514 sesiones por empresa | datos de frecuencia bursátil XNYS."),
        para(f"Contrato: {reports[2]['data_contract']} | Artefactos verificados: {artefacts}.",
             "TempusTiny"),
        para("Resumen metodológico", "TempusHeading"),
        para("Entrenamiento y validación cronológicos (70/15/15); selección en validación, "
             "prueba fuera de muestra, embargo de cuatro objetivos para horizonte cinco, "
             "reentrenamiento AR(1) con información disponible hasta cada origen. "
             "La evaluación de incertidumbre S06 es retrospectiva."),
        para("Resultados comparativos de prueba", "TempusHeading"),
    ]
    header = ["Activo", "h", "Modelo", "MAE", "Ingenuo", "Mej. %", "Cob. fija", "Cob. móvil"]
    tdata = [header]
    for r in rows:
        tdata.append([
            r["activo"], str(r["horizonte"]), r["modelo"],
            f"{r['mae']:.3f}", f"{r['mae_ingenuo']:.3f}",
            f"{r['mejora_mae_pct']:+.2f}", f"{r['cobertura_fija_pct']:.1f}%",
            f"{r['cobertura_movil_pct']:.1f}%",
        ])
    table = Table(tdata, colWidths=[51,23,88,65,65,57,74,78], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),fam), ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183153")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#edf2f7")]),
        ("BOTTOMPADDING",(0,0),(-1,-1),7), ("TOPPADDING",(0,0),(-1,-1),7),
        ("ALIGN",(1,0),(-1,-1),"RIGHT"),
    ]))
    story += [table,Spacer(1,12),
              para("Amplitud media de los intervalos fijos frente a móviles (USD): "
                   + "; ".join(
                       f"{r['activo']} h{r['horizonte']}: "
                       f"{r['ancho_fijo_usd']:.1f} / {r['ancho_movil_usd']:.1f}"
                       for r in rows), "TempusTiny"),
              para("Cobertura de referencia: 90 %. Las mejoras pequeñas de MAE y los "
                   "contrastes estadísticos exploratorios no demuestran beneficio económico."),
              PageBreak()]
    for idx, section in enumerate(SECTIONS):
        story += [
            para(section, "TempusHeading"),
            para("Propósito: " + PURPOSES[idx]),
            para("Interpretación: " + REMARKS[idx]),
        ]
        if idx == 0:
            story += [para("Las cinco series rechazan normalidad de log-retornos en "
                           "D'Agostino y Jarque-Bera. Los histogramas y la curva normal "
                           "permiten visualizar discrepancias, no certifican normalidad.")]
        for symbol in SYMBOLS:
            sprint, templ = PLOTS[idx]
            imgfile = folders[sprint] / templ.format(a=symbol)
            if imgfile.is_file():
                image = Image(str(imgfile))
                width, height = image.wrap(0,0)
                factor = min(file_width/width, 225/height, 1.0)
                image.drawWidth = width * factor
                image.drawHeight = height * factor
                story += [para(f"{symbol} - evidencia gráfica", "TempusTiny"), image, Spacer(1,7)]
        story.append(PageBreak())
    story += [
        para("Conclusiones y límites", "TempusHeading"),
        para("El pipeline documenta los ocho procedimientos. Las tendencias y picos "
             "espectrales son descriptivos; los modelos muestran mejoras MAE modestas "
             "contra una línea base ingenua. Las bandas de incertidumbre demandan "
             "comprobar simultáneamente cobertura y amplitud."),
        para("No se garantiza cobertura futura ni se evalúan rentabilidades con costes. "
             "Las pruebas HAC son exploratorias y están sometidas a selección y multiplicidad."),
        para("Procedencia y reproducibilidad", "TempusHeading"),
    ]
    for i, folder in sorted(folders.items()):
        story.append(para(f"S{i:02d}: {folder.name}/{REPORTS[i]}", "TempusTiny"))
    doc.build(story)


def finalize(results_root: Path, output_root: Path) -> Path:
    """Produce carpeta nueva; no sobrescribe reportes previos ni datos fuente."""
    folders, reports = select_latest_complete(results_root)
    artefacts = validate_artifacts(results_root, folders)
    if artefacts != 70:
        raise ValidationError(f"Se esperaban 70 artefactos y hay {artefacts}")
    rows = summary_matrix(reports)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = output_root / f"sprint08_{token}"
    out.mkdir(parents=True, exist_ok=False)
    try:
        text = md_report(reports, folders, rows, artefacts)
        (out/"INFORME_PRACTICA_2_FINAL.md").write_text(text, encoding="utf-8")
        pdf_report(out/"INFORME_PRACTICA_2_FINAL.pdf", reports, folders, rows, artefacts)
        import csv
        fields = list(rows[0])
        with (out/"resultados_finales.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        checksums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in out.iterdir() if f.is_file()}
        (out/"manifest_sha256.json").write_text(
            json.dumps({"contract":"tempus-financial-ohlcv/1.0",
                        "source_sha256": reports[2]["data_sha256"],
                        "n_experiments": len(rows), "n_artifacts_verified": artefacts,
                        "files": checksums}, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        import shutil
        shutil.rmtree(out)
        raise
    LOG.info("SPRINT 08 PASS: %s", out)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output = finalize(ROOT / "resultados", ROOT / "resultados")
        print(f"SPRINT 08 PASS: {output}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        LOG.error("SPRINT 08 FAIL: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
