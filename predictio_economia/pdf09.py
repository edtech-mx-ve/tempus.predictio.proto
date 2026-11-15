"""Anexa lectura económica factual al final del cuerpo técnico 09, sin reescribir páginas."""
from __future__ import annotations

import io
from typing import Any

from predictio_economia.lectura import historical_economic_reading, historical_company_interpretations


def add_historical_economic_page(pdf_bytes: bytes, rows: list[dict[str, Any]]) -> bytes:
    """Inserta al final del cuerpo técnico una página derivada de métricas auditadas."""
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

    data = historical_economic_reading(rows)
    stream = io.BytesIO()
    report = SimpleDocTemplate(stream,pagesize=A4,topMargin=1.8*cm,
                               bottomMargin=1.6*cm,leftMargin=1.7*cm,rightMargin=1.7*cm)
    styles = getSampleStyleSheet()
    table_rows=[["Empresa","h","MAE USD","Ancho intervalo USD","Cobertura %"]]
    for d in data:
        table_rows.append([d["activo"], str(d["horizonte"]),
                           f'{d["mae_usd"]:.3f}',f'{d["ancho_intervalo_usd"]:.2f}',
                           f'{d["cobertura_pct"]:.1f}'])
    tab = Table(table_rows,repeatRows=1,
                colWidths=[2.1*cm,1.0*cm,3.2*cm,5.0*cm,3.7*cm])
    tab.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dfeaf2")),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#c5d1da")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),
         [colors.white, colors.HexColor("#f5f8fa")]),
    ]))
    report.build([
        Paragraph("Lectura económica de resultados históricos",styles["Heading1"]),
        Paragraph("Módulo 09 · Interpretación EXCLUSIVAMENTE retrospectiva",
                  styles["Heading3"]),
        Spacer(1,8),tab,Spacer(1,12),
        Paragraph("Implicaciones de los resultados observados",styles["Heading2"]),
        Paragraph("El MAE representa el error absoluto medio del pronóstico "
                  "en dólares por acción. No equivale a una ganancia ni "
                  "a una pérdida de un inversionista. "
                  "El ancho del intervalo móvil resume incertidumbre "
                  "histórica estimada, no un rango garantizado de precio.",
                  styles["BodyText"]),
        Spacer(1,8),
        Paragraph("Para evaluar el riesgo de interpretar variaciones bursátiles "
                  "se consideran conjuntamente la cobertura empírica de los "
                  "intervalos y su amplitud. "
                  "Una cobertura inferior a la nominal indica que, en estos "
                  "datos de prueba, más cierres de los previstos quedaron "
                  "fuera del intervalo; ello no cuantifica pérdidas monetarias.",
                  styles["BodyText"]),
        Spacer(1,8),
        Paragraph("Límites de la evidencia económica",styles["Heading2"]),
        Paragraph("El módulo 09 no ofrece una señal de compra o venta, "
                  "no anticipa precios futuros y no permite calcular "
                  "plusvalías personales ni dividendos percibidos. "
                  "No se suministraron precios de adquisición, posiciones "
                  "del inversionista ni un historial auditado de dividendos. "
                  "Por ello no se afirman rendimientos económicos observados.",
                  styles["BodyText"]),
        Spacer(1,8),
        Paragraph("Esta sección se deriva de la misma tabla de resultados "
                  "que el informe técnico, sin alterar las observaciones "
                  "ni los procedimientos previamente documentados.",
                  styles["BodyText"]),
    ])
    # Segunda página: lectura específica de resultados históricos por empresa.
    detail = io.BytesIO()
    short_doc = SimpleDocTemplate(detail, pagesize=A4, topMargin=1.65*cm,
                                  bottomMargin=1.5*cm, leftMargin=1.75*cm,
                                  rightMargin=1.75*cm)
    company_story = [
        Paragraph("Análisis económico de las cinco empresas",styles["Heading1"]),
        Paragraph("Interpretación posterior a resultados observados (sin predicción nueva)",
                  styles["BodyText"]),
        Spacer(1,10),
    ]
    for item in historical_company_interpretations(rows):
        company_story.extend([
            Paragraph(item["activo"], styles["Heading3"]),
            Paragraph(item["texto"], styles["BodyText"]),
            Spacer(1,6),
        ])
    company_story.extend([
        Spacer(1,8),
        Paragraph("Lectura integral", styles["Heading2"]),
        Paragraph("Las magnitudes en USD son errores e intervalos de resultados "
                  "históricos, no ganancias o pérdidas efectivas. La cobertura "
                  "constata incertidumbre observada por empresa y horizonte; "
                  "sin historial de dividendos auditado no puede cuantificarse "
                  "el ingreso por distribuciones.", styles["BodyText"]),
    ])
    short_doc.build(company_story)
    original = PdfReader(io.BytesIO(pdf_bytes),strict=True)
    extra = PdfReader(io.BytesIO(stream.getvalue()),strict=True)
    if len(extra.pages) != 1:
        raise ValueError("La lectura económica histórica debe ocupar una página")
    writer = PdfWriter()
    for page in original.pages:
        writer.add_page(page)
    writer.add_page(extra.pages[0])
    detailed = PdfReader(io.BytesIO(detail.getvalue()), strict=True)
    if len(detailed.pages) != 1:
        raise ValueError("El análisis por empresa histórico debe ocupar una página")
    writer.add_page(detailed.pages[0])
    out=io.BytesIO()
    writer.write(out)
    return out.getvalue()
