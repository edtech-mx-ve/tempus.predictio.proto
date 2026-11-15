"""Portadas y paginación académica sin alterar las páginas de resultados.

Dos páginas preliminares y todas las páginas originales se conservan intactas.
Las etiquetas PDF permiten que el cuerpo comience nuevamente en la página 1.
No supone conformidad con una norma institucional desconocida.
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from typing import Literal
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from pypdf.constants import PageLabelStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
)

TipoInforme = Literal["practica2", "prediccion"]

INSTITUCION = "Instituto Internacional de Aguascalientes"
POSGRADO = "Maestría en Inteligencia Artificial para la Transformación Digital"
ASIGNATURA = "Tópicos de IA para la Transformación Digital"
UNIDAD = "Unidad 4"
TUTOR = "Dr. Francisco Javier Luna Rosas"
ALUMNO = "Antonio Nicolás Toro González"
PERIODO = "31 de octubre de 2016 al 30 de octubre de 2026"
IDENTIDAD = {
    "practica2": {
        "titulo": "Práctica 2",
        "subtitulo": "Análisis, implementación y evaluación de técnicas de IA aplicadas a series de tiempo",
        "informe": "Informe académico de resultados",
        "tipo": "Informe de análisis y evaluación histórica",
        "candidatos": [
            ("Normalidad", ["01 · Normalidad", "01 - Normalidad"]),
            ("Valores atípicos", ["02 · Valores atípicos", "02 - Valores atípicos"]),
            ("Medias móviles", ["03 · Medias móviles", "03 - Medias móviles"]),
            ("Descomposición temporal", ["04 · Descomposición temporal"]),
            ("Regresión lineal y tendencia", ["05 · Regresión lineal y tendencia"]),
            ("Fourier y periodograma", ["06 · Fourier y periodograma"]),
            ("Modelos predictivos", ["07 · Modelos predictivos"]),
            ("Evaluación e incertidumbre", ["08 · Evaluación e incertidumbre"]),
            ("Conclusiones", ["Conclusiones y límites"]),
            ("Procedencia y reproducibilidad", ["Procedencia y reproducibilidad"]),
            ("Lectura económica histórica", ["Lectura económica de resultados históricos"]),
        ],
    },
    "prediccion": {
        "titulo": "Predictio Proto",
        "subtitulo": "Box-Jenkins (ARIMA) y Holt-Winters: evaluación de predicción temporal",
        "informe": "Informe académico de predicción",
        "tipo": "Informe de experimentación retrospectiva",
        "candidatos": [
            ("Objetivo y alcance", ["Objetivo y alcance"]),
            ("Metodología", ["Metodología"]),
            ("Advertencias y limitaciones", ["Advertencia de convergencia y limitaciones"]),
            ("Resultados fuera de muestra", ["Resultados fuera de muestra"]),
            ("Interpretación", ["Interpretación"]),
            ("Resultados de JNJ", ["JNJ: comparación gráfica"]),
            ("Resultados de MSFT", ["MSFT: comparación gráfica"]),
            ("Resultados de JPM", ["JPM: comparación gráfica"]),
            ("Resultados de WMT", ["WMT: comparación gráfica"]),
            ("Resultados de XOM", ["XOM: comparación gráfica"]),
            ("Análisis Integral de Predicción", ["Análisis Integral de Predicción de las Cinco Empresas"]),
            ("Valoración económica de predicciones", ["Valoración económica posterior a los resultados"]),
            ("Conclusiones y trazabilidad", ["Conclusiones y trazabilidad"]),
        ],
    },
}


def _index_sections(reader: PdfReader, tipo: TipoInforme) -> list[tuple[str, int]]:
    """Localiza secciones solo cuando existen realmente en el documento."""
    sections: list[tuple[str, int]] = []
    text_pages = [re.sub(r"\s+", " ", page.extract_text() or "").casefold()
                  for page in reader.pages]
    for title, candidates in IDENTIDAD[tipo]["candidatos"]:
        found = next(
            (i for i, text in enumerate(text_pages)
             if any(re.sub(r"\s+", " ", pattern).casefold() in text
                    for pattern in candidates)), None)
        if found is not None:
            sections.append((title, found))
    return sections


def _front_matter(tipo: TipoInforme, sections: list[tuple[str,int]],
                  n_pages: int, version: str) -> bytes:
    """Produce carátula y ficha documental (sin reproducciones de datos)."""
    cfg = IDENTIDAD[tipo]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=2.6*cm, leftMargin=2.6*cm,
        topMargin=2.9*cm, bottomMargin=2.3*cm,
        title=f"TEMPUS PREDICTIO PROTO - {cfg['informe']}",
        author=ALUMNO
    )
    nav = colors.HexColor("#203650")
    small = ParagraphStyle(
        "small", fontName="Helvetica", fontSize=10, leading=16,
        alignment=TA_CENTER, textColor=colors.HexColor("#263748")
    )
    heading = ParagraphStyle(
        "heading", parent=small, fontName="Helvetica-Bold", fontSize=17,
        leading=24, spaceAfter=16
    )
    title = ParagraphStyle(
        "title", parent=small, fontName="Helvetica-Bold", fontSize=21,
        leading=28, spaceBefore=20, spaceAfter=22
    )
    normal = ParagraphStyle(
        "normal", fontName="Helvetica", fontSize=9.5, leading=15,
        textColor=colors.HexColor("#263748"), spaceAfter=9
    )
    section_style = ParagraphStyle(
        "section", parent=normal, fontName="Helvetica-Bold", fontSize=12,
        leading=17, spaceBefore=14, spaceAfter=12
    )
    story = [
        Spacer(1, 1.2*cm),
        Paragraph(escape(INSTITUCION), heading),
        Paragraph(escape(POSGRADO), small),
        Paragraph(escape(ASIGNATURA), small),
        Paragraph(UNIDAD, small),
        Spacer(1, 1.5*cm),
        HRFlowable(width="85%", thickness=1, color=nav, hAlign="CENTER"),
        Paragraph("TEMPUS PREDICTIO PROTO", title),
        Paragraph(escape(cfg["titulo"]), heading),
        Paragraph(escape(cfg["subtitulo"]), small),
        Spacer(1, 0.7*cm),
        Paragraph(escape(cfg["informe"]), heading),
        Spacer(1, 1.0*cm),
        Paragraph(f"<b>Estudiante:</b> {escape(ALUMNO)}", small),
        Paragraph(f"<b>Tutor:</b> {escape(TUTOR)}", small),
        Paragraph("<b>Período estudiado:</b> " + escape(PERIODO), small),
        Spacer(1, 1.1*cm),
        Paragraph("Documento de uso académico · resultados reproducibles", small),
        Paragraph("Fecha documental: " + escape(version), small),
    ]
    from reportlab.platypus import PageBreak
    story.extend([
        PageBreak(),
        Paragraph("Identificación y control documental", heading),
        Paragraph("<b>Tipo de documento:</b> " + escape(cfg["tipo"]), normal),
        Paragraph("<b>Institución:</b> " + escape(INSTITUCION), normal),
        Paragraph("<b>Programa:</b> " + escape(POSGRADO), normal),
        Paragraph("<b>Asignatura:</b> " + escape(ASIGNATURA) + " · " + UNIDAD, normal),
        Paragraph("<b>Responsable:</b> " + escape(ALUMNO), normal),
        Paragraph("<b>Versión:</b> " + escape(version), normal),
        Paragraph("<b>Extensión del cuerpo original:</b> "
                  + str(n_pages) + " páginas, íntegramente conservadas.", normal),
        Paragraph("Índice del cuerpo técnico", section_style),
    ])
    if sections:
        for item, page in sections:
            story.append(Paragraph(
                escape(item) + f" <font color='#637387'>· Página {page+1} del cuerpo técnico</font>",
                normal))
    else:
        story.append(Paragraph(
            "No se identificaron encabezados automáticamente; consultar el cuerpo técnico.",
            normal))
    story.extend([
        Spacer(1, 0.3*cm),
        Paragraph("Nota de integridad editorial", section_style),
        Paragraph(
            "Esta edición agrega únicamente páginas preliminares, marcadores de navegación "
            "y metadatos. Todas las páginas del informe técnico original —incluidos "
            "gráficos, tablas, procedimientos, resultados, interpretaciones y referencias "
            "que ya contenga— se anexan sin modificar. La paginación interna del cuerpo "
            "técnico se conserva y empieza en la página 1. No se agregan resultados "
            "científicos ni fuentes bibliográficas no verificadas.", normal),
    ])
    doc.build(story)
    return buffer.getvalue()


def formalizar_pdf(original: bytes, tipo: TipoInforme,
                   version: str | None = None) -> bytes:
    """Enriquece el PDF; verifica que cada página del cuerpo sea idéntica al original."""
    if tipo not in IDENTIDAD:
        raise ValueError("Tipo de informe no permitido")
    if not isinstance(original, bytes) or len(original) < 500 or not original.startswith(b"%PDF"):
        raise ValueError("El informe debe ser un PDF original no vacío")
    if len(original) > 80 * 1024 * 1024:
        raise ValueError("El informe excede el límite permitido (80 MiB)")
    version = version or datetime.now(timezone.utc).strftime("%Y-%m-%d UTC")
    if not re.fullmatch(r"[\w.: /-]{4,70}", version):
        raise ValueError("Versión documental no válida")
    reader = PdfReader(io.BytesIO(original), strict=True)
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 200:
        raise ValueError("Informe cifrado o número de páginas incompatible")
    sections = _index_sections(reader, tipo)
    front = PdfReader(io.BytesIO(_front_matter(tipo, sections, len(reader.pages), version)))
    if len(front.pages) != 2:
        raise ValueError("La portada y ficha académica deben ocupar exactamente dos páginas")
    writer = PdfWriter()
    for page in front.pages:
        writer.add_page(page)
    for page in reader.pages:
        writer.add_page(page)
    writer.set_page_label(0, 1, style=PageLabelStyle.LOWERCASE_ROMAN, start=1)
    writer.set_page_label(2, 1 + len(reader.pages),
                          style=PageLabelStyle.DECIMAL, start=1)
    writer.add_outline_item("Portada institucional", 0)
    writer.add_outline_item("Ficha de control e índice", 1)
    body = writer.add_outline_item("Cuerpo técnico original (íntegro)", 2)
    for title, page_no in sections:
        writer.add_outline_item(title, 2 + page_no, parent=body)
    meta = {
        "/Title": f"TEMPUS PREDICTIO PROTO | {IDENTIDAD[tipo]['informe']}",
        "/Author": ALUMNO,
        "/Subject": IDENTIDAD[tipo]["subtitulo"],
        "/Keywords": "series de tiempo, metodología, evaluación, TEMPUS, uso académico",
        "/Creator": "TEMPUS PREDICTIO PROTO · Protocolo académico",
    }
    writer.add_metadata(meta)
    out = io.BytesIO()
    writer.write(out)
    final = out.getvalue()
    verified = PdfReader(io.BytesIO(final))
    if len(verified.pages) != len(reader.pages) + 2:
        raise ValueError("Integridad: número de páginas incorrecto")
    for i, page in enumerate(reader.pages):
        if page.get_contents().get_data() != verified.pages[i + 2].get_contents().get_data():
            raise ValueError(f"Integridad: el contenido original de la página {i+1} cambió")
    return final
