"""Presentación y navegación documental de TEMPUS PREDICTIO PROTO.

El módulo no altera datos financieros ni genera pronósticos.
"""
from __future__ import annotations

from typing import Any

CSS = """
<style>
/* Identidad visual TEMPUS, coherente con EXPLORATIO PROTO. */
[data-testid="stAppViewContainer"] { background-color: #ffffff; color: #172b3a; }
[data-testid="stSidebar"] { background-color: #f0f4f8; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #172b3a; font-weight: 750; letter-spacing: .01em;
}
h1, h2, h3 { color: #172b3a; }
[data-testid="stSidebar"] hr { border-color: #cdd8e2; }
[data-testid="stMetricValue"] { color: #172b3a; }
div[data-testid="stAlert"] { border-radius: .45rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { border-radius: .35rem; }
</style>
"""


def apply_style(streamlit: Any) -> None:
    """Aplica únicamente estilo visual, sin modificar datos ni estado."""
    streamlit.markdown(CSS, unsafe_allow_html=True)


def render_project_information(streamlit: Any, page: str) -> None:
    """Documenta la Práctica 2 sin alterar los datos, modelos ni el tablero.

    Las seis páginas son informativas, no ejecutan modelos ni descargan datos.
    Args:
        streamlit: API de Streamlit.
        page: Una de las seis páginas declaradas en la navegación.
    Raises:
        ValueError: Cuando el identificador no corresponde a una página admitida.
    """
    pages = ("Acerca de", "Dataset", "Fundamentos", "Ayuda", "Institucional", "Tecnologías")
    if page not in pages:
        raise ValueError(f"Página informativa no reconocida: {page}")

    if page == "Acerca de":
        streamlit.header("Acerca de TEMPUS PREDICTIO PROTO")
        streamlit.caption("TEMPUS DECISIO PROTO · Módulo II · Práctica 2 · Unidad 4")
        streamlit.markdown("""
**TEMPUS PREDICTIO PROTO** es la aplicación académica de **análisis,
implementación y prueba de técnicas de inteligencia artificial aplicadas a
series de tiempo financieras**. Continúa el trabajo de **TEMPUS EXPLORATIO
PROTO**, que valida y describe los históricos; en este segundo módulo se
analizan propiedades estadísticas y se evalúan predicciones retrospectivas.

El objeto de estudio son los **cierres bursátiles de cinco empresas**:
**JNJ (Johnson & Johnson), MSFT (Microsoft), JPM (JPMorgan Chase),
WMT (Walmart) y XOM (Exxon Mobil)**. Cada serie abarca
**2016-10-31 a 2026-10-30**, con **2.514 sesiones**; el conjunto contiene
**12.570 registros históricos OHLCV**.

**Propósito académico:** estudiar qué puede inferirse del comportamiento
temporal observado y con qué precisión los modelos ARIMA y Holt-Winters
habrían pronosticado cierres a **1 y 5 sesiones bursátiles**. El análisis
económico viene **después de los resultados**, separado entre observación
histórica (informe 09) y evaluación predictiva (informe 10).
""")
        a,b,c=streamlit.columns(3)
        a.metric("Empresas analizadas","5")
        b.metric("Procedimientos","10")
        c.metric("Modelos principales","2")
        streamlit.subheader("Objetivos y estructura")
        streamlit.markdown("""
1. Comprobar distribución y normalidad de **log-retornos**.
2. Señalar movimientos atípicos mediante **MAD**, sin eliminarlos.
3. Estudiar **medias móviles causales**, descomposición **STL**, tendencia
   **OLS** y frecuencias mediante **FFT**.
4. Comparar métodos de pronóstico en particiones cronológicas; evaluar
   precisión, incertidumbre y resultados fuera de muestra.
5. Contrastar **ARIMA (Box-Jenkins)** y **Holt-Winters**. El ingenuo
   permanece **únicamente como referencia técnica**, nunca como modelo
   principal seleccionable.
6. Obtener **dos informes académicos independientes**, evidencias tabulares,
   gráficas y archivos exportables.
""")
        streamlit.subheader("Continuidad académica")
        streamlit.markdown("""
**TEMPUS EXPLORATIO PROTO · Práctica 1:** adquisición, validación,
visualización y descripción de precios históricos.

**TEMPUS PREDICTIO PROTO · Práctica 2:** análisis estadístico más profundo,
evaluación retrospectiva de predicción y lectura económica de resultados.
Los dos proyectos estudian las mismas cinco empresas, pero sus funciones
no son intercambiables.
""")
        streamlit.subheader("Alcance responsable")
        streamlit.markdown("""
Las predicciones aquí evaluadas pertenecen a fechas históricas que ya
pueden contrastarse con el cierre real. **No son previsiones vigentes,
órdenes de compra ni asesoramiento financiero.** Un error MAE no es
beneficio o pérdida de inversión. Dividendos, rendimientos netos y
ganancias realizadas requieren precios de adquisición, fechas y pagos
corporativos verificables que no forman parte del contrato OHLCV usado.
""")

    elif page == "Dataset":
        streamlit.header("Dataset · cinco series financieras")
        streamlit.caption("Fuente de análisis congelada y trazable · 2016-10-31 a 2026-10-30")
        streamlit.markdown("""
El estudio utiliza **cinco series por sesión bursátil del calendario XNYS**.
La base es **histórica, fija y auditada**: ejecutar la interfaz no actualiza
las cotizaciones ni sobrescribe los archivos originales. Las fechas sin
sesión de bolsa no se interpretan automáticamente como valores perdidos.
""")
        streamlit.markdown("""
| Símbolo | Empresa | Ámbito sectorial | Registros |
|---|---|---|---:|
| JNJ | Johnson & Johnson | Salud | 2.514 |
| MSFT | Microsoft | Tecnología | 2.514 |
| JPM | JPMorgan Chase | Servicios financieros | 2.514 |
| WMT | Walmart | Comercio minorista | 2.514 |
| XOM | Exxon Mobil | Energía | 2.514 |

**Total: 12.570 observaciones**; diez años de cobertura histórica.
""")
        streamlit.subheader("Campos y unidades")
        streamlit.markdown("""
- **Date:** fecha de la sesión bursátil, ordenada cronológicamente.
- **Open:** precio de apertura en USD por acción.
- **High:** precio máximo en USD por acción.
- **Low:** precio mínimo en USD por acción.
- **Close:** precio de cierre en USD por acción; variable central del estudio.
- **Volume:** número de acciones negociadas; no se expresa en USD por acción.

Los log-retornos, promedios móviles y demás variables analíticas se
**derivan de los datos originales**; no sustituyen las cotizaciones.
El tratamiento de ajustes corporativos debe respetar la convención del
archivo fuente y su manifiesto.
""")
        streamlit.subheader("Controles de calidad y procedencia")
        streamlit.markdown("""
1. **Cobertura y sesiones XNYS:** período, frecuencia y conteo esperados.
2. **Integridad cronológica:** fechas válidas, sin duplicados ni desorden.
3. **Consistencia OHLCV:** variables numéricas, faltantes y relaciones
   entre apertura, máximo, mínimo y cierre.
4. **Trazabilidad:** los archivos de entrada y los artefactos generados
   conservan referencias e integridad **SHA-256**.
5. **Separación temporal:** entrenamiento / validación / prueba en orden
   cronológico, para evitar usar información futura.
""")
        streamlit.info(
            "Consulta los valores y sus gráficos en «Análisis financiero». "
            "El procedimiento 09 permite descargar el conjunto de evidencias; "
            "el procedimiento 10 contiene las predicciones retrospectivas."
        )
        streamlit.subheader("Limitaciones de la base")
        streamlit.markdown("""
El conjunto **no incorpora una serie auditada de dividendos, resultados
contables, noticias, costos transaccionales ni carteras personales**.
Por ello, ninguna lectura de rentabilidad total o decisión de compra
puede calcularse honestamente solo con OHLCV.
""")

    elif page == "Fundamentos":
        streamlit.header("Fundamentos estadísticos y predictivos")
        streamlit.caption("Conceptos vinculados a los diez procedimientos de la Práctica 2")
        streamlit.subheader("1. Serie temporal y retorno")
        streamlit.markdown("""
Una serie de tiempo es una variable cuantitativa observada en orden
cronológico. Aquí, la variable principal es **Close (USD por acción)**
por sesión XNYS. Los **log-retornos**, calculados a partir de cierres
consecutivos, facilitan estudiar cambios relativos y su distribución.
Los cierres y los retornos responden a preguntas distintas: el primero
es un nivel de precio; el segundo mide su cambio.
""")
        streamlit.subheader("2. Análisis retrospectivo · procedimientos 01 a 06")
        streamlit.markdown("""
| Proc. | Técnica | Pregunta que aborda |
|---|---|---|
| 01 | Histograma, Q-Q, D'Agostino y Jarque–Bera | ¿Son compatibles los log-retornos con la normalidad? |
| 02 | Desviación absoluta mediana (MAD) | ¿Qué movimientos destacan como atípicos? |
| 03 | Medias móviles causales MA5, MA20, MA50 | ¿Qué cambia al suavizar ruido de corto plazo? |
| 04 | STL de log-retornos | ¿Cómo se separan componentes descriptivas? |
| 05 | Tendencia OLS | ¿Qué pendiente lineal describe el histórico? |
| 06 | FFT y periodograma | ¿Qué frecuencias sobresalen exploratoriamente? |

**Interpretación:** los resultados anteriores describen el pasado; ni un
ciclo aparente ni una tendencia descriptiva garantizan resultados futuros.
""")
        streamlit.subheader("3. Evaluación temporal · procedimientos 07 y 08")
        streamlit.markdown("""
Se comparan predicciones hechas con información disponible hasta cada
**origen histórico**. El conjunto de **validación** guía la selección
del método; el de **prueba** queda reservado para medir desempeño fuera
de muestra. Los horizontes son **1 y 5 sesiones bursátiles**.

- **MAE (USD):** error absoluto promedio. Menor es mejor al comparar
  la misma empresa, horizonte y fechas objetivo.
- **RMSE (USD):** penaliza más los errores grandes.
- **Sesgo:** dirección promedio del error; no es rentabilidad.
- **Intervalos y cobertura:** expresan incertidumbre; una cobertura
  próxima a la nominal no elimina el riesgo.
- **Walk-forward:** evaluación en varios orígenes históricos sin consultar
  el precio de la sesión objetivo antes de pronosticar.
""")
        streamlit.subheader("4. Modelos clásicos · procedimiento 10")
        streamlit.markdown("""
**ARIMA(1,1,0) — Box-Jenkins:** modela la dependencia temporal de una
serie diferenciada. La configuración usada es específica del experimento;
no representa una búsqueda exhaustiva de órdenes.

**Holt-Winters:** suavizado exponencial con **tendencia amortiguada**,
sin imponer una estacionalidad no justificada.

**Ingenuo (baseline):** repite el último nivel conocido. Es una
referencia para valorar los otros dos métodos, **no participa en la
selección de modelo principal**.

**Criterio:** menor **MAE en validación entre ARIMA y Holt-Winters**.
La prueba proporciona evidencia independiente y no puede retroalimentar
la elección.
""")
        streamlit.subheader("5. Interpretación económica: orden correcto")
        streamlit.markdown("""
**Datos → análisis → resultados → interpretación económica.**

- **Informe 09:** lectura **exclusivamente histórica** de dispersión,
  cambios, errores e incertidumbre observada; sin nuevos pronósticos.
- **Informe 10:** análisis económico de los **precios pronosticados
  retrospectivamente y los cierres realmente observados**. Una diferencia
  entre pronóstico y realidad no equivale a ganancia obtenida.
- **Tenencia y compra:** plusvalía, minusvalía o dividendo solo se
  cuantifican con precios de compra y pagos verificables. ARIMA y
  Holt-Winters pronostican **precios**, no dividendos.
""")
        streamlit.warning(
            "Algunos ajustes ARIMA pueden producir avisos de no convergencia; "
            "los registros originales no permiten garantizar convergencia "
            "individual para cada origen."
        )

    elif page == "Ayuda":
        streamlit.header("Ayuda · TEMPUS PREDICTIO PROTO")
        streamlit.caption("Manual de uso de la Práctica 2 y verificación de resultados")
        streamlit.subheader("1. Navegación")
        streamlit.markdown("""
La barra lateral ofrece **exactamente dos áreas principales**:

- **Análisis financiero:** diez procedimientos; selectores de empresa
  y horizonte en sesiones bursátiles.
- **Información del proyecto:** **Acerca de, Dataset, Fundamentos, Ayuda,
  Institucional y Tecnologías**.

Los informes y las gráficas se consultan desde **Análisis financiero**.
Estas páginas informativas **no ejecutan modelos**.
""")
        streamlit.subheader("2. Recorrido de los procedimientos")
        streamlit.markdown("""
1. Elige **Análisis financiero** y después la empresa
   **JNJ, MSFT, JPM, WMT o XOM**.
2. Revisa los procedimientos **01 a 06** para comprender normalidad,
   atípicos, suavizado, STL, tendencia y frecuencias.
3. En **07 y 08**, elige horizonte de **1 o 5 sesiones** y examina
   comparación de predicciones, errores e incertidumbre.
4. Abre **09 · Entregables** para descargar el
   **Informe final de la Práctica 2** y el ZIP con evidencias de análisis.
5. Abre **10 · Predictio Proto** para visualizar **ARIMA y Holt-Winters**,
   la tabla de MAE y RMSE, y descargar el
   **Informe de Predicción** y los resultados del procedimiento.
6. Lee primero tablas y gráficos y **después** la interpretación económica.
""")
        streamlit.subheader("3. ¿Qué contiene cada informe?")
        streamlit.markdown("""
| Documento | Enfoque | Descarga |
|---|---|---|
| **09 · Informe final de la Práctica 2** | Ocho procedimientos, resultados históricos, tablas, gráficos y lectura económica retrospectiva | Procedimiento 09 |
| **10 · Informe de Predicción** | Comparación de ARIMA y Holt-Winters, errores de prueba, gráficos, análisis integral por empresa y valoración económica posterior | Procedimiento 10 |

**Son documentos distintos**, aunque ambos pertenecen a la Práctica 2.
""")
        streamlit.subheader("4. Ejemplo de lectura · JNJ")
        streamlit.markdown("""
1. Selecciona **JNJ** en el área de Análisis financiero.
2. Abre **01 · Normalidad** para observar el histograma y Q-Q de log-retornos.
3. En **08 · Evaluación**, compara error e incertidumbre a 1 y 5 sesiones.
4. En **10 · Predictio Proto**, observa precios reales frente a los
   pronosticados, y la selección de ARIMA/Holt-Winters por validación.
5. Descarga el PDF 10; examina los resultados de JNJ antes de leer su
   **interpretación económica**.
""")
        streamlit.subheader("5. Problemas frecuentes")
        streamlit.markdown("""
- **No aparecen procedimientos:** selecciona «Análisis financiero».
- **No aparecen páginas informativas:** selecciona «Información del proyecto».
- **Faltan datos o archivos:** verifica `resultados/`, los archivos
  necesarios y los mensajes de integridad en la pantalla principal.
- **No aparece el procedimiento 10:** la ejecución histórica ARIMA/
  Holt-Winters debe estar generada y validada previamente.
- **No se generan PDF o ZIP:** revisa permisos de lectura, dependencias
  y errores presentados por Streamlit.
- **Ves una versión anterior:** detén Streamlit, reinícialo y recarga
  el navegador. Los PDF descargados antes no se actualizan solos.
- **El ingenuo parece seleccionado:** confirma que utilizas la
  actualización en la que solo compiten ARIMA y Holt-Winters.
""")
        streamlit.subheader("6. Ejecutar la aplicación en Windows")
        streamlit.code(
            r".\.venv\Scripts\python.exe -m streamlit run .\predictio_s09\dashboard.py --server.address 127.0.0.1",
            language="powershell",
        )
        streamlit.caption(
            "Abrir http://127.0.0.1:8501 si el puerto predeterminado está libre. "
            "Cerrar con Ctrl+C en la terminal."
        )

    elif page == "Institucional":
        streamlit.header("Identificación institucional y académica")
        streamlit.caption("TEMPUS DECISIO PROTO · Módulo II · Práctica 2")
        streamlit.markdown("""
| Identificación | Información |
|---|---|
| Institución | **Instituto Internacional de Aguascalientes** |
| Programa | **Maestría en Inteligencia Artificial para la Transformación Digital** |
| Asignatura | **Tópicos de IA para la Transformación Digital** |
| Unidad | **4** |
| Plataforma | **TEMPUS DECISIO PROTO** |
| Aplicación | **TEMPUS PREDICTIO PROTO** |
| Actividad | **Práctica 2: Análisis, implementación y prueba de técnicas de IA aplicadas al Big Data (series de tiempo)** |
| Tutor | **Dr. Francisco Javier Luna Rosas** |
| Estudiante | **Antonio Nicolás Toro González** |
| Período de datos | **31/10/2016 a 30/10/2026** |
""")
        streamlit.subheader("Alcance de los informes académicos")
        streamlit.markdown("""
**Informe final de la Práctica 2 (09):** documenta métodos analíticos,
evidencias, resultados históricos y su interpretación económica.

**Informe de Predicción (10):** documenta modelos, evaluación retrospectiva,
predicciones históricas por empresa y análisis económico posterior.

Los documentos proceden de datos y resultados auditados. La aplicación
no representa una publicación institucional oficial ni la aprobación
automática del trabajo por parte de la institución.
""")
        streamlit.subheader("Integridad académica")
        streamlit.markdown("""
Se distinguen **hechos observados**, **resultados experimentales** e
**interpretaciones condicionadas por la evidencia**. Las advertencias de
convergencia, el muestreo de orígenes y la ausencia de datos de dividendos
deben presentarse expresamente; no se convertirán estimaciones retrospectivas
en recomendaciones bursátiles actuales.
""")

    elif page == "Tecnologías":
        streamlit.header("Tecnologías y arquitectura")
        streamlit.caption("Herramientas usadas en la Práctica 2 y trazabilidad de resultados")
        streamlit.markdown("""
**Flujo reproducible:** datos OHLCV históricos → validación y control
temporal → transformaciones → análisis exploratorio → modelos y
referencia → validación → prueba independiente → gráficos y métricas →
informes académicos → interpretación económica.
""")
        streamlit.subheader("Bibliotecas y componentes")
        streamlit.markdown("""
| Tecnología | Empleo dentro de PREDICTIO |
|---|---|
| **Python 3.12** | Lenguaje de implementación de módulos y pruebas |
| **Streamlit** | Dos áreas de navegación, gráficos, controles y descargas |
| **pandas** | Lectura CSV, fechas, series, transformaciones y tablas |
| **NumPy** | Operaciones numéricas, log-retornos, FFT y métricas |
| **SciPy** | Pruebas estadísticas y herramientas de análisis |
| **statsmodels** | ARIMA, suavizado exponencial, STL y componentes estadísticos |
| **Matplotlib** | Histogramas, Q-Q, periodogramas y gráficos exportados |
| **Bibliotecas PDF del proyecto** | Generación y composición de informes académicos |
| **SHA-256 / JSON / CSV / ZIP** | Integridad, trazabilidad y exportaciones reproducibles |
| **unittest** | Comprobación de componentes, informes e integración |
""")
        streamlit.subheader("Separación de responsabilidades")
        streamlit.markdown("""
| Componente | Responsabilidad |
|---|---|
| `predictio_s09/dashboard.py` | Navegación y representación; no modifica los datos fuente |
| `predictio_s09/estetica.py` | Estilos y documentación informativa |
| `predictio_s09/normality.py` | Histogramas y Q-Q |
| `predictio_s07/integration.py` | Identificación, lectura y validación de artefactos históricos |
| `predictio_visual07/charts.py` | Gráficos de comparación y errores |
| `predictio_entregables/export.py` | ZIP e Informe final de la Práctica 2 |
| `predictio_proto10/integration_final.py` | Lectura y exportación de los resultados de ARIMA/Holt-Winters |
| `resultados/` | Ejecuciones y evidencias auditadas, en lugar de valores codificados en la interfaz |
""")
        streamlit.subheader("Robustez y reproducibilidad")
        streamlit.markdown("""
- **Controles previos:** existencia de archivos, columnas requeridas,
  fechas, cardinalidad e integridad de artefactos.
- **Separación temporal:** entrenamiento, validación y prueba, con
  selección del modelo solo en validación.
- **Integridad:** manifiestos y huellas SHA-256; no se editan los datos
  históricos desde el menú informativo.
- **Pruebas:** validación de componentes y regresión antes de instalar
  actualizaciones; la inspección visual de PDFs sigue siendo necesaria.
- **Limitaciones:** los ajustes ARIMA pueden emitir avisos de
  convergencia; el análisis retrospectivo no equivale a un sistema de
  ejecución bursátil en tiempo real.
""")
