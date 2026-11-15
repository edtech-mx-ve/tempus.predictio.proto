<div align="center">

<img src="assets/tempus-logo.png" width="150" alt="Logotipo TEMPUS DECISIO PROTO">

# TEMPUS PREDICTIO PROTO

### TEMPUS DECISIO PROTO · Módulo II

**Análisis, implementación y evaluación de técnicas de inteligencia artificial aplicadas a cinco series temporales bursátiles.**

**Python 3.12 · Streamlit · ARIMA (Box-Jenkins) · Holt-Winters · Práctica 2 · Unidad 4**

[Aplicación Streamlit](https://tempus-predictio-proto.streamlit.app/) ·
[Repositorio GitHub](https://github.com/edtech-mx-ve/tempus.predictio.proto)

</div>

> **Dirección pública asignada para la publicación:** https://tempus-predictio-proto.streamlit.app/
>
> **Repositorio de destino:** https://github.com/edtech-mx-ve/tempus.predictio.proto
>
> **Estado:** la dirección de la aplicación está definida por el responsable del proyecto. Su disponibilidad se confirma después del despliegue en Streamlit Community Cloud; no se presenta como ya desplegada.

## 1. Presentación y alcance

**TEMPUS PREDICTIO PROTO** es el Módulo II de la plataforma académica **TEMPUS DECISIO PROTO**. Corresponde a la **Práctica 2: Análisis, implementación y prueba de técnicas de IA aplicadas al Big Data (series de tiempo)**, de la Unidad 4 de *Tópicos de IA para la Transformación Digital*.

Su objetivo es aplicar procedimientos de análisis estadístico y técnicas clásicas de predicción sobre cinco acciones bursátiles, comprobar su desempeño con separación temporal y documentar tanto los resultados técnicos como su **interpretación económica fundamentada**.

El proyecto aprovecha los históricos validados previamente en **TEMPUS EXPLORATIO PROTO (Práctica 1)**, pero desarrolla un trabajo diferente: diagnóstico estadístico, transformación, evaluación retrospectiva y comparación de los modelos **ARIMA (Box-Jenkins)** y **Holt-Winters**.

- **Módulo 09 — Entregables:** resume los análisis, resultados y hechos históricos observados; su lectura económica es **retrospectiva**.
- **Módulo 10 — Predictio Proto:** evalúa predicciones y explica sus implicaciones económicas; los pronósticos registrados son **retrospectivos y contrastados con valores reales**, no cotizaciones futuras en vivo.
- El método **ingenuo** se conserva exclusivamente como **línea base técnica**; **no participa en la elección del modelo principal**, realizada solo entre ARIMA y Holt-Winters.

**Institución:** Instituto Internacional de Aguascalientes.
**Programa:** Maestría en Inteligencia Artificial para la Transformación Digital.
**Asignatura:** Tópicos de IA para la Transformación Digital · Unidad 4.
**Alumno:** Antonio Nicolás Toro González.
**Tutor:** Dr. Francisco Javier Luna Rosas.

## 2. Conjunto de datos oficial

Se utilizan cinco series de cotizaciones históricas, una por empresa:

| Empresa | Símbolo | Sector | Sesiones |
|---|---|---|---:|
| Johnson & Johnson | `JNJ` | Salud | 2.514 |
| Microsoft | `MSFT` | Tecnología | 2.514 |
| JPMorgan Chase | `JPM` | Servicios financieros | 2.514 |
| Walmart | `WMT` | Comercio minorista | 2.514 |
| Exxon Mobil | `XOM` | Energía | 2.514 |
| **Total** | | | **12.570** |

**Cobertura:** 31 de octubre de 2016 al 30 de octubre de 2026.
**Frecuencia:** sesiones efectivas del mercado bursátil, contrastadas con el calendario **XNYS**.
**Variables originales:** `Open`, `High`, `Low`, `Close`, `Volume` (OHLCV), con precios en USD por acción conforme a los ajustes registrados por el proveedor.
**Procedencia histórica declarada:** Yahoo Finance; el trabajo analítico utiliza copias locales auditadas y evidencia de integridad **SHA-256**.

### Control y trazabilidad

- Validación de columnas, tipos, valores finitos, precios positivos, volumen y fechas.
- Comprobación de cronología, unicidad y calendario bursátil; los fines de semana y cierres oficiales **no son faltantes**.
- Preservación de archivos originales: las transformaciones y resultados se guardan **separados** de los históricos.
- El SHA-256 prueba coincidencia con una referencia registrada, **no** por sí mismo veracidad financiera ni licencia de redistribución.
- No se deben confundir cierres ajustados, rendimientos por precio y rendimientos que incluyan dividendos.

**Alcance Big Data:** el caso aborda técnicas de IA y análisis temporal relevantes para datos procedentes de la web; **12.570 registros no acreditan por sí solos procesamiento masivo distribuido**.

## 3. Navegación: dos áreas principales

La aplicación mantiene el diseño académico de TEMPUS EXPLORATIO PROTO, con logotipo y tema claro, y dispone de **dos áreas en la barra lateral**.

### 3.1. Análisis financiero

Permite elegir el **procedimiento**, la **empresa** y el horizonte de prueba (**1 o 5 sesiones**, cuando corresponda).

| Procedimiento | Trabajo realizado |
|---|---|
| **01 · Normalidad** | Histogramas, curva normal de referencia, gráficos Q–Q y pruebas de normalidad de log-retornos. |
| **02 · Valores atípicos** | Identificación robusta mediante desviación absoluta mediana (MAD), sin eliminar automáticamente movimientos financieros reales. |
| **03 · Medias móviles** | Promedios causales de 5, 20 y 50 sesiones para distinguir fluctuaciones de corto y mediano plazo. |
| **04 · Descomposición STL** | Separación descriptiva retrospectiva de componentes temporales; no extrapolación de estacionalidad no verificada. |
| **05 · Tendencia OLS** | Ajuste e interpretación de tendencia lineal por mínimos cuadrados ordinarios. |
| **06 · Espectro de frecuencias** | FFT y periodogramas, con cautela frente a periodicidades aparentes. |
| **07 · Comparación predictiva** | Evaluación cronológica de referencias y modelos simples (ingenuo, Drift20 y AR(1)), sin acceso anticipado al test. |
| **08 · Evaluación fuera de muestra** | Errores, incertidumbre y cobertura de intervalos mediante evaluación temporal. |
| **09 · Entregables** | Descarga del **Informe final de la Práctica 2** y sus evidencias de datos, tablas y gráficas. |
| **10 · Predictio Proto** | Comparación de **ARIMA y Holt-Winters**, selección por validación y descarga del **Informe de Predicción**. |

Los procedimientos 07 y 08 mantienen sus comparaciones y líneas base de investigación. La decisión académica de dejar **dos modelos principales** corresponde al procedimiento 10 y no modifica los experimentos anteriores.

### 3.2. Información del proyecto

Seis páginas informativas, sin ejecutar modelos ni modificar datos:

1. **Acerca de:** propósito, contexto de la Práctica 2 y alcance de PREDICTIO.
2. **Dataset:** empresas, fuente histórica, OHLCV, cobertura y validaciones.
3. **Fundamentos:** las diez etapas analíticas, ARIMA, Holt-Winters y métricas.
4. **Ayuda:** recorrido de la interfaz, selección, consulta y descargas.
5. **Institucional:** identificación académica y contexto curricular.
6. **Tecnologías:** componentes, bibliotecas, arquitectura, calidad y reproducibilidad.

La interfaz **no** presenta «Guía del Tutor» ni la palabra «Sprint» en sus rótulos.

## 4. Enfoque metodológico

**Problema → datos → validación → exploración → transformaciones → modelado → validación temporal → prueba final → interpretación económica → visualización → informe.**

La implementación combina funciones de transformación y métricas comprobables, componentes modulares de responsabilidad definida, controles de errores y procedimientos ejecutables de verificación.

### 4.1. Separación temporal y prevención de fuga

- Las particiones **entrenamiento, validación y prueba** respetan el orden cronológico.
- Las selecciones de modelos e hiperparámetros se fundamentan en **validación** y **no en prueba**.
- Se utilizan orígenes retrospectivos de evaluación *walk-forward*, sin incorporar datos futuros en cada origen.
- Los resultados del test se reservan para medir la generalización histórica, no para reajustar la elección una vez observados.
- Los análisis retrospectivos que emplean el histórico completo se identifican expresamente como **descriptivos**, no como pronósticos en tiempo real.

### 4.2. Modelos predictivos principales

| Modelo | Función académica |
|---|---|
| **ARIMA (Box-Jenkins)** | Modelar dependencia temporal e integración de la serie mediante un enfoque autorregresivo y de medias móviles. |
| **Holt-Winters** | Suavizado exponencial con nivel y tendencia; no se presupone una estacionalidad no acreditada. |
| **Ingenuo** | Línea base de comparación de errores, sin opción de ser el modelo principal elegido en el procedimiento 10. |

Se comparan horizontes **h = 1** y **h = 5** sesiones para cada empresa. La selección del **modelo principal** se realiza mediante el **menor MAE en validación entre ARIMA y Holt-Winters**. El test aporta MAE, RMSE y otras evidencias de desempeño; una cifra aislada no constituye evaluación suficiente.

**Importante:** una advertencia de convergencia de un ajuste ARIMA debe permanecer documentada como limitación técnica, no ocultarse.

## 5. Resultados de referencia del procedimiento 10

En la ejecución académica documentada se seleccionaron **Holt-Winters en seis** y **ARIMA en cuatro** de los diez escenarios (cinco activos × dos horizontes). La selección es por validación; la tabla siguiente corresponde a **un último origen de prueba retrospectiva por escenario**, y no a un pronóstico emitido hoy.

| Activo | h | Modelo principal | Previsto (USD) | Observado (USD) | Error: previsto − real (USD) |
|---|---:|---|---:|---:|---:|
| JNJ | 1 | Holt-Winters | 264,00 | 266,30 | −2,30 |
| JNJ | 5 | ARIMA | 264,03 | 273,20 | −9,17 |
| MSFT | 1 | Holt-Winters | 524,40 | 532,60 | −8,20 |
| MSFT | 5 | ARIMA | 524,13 | 521,91 | +2,22 |
| JPM | 1 | Holt-Winters | 331,97 | 334,04 | −2,07 |
| JPM | 5 | Holt-Winters | 331,96 | 337,04 | −5,08 |
| WMT | 1 | Holt-Winters | 119,93 | 120,29 | −0,36 |
| WMT | 5 | Holt-Winters | 119,98 | 123,51 | −3,53 |
| XOM | 1 | ARIMA | 183,12 | 184,54 | −1,42 |
| XOM | 5 | ARIMA | 183,12 | 187,25 | −4,13 |

**Lectura posterior a los resultados:** en nueve de los diez escenarios el modelo subestimó el precio observado; la excepción fue MSFT a cinco sesiones. Los errores absolutos de estos orígenes fueron mayores a cinco sesiones que a una para los cinco activos. Esta observación **no sustituye** los MAE calculados sobre el conjunto completo de prueba.

**Lectura económica por empresa:**

- **JNJ:** subestimación de USD 2,30 y 9,17; la diferencia retrospectiva es mayor a cinco sesiones.
- **MSFT:** subestimación de USD 8,20 a una sesión y sobreestimación de USD 2,22 a cinco.
- **JPM:** subestimación de USD 2,07 y 5,08 en ambos horizontes.
- **WMT:** las menores desviaciones monetarias de la tabla, USD 0,36 y 3,53.
- **XOM:** subestimación de USD 1,42 y 4,13 en ambos horizontes.

Estas diferencias **no son ganancias ni pérdidas realizadas**. Una apreciación bursátil requiere conocer el precio de entrada y, para materializarla, una venta; los **dividendos** requieren información independiente sobre pagos, fechas y derechos. Los modelos de este procedimiento **pronostican precios, no dividendos**.

## 6. Informes académicos y descargas

La aplicación produce **dos informes diferentes de la Práctica 2**, preservando el contenido técnico y su identificación institucional:

### 6.1. Procedimiento 09 — Informe final de la Práctica 2

Documento general que reúne los ocho primeros análisis, sus tablas y figuras, criterios temporales, evaluaciones y una **lectura económica de hechos históricos**, presentada **después de los resultados**.

**No emite predicciones nuevas ni recomendaciones bursátiles.**

### 6.2. Procedimiento 10 — Informe de Predicción

Documento dedicado a ARIMA y Holt-Winters, con comparación a 1 y 5 sesiones, elección mediante validación, métricas fuera de muestra, figuras, análisis integral de las cinco empresas y **valoración económica posterior a los resultados**.

La interpretación contempla dos perspectivas:

- **Quien ya posee acciones:** el pronóstico de precio puede orientar el análisis de plusvalía o minusvalía *potencial*, pero no calcula una ganancia personal sin precio de compra; los dividendos deben verificarse por separado.
- **Quien no posee acciones:** la expectativa de variación de precio puede apoyar un escenario hipotético de compra, pero no demuestra rentabilidad neta ni constituye una orden de inversión.

Los PDF y archivos ZIP de evidencias se generan o descargan desde los procedimientos respectivos. **Después de cambios en la aplicación, deben regenerarse los documentos**; un PDF descargado previamente no se actualiza solo.

## 7. Arquitectura y estructura de archivos

La aplicación principal se ejecuta desde **`predictio_s09/dashboard.py`** y consulta los resultados auditados ya producidos. Los componentes implementados en el proyecto incluyen, según su distribución:

```text
TEMPUS PREDICTIO PROTO/
├── README.md
├── .venv/                         # Entorno Python local; no subir a Git
├── assets/
│   └── tempus-logo.png
├── predictio_s09/
│   ├── dashboard.py               # Interfaz Streamlit principal
│   ├── estetica.py                # Estilo y páginas informativas
│   └── normality.py               # Visualizaciones de normalidad
├── predictio_s07/
│   └── integration.py             # Selección y verificación de artefactos
├── predictio_visual07/
│   └── charts.py                  # Gráficas de evaluación
├── predictio_entregables/
│   └── export.py                  # Evidencias e Informe final (09)
├── predictio_proto10/
│   ├── modelos.py                 # ARIMA y Holt-Winters
│   └── integration_final.py       # Integración e Informe de Predicción
├── predictio_economia/
│   ├── lectura.py                 # Interpretaciones por empresa
│   └── pdf09.py                   # Lectura económica retrospectiva
├── resultados/                     # Ejecuciones y salidas auditadas
├── tests_predictio_final/         # Pruebas del módulo predictivo
├── tests_economia/                # Pruebas de valoración económica
├── tests_estetica/                # Pruebas de interfaz
└── tests_informacion/             # Pruebas de páginas informativas
```

El árbol es una **referencia de componentes**: los archivos exactos presentes dependen del paquete acumulado e incrementos instalados. Existen también directorios intermedios de procedimientos, pruebas, respaldos y requisitos específicos. **No eliminar módulos ni mover `resultados/` sin validar las dependencias.**

### Dependencias principales

- **Python 3.12:** entorno de ejecución utilizado.
- **Streamlit:** interfaz y descargas.
- **pandas / NumPy:** tablas, series y cálculos.
- **statsmodels:** ARIMA, Holt-Winters, STL y análisis estadísticos.
- **Matplotlib / bibliotecas gráficas:** histogramas, diagnósticos y figuras.
- **exchange_calendars:** sesiones XNYS, cuando aplica.
- **ReportLab / bibliotecas PDF del proyecto:** generación de entregables.
- **unittest / pruebas adicionales:** comprobación automatizada de los componentes instalados.

Instalar las versiones declaradas en los **archivos de requisitos existentes**, sin inventar un archivo único que no esté presente. Antes de publicar en Streamlit Community Cloud, preparar y verificar un `requirements.txt` completo para el entorno de despliegue.

## 8. Instalación y ejecución en Windows

Trabajar desde la **carpeta existente**, sin crear un proyecto distinto:

```powershell
# En PowerShell, ubicado en TEMPUS PREDICTIO PROTO:
.\.venv\Scripts\python.exe --version

# Iniciar la aplicación:
.\.venv\Scripts\python.exe -m streamlit run .\predictio_s09\dashboard.py --server.address 127.0.0.1
```

**URL local:** http://127.0.0.1:8501

Si `.venv` no existe, puede crearse con:

```powershell
py -3.12 -m venv .venv
```

Después se instalan las dependencias **realmente declaradas por el proyecto**. Si faltan resultados históricos, la aplicación **no los inventa ni los descarga automáticamente**; primero debe completarse su generación y validación.

### Pruebas de la instalación local

Según los paquetes de pruebas disponibles:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests_predictio_final -v
.\.venv\Scripts\python.exe -m unittest discover -s tests_economia -v
.\.venv\Scripts\python.exe -m unittest discover -s tests_estetica -v
.\.venv\Scripts\python.exe -m unittest discover -s tests_informacion -v
.\.venv\Scripts\python.exe -m pip check
```

Cada ejecución debe comprobarse en su entorno; **las aprobaciones de instalaciones anteriores no certifican automáticamente una revisión futura**.

### Comprobación manual mínima

1. Iniciar Streamlit y comprobar la portada, el logotipo, el tema claro y las dos áreas principales.
2. Abrir las seis páginas de **Información del proyecto** y verificar su contenido.
3. Recorrer los **diez procedimientos** y cambiar entre las cinco empresas.
4. Verificar tablas, imágenes, horizontes 1 y 5, unidades, errores y selección exclusiva entre ARIMA/Holt-Winters en el procedimiento 10.
5. Generar y descargar el Informe 09 y el Informe 10; revisar sus portadas, gráficas, textos, tablas, orden **resultado → interpretación** y contenido económico.
6. Confirmar que los datos originales se mantienen intactos y que el ZIP de evidencias contiene los archivos esperados.

## 9. Repositorio GitHub y publicación en Streamlit Community Cloud

**Repositorio de destino:** https://github.com/edtech-mx-ve/tempus.predictio.proto
**Nombre de la aplicación asignado:** `tempus-predictio-proto`
**URL pública definida:** https://tempus-predictio-proto.streamlit.app/
**Rama de publicación:** `main`
**Archivo principal (entrypoint):** `predictio_s09/dashboard.py`
**Versión objetivo de Python:** 3.12

La URL anterior es el **nombre y la dirección establecidos para la aplicación**. Su disponibilidad operativa deberá comprobarse después de configurar y desplegar el repositorio. No se requiere crear un segundo directorio local, ni clonar el proyecto otra vez.

El repositorio se administra **desde la carpeta local existente** de TEMPUS PREDICTIO PROTO. La publicación utiliza exclusivamente el código, los recursos y los datos auditados necesarios para que funcione la aplicación; no se suben entornos virtuales, respaldos, ejecuciones temporales ni versiones duplicadas.

### Publicación responsable

Antes de subir el repositorio:

- Revisar dependencias, licencias y permisos de publicación de datos bursátiles.
- No incluir `.venv/`, cachés, respaldos, archivos temporales, secretos ni resultados privados.
- Asegurar que el despliegue pueda encontrar los **artefactos auditados requeridos**; el panel no recalcula todos los procedimientos por abrirse.
- Verificar que `requirements.txt` y la estructura de importación permitan arrancar con el archivo principal indicado.
- Ejecutar las pruebas y revisar la aplicación localmente antes de enviar cambios.

Ejemplo de revisión Git, **sin publicar automáticamente todo el directorio**:

```powershell
git status
git diff -- README.md
git add README.md
git diff --cached --name-only
git commit -m "Documentar TEMPUS PREDICTIO PROTO Practica 2"
git push origin main
```

Este ejemplo presupone que el repositorio Git ya está configurado, que la rama principal es `main` y que el usuario está autorizado para realizar la publicación.

## 10. Limitaciones, seguridad y uso académico

1. **Ventana histórica fija:** 2016-10-31 a 2026-10-30; los pronósticos de prueba ya confrontados con precios reales **no son proyecciones actuales**.
2. **Modelos clásicos:** ARIMA y Holt-Winters dependen de supuestos y condiciones de la serie; los avisos de ajuste o convergencia deben investigarse.
3. **Mayor horizonte, mayor incertidumbre posible:** las métricas deben interpretarse por empresa, horizonte y partición, sin extrapolar un solo resultado a todo el mercado.
4. **Datos financieros:** pueden estar ajustados por eventos corporativos; no se presupone equivalencia con precios nominales no ajustados.
5. **Rentabilidad:** MAE, RMSE o diferencias entre previsto y real **no son ganancias de inversión**; para rentabilidad neta se necesitarían precio de adquisición, comisiones, impuestos y dividendos pertinentes.
6. **Dividendos:** no se modelan mediante ARIMA/Holt-Winters en este proyecto; no deben inventarse montos ni fechas.
7. **Alcance formativo:** no hay garantía de acierto, asesoría financiera personalizada ni órdenes automáticas de compra/venta.
8. **Reproducibilidad:** conservar hashes, parámetros, particiones, informes y resultados originales; las mejoras deben ser incrementales y probadas.
9. **Seguridad:** no registrar secretos, no sobrescribir históricos y no ejecutar scripts externos no revisados.

---

**TEMPUS DECISIO PROTO · TEMPUS PREDICTIO PROTO · Módulo II · Práctica 2**
*Análisis temporal, predicción retrospectiva e interpretación económica basada en evidencia.*
