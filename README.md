# PI_M5_V1: MLOps Pipeline - Monitoreo y Detección de Data Drift

## 📋 Resumen Ejecutivo

Este proyecto implementa un **pipeline de MLOps completo** con énfasis en **monitoreo de modelos en producción** y **detección automática de data drift**. Utiliza técnicas estadísticas avanzadas como el **Índice de Estabilidad Poblacional (PSI)** para garantizar que el desempeño del modelo se mantenga estable en el tiempo.

---

## 🎯 Caso de Negocio

### Problema

Un modelo de predicción de riesgo crediticio fue entrenado con datos históricos, pero los datos en producción evolucionan con el tiempo. Sin monitoreo, el modelo puede perder precisión sin que el equipo se percate, derivando en:

- ❌ Pérdidas económicas
- ❌ Decisiones de crédito incorrectas
- ❌ Incumplimiento regulatorio
- ❌ Degradación silenciosa del modelo

### Solución

Implementar un **sistema de monitoreo en tiempo real** que:

- ✅ Detecte cambios en la distribución de datos (data drift)
- ✅ Emita alertas automáticas según severidad
- ✅ Proporcione recomendaciones de acción (retraining)
- ✅ Visualice métricas clave para tomar decisiones

### Impacto Esperado

- 📊 **Visibilidad continua** del desempeño del modelo
- 🚨 **Alertas tempranas** ante degradación de datos
- ⚡ **Decisiones ágiles** sobre retraining
- 📈 **ROI mejorado** mediante mantención proactiva del modelo

---

## 🏗️ Arquitectura del Proyecto

```
PI_M5_V1/
├── README.md                          # Documentación
├── requirements.txt                   # Dependencias
├── mlops_pipeline/
│   └── src/
│       ├── main.py                    # API FastAPI para predicciones
│       ├── model_monitoring.py        # 🎯 APP STREAMLIT - Monitoreo
│       ├── model_training_evaluation.py # Entrenamiento y optimización
│       ├── ft_engineering.py          # Feature engineering
│       ├── cargar_datos.py            # Carga de datos
│       ├── model_deploy.py            # Deploy del modelo
│       ├── Base_de_datos.csv          # Log de predicciones
│       └── models/
│           ├── RandomForestClassifier_optuna.pkl
│           └── feature_names.pkl      # Columnas del entrenamiento
```

---

## 🔍 Hallazgos Principales

### 1. PSI (Population Stability Index) - Métrica Clave

El PSI es un índice estadístico que mide la divergencia entre dos distribuciones:

| PSI        | Interpretación  | Acción                                     |
| ---------- | --------------- | ------------------------------------------ |
| < 0.1      | 🟢 **Estable**  | Continuar monitoreo rutinario              |
| 0.1 - 0.25 | 🟡 **Moderado** | Aumentar frecuencia, considerar retraining |
| > 0.25     | 🔴 **Crítico**  | RETRAINING URGENTE                         |

**Ventajas del PSI:**

- ✅ Funciona para variables **continuas y categóricas**
- ✅ Independiente de la escala
- ✅ Interpretable: cuantifica el cambio porcentual

### 2. Exclusión de Variables Temporales

Se excluyen automáticamente variables que cambian naturalmente con el tiempo:

- `mes_prestamo`, `anio_prestamo`, `dia_semana_prestamo`, `fin_de_mes`

Estas variables **no indican drift real**, solo cambios temporales esperados.

### 3. Análisis Temporal de Drift

El sistema detecta **tendencias del drift**:

- 📈 **Creciente**: Drift aumentando → acción inmediata
- 📉 **Decreciente**: Datos estabilizándose → buena señal
- ➡️ **Estable**: Drift constante → mantener vigilancia

---

## 🚀 Guía de Uso

### 1. Instalación y Setup

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ubicarse en el directorio correcto
cd mlops_pipeline/src
```

### 2. Entrenar el Modelo

```bash
python model_training_evaluation.py
```

Genera:

- `models/RandomForestClassifier_optuna.pkl` (modelo)
- `models/feature_names.pkl` (columnas)

### 3. Levantar la API

```bash
# Terminal 1: Iniciar servidor API
cd mlops_pipeline/src
uvicorn main:app --reload
```

La API estará disponible en: `http://localhost:8000`

### 4. Ejecutar el Monitoreo (Streamlit)

```bash
# Terminal 2: Iniciar app de monitoreo
cd mlops_pipeline/src
streamlit run model_monitoring.py
```

La app estará disponible en: `http://localhost:8501`

---

## 📊 Funcionalidades de la App Streamlit

### 🔹 Tab 1: Gráficas

- Histograma de distribución de predicciones
- Evolución temporal de predicciones
- Comparación de medias por variable (Referencia vs Actual)

### 🔹 Tab 2: Data Drift

- **ALERTAS Y RECOMENDACIONES** (automáticas)
- Tabla de PSI por variable
- Resumen de drift (alto/moderado/estable)
- Gráfico de barras de PSI con umbrales
- Reporte de Data Drift (Evidently)

### 🔹 Tab 3: Logs

- Tabla de predicciones registradas
- Descarga de CSV completo
- Filtro de últimas N filas

### 🔹 Tab 4: Análisis Temporal

- Gráfico de evolución temporal del PSI máximo
- Tabla de métricas por ventana
- **Análisis de tendencias** (creciente/decreciente/estable)

---

## 🎮 Workflow Principal - Explicación Detallada

### 🔵 Paso 1: Generar Predicciones

**¿Qué hace el botón "Generar nuevas predicciones y actualizar log"?**

Cuando presionas este botón:

1. **Toma una muestra de datos nuevos** del conjunto `X_new` (datos que el modelo NO ha visto antes)
2. **Envía cada registro a la API FastAPI** (http://localhost:8000/predict) para obtener predicciones
3. **Registra las predicciones en el archivo** `Base_de_datos.csv` con timestamp
4. **Actualiza automáticamente todas las visualizaciones** y métricas

**Ejemplo práctico:**

```
Si seleccionas "Tamaño de muestra: 200"
→ La app toma 200 registros de clientes
→ Los envía a la API para predecir riesgo crediticio
→ Guarda las 200 predicciones + variables + timestamp
→ Las acumula en el log para análisis histórico
```

**¿Para qué sirve?**

- Simular cómo el modelo funcionaría en producción
- Acumular datos para detectar drift a lo largo del tiempo
- Comparar predicciones actuales vs datos de referencia

---

### ⚙️ Paso 2: Opciones en el Sidebar

#### 📊 **Tamaño de muestra para monitoreo (50-500)**

**¿Qué controla?**

- Cuántos registros se enviarán a la API cuando presiones el botón

**¿Cómo elegir el valor?**

- **50-100**: Rápido, ideal para pruebas iniciales
- **200**: Valor recomendado para balance velocidad/análisis
- **400-500**: Más datos = análisis estadístico más robusto (pero más lento)

**Ejemplo:**

```
Slider en 100 → Se generan 100 predicciones → Tarda ~5-10 segundos
Slider en 500 → Se generan 500 predicciones → Tarda ~30-50 segundos
```

#### 📈 **Tamaño de ventana para análisis (20-200)**

**¿Qué controla?**

- En la pestaña "Análisis Temporal", divide el log en ventanas para ver evolución del drift

**¿Qué significa "dividir en ventanas"?**

Imagina que tienes un archivo con 1000 predicciones acumuladas en el tiempo:

```
Predicción 1   → 10:00 AM
Predicción 2   → 10:01 AM
Predicción 3   → 10:02 AM
...
Predicción 1000 → 5:00 PM
```

**SIN ventanas** (todo junto):

- Calcularías 1 solo PSI para las 1000 predicciones
- NO sabrías si el drift está aumentando o disminuyendo
- Solo verías un número promedio

**CON ventanas de tamaño 50:**

```
┌─────────────────┐
│ Ventana 1       │ → Predicciones 1-50   → PSI = 0.05 🟢
├─────────────────┤
│ Ventana 2       │ → Predicciones 51-100 → PSI = 0.08 🟢
├─────────────────┤
│ Ventana 3       │ → Predicciones 101-150 → PSI = 0.15 🟡
├─────────────────┤
│ Ventana 4       │ → Predicciones 151-200 → PSI = 0.28 🔴
├─────────────────┤
│ ...             │
└─────────────────┘
```

Ahora puedes **graficar** la evolución:

```
PSI
 |
0.3|              ●  <-- Ventana 4 (CRÍTICO)
   |           ●     <-- Ventana 3 (moderado)
0.2|
   |        ●        <-- Ventana 2
0.1|     ●           <-- Ventana 1 (estable)
   |
 0 +------------------→ Tiempo
    V1   V2   V3   V4
```

**¿Por qué es útil?**

- ✅ Detectas CUÁNDO empezó a aumentar el drift
- ✅ Ves TENDENCIAS (creciente, decreciente, estable)
- ✅ Tomas decisiones ANTICIPADAS antes de que sea crítico

**¿Cómo funciona?**

```
Si tienes 1000 predicciones acumuladas y ventana = 50:
→ Se crean 20 ventanas (1000 / 50 = 20)
→ Ventana 1: Predicciones 1-50
→ Ventana 2: Predicciones 51-100
→ Ventana 3: Predicciones 101-150
→ ...
→ Ventana 20: Predicciones 951-1000

En cada ventana se calcula el PSI máximo de TODAS las variables
```

**¿Cómo elegir el valor?**

- **Ventana pequeña (20-50)**: Detecta cambios rápidos, más sensible
- **Ventana grande (100-200)**: Suaviza fluctuaciones, ve tendencias largas

**Ejemplo práctico visual:**

Con ventana = 100 (tienes 400 predicciones):

```
[Predicciones 1-100]  → PSI_max = 0.05
[Predicciones 101-200] → PSI_max = 0.12
[Predicciones 201-300] → PSI_max = 0.22
[Predicciones 301-400] → PSI_max = 0.30

Gráfico: 4 puntos → Tendencia clara: CRECIENTE 📈
```

Con ventana = 50 (mismas 400 predicciones):

```
[Predicciones 1-50]   → PSI_max = 0.04
[Predicciones 51-100]  → PSI_max = 0.06
[Predicciones 101-150] → PSI_max = 0.10
[Predicciones 151-200] → PSI_max = 0.14
[Predicciones 201-250] → PSI_max = 0.20
[Predicciones 251-300] → PSI_max = 0.24
[Predicciones 301-350] → PSI_max = 0.28
[Predicciones 351-400] → PSI_max = 0.32

Gráfico: 8 puntos → Más detalle, detectas cambio más temprano
```

**Recomendación:**

- Comienza con ventana = 50
- Si el gráfico está muy "ruidoso" (sube y baja mucho), aumenta a 100
- Si no ves suficiente detalle, baja a 20-30

---

### 🔄 Flujo Completo Paso a Paso

**Primera vez usando la app:**

1. **Abre la app Streamlit** → No hay datos de monitoreo aún
2. **Ajusta "Tamaño de muestra"** a 200 (recomendado)
3. **Presiona "Generar nuevas predicciones"**
   - ⏳ Barra de progreso muestra avance (1/200, 2/200...)
   - ✅ Mensaje: "Nuevas predicciones agregadas al log"
4. **Automáticamente se actualizan:**
   - Métricas superiores (Total predicciones, promedio, etc.)
   - Tab "Gráficas": Histograma + línea temporal
   - Tab "Data Drift": **ALERTAS** + tabla PSI + gráfico
   - Tab "Logs": Tabla con las 200 predicciones
   - Tab "Análisis Temporal": Gráfico de evolución

**Segunda vez (simulando paso del tiempo):**

5. **Espera unos minutos** (simula que pasa el tiempo)
6. **Ajusta muestra** a 150
7. **Presiona botón nuevamente**
   - ✅ Se **agregan** 150 predicciones más al log
   - ✅ Ahora tienes 350 predicciones acumuladas (200 + 150)
8. **Observa cambios:**
   - Tab "Data Drift" → ¿Las alertas cambiaron?
   - Tab "Análisis Temporal" → ¿Más ventanas? ¿Drift creciente?

**Ciclo de monitoreo continuo:**

9. **Repite paso 6-8** varias veces
10. **Analiza tendencias:**
    - Si PSI sube → 🔴 Retraining necesario
    - Si PSI se mantiene → 🟢 Modelo estable

---

### 💡 Casos de Uso Reales

#### **Escenario 1: Primera evaluación**

```
→ Muestra: 200
→ Ventana: 50
→ Resultado: 2 alertas moderadas
→ Decisión: Continuar monitoreando
```

#### **Escenario 2: Después de 1 semana**

```
→ Total acumulado: 1500 predicciones
→ Ventana: 100 (para ver tendencia semanal)
→ Resultado: 5 alertas críticas + tendencia creciente
→ Decisión: RETRAINING URGENTE
```

#### **Escenario 3: Validación rápida**

```
→ Muestra: 50
→ Solo quieres ver si la API funciona
→ No analizas drift todavía
```

---

## 🎮 Workflow Principal (Resumen)

1. **Generar Predicciones**
   - Ajustar tamaño de muestra (slider en sidebar)
   - Presionar "Generar nuevas predicciones y actualizar log"
   - Ver progreso en barra visual

2. **Revisar Alertas**
   - ✅ Al generarse predicciones, automáticamente se calcula el PSI
   - ✅ Se emiten recomendaciones en rojo/amarillo/verde
   - ✅ Decisión rápida: ¿Retrainar o continuar?

3. **Analizar Drift**
   - Revisar tabla de PSI por variable
   - Identificar cuáles variables están driftando
   - Comparar con datos de referencia

4. **Monitoreo Temporal**
   - Ajustar ventana de análisis
   - Observar evolución del drift
   - Detectar patrones y tendencias

---

## 📌 Umbrales Configurables

Editar en `model_monitoring.py`:

```python
# Línea ~420: Umbrales de PSI
exclude_temporal = ["mes_prestamo", "anio_prestamo", "dia_semana_prestamo", "fin_de_mes"]

# Para cambiar umbrales de alert, editar:
elif psi > 0.25:    # Umbral CRÍTICO
    drift_status = "🔴 Alto"
elif psi > 0.1:     # Umbral MODERADO
    drift_status = "🟡 Moderado"
```

---

## 📈 Métricas y KPIs

| Métrica                      | Descripción                          | Ubicación             |
| ---------------------------- | ------------------------------------ | --------------------- |
| **Total Predicciones**       | Cantidad de predicciones registradas | Top izquierda         |
| **Predicción Promedio**      | Media de score de riesgo             | Top centro            |
| **Desviación Estándar**      | Variabilidad de predicciones         | Top derecha           |
| **Tasa Positiva (%)**        | % de predicciones > 0.5              | Top extremo           |
| **Variables con Drift Alto** | Cuenta de PSI > 0.25                 | Tab Data Drift        |
| **PSI Máximo por Ventana**   | Evolución temporal                   | Tab Análisis Temporal |

---

## 🛠️ Tecnologías Utilizadas

- **FastAPI**: API REST para predicciones
- **Streamlit**: Interfaz interactiva de monitoreo
- **Pandas & NumPy**: Procesamiento de datos
- **Scikit-learn**: Modelo (RandomForest + Optuna)
- **Plotly**: Visualizaciones interactivas
- **Evidently**: Reportes de data drift
- **Joblib**: Serialización de modelos

---

## ⚙️ Configuración y Personalización

### Modificar Variables Excluidas

Editar línea en `model_monitoring.py`:

```python
exclude_temporal = ["mes_prestamo", "anio_prestamo", "dia_semana_prestamo", "fin_de_mes"]
```

### Cambiar Umbral de PSI

Editar función `calculate_drift_metrics()`:

```python
if psi > 0.30:  # Umbral más alto
    drift_status = "🔴 Alto"
```

### Ajustar Tamaño de Ventana

El slider en sidebar controla el tamaño de ventana para análisis temporal (20-200 muestras).

---

## 📋 Checklist de Puesta en Marcha

- [ ] Instalar dependencias: `pip install -r requirements.txt`
- [ ] Entrenar modelo: `python model_training_evaluation.py`
- [ ] Verificar archivos en `models/`: `.pkl` y `feature_names.pkl`
- [ ] Levantar API: `uvicorn main:app --reload`
- [ ] Ejecutar Streamlit: `streamlit run model_monitoring.py`
- [ ] Generar primeras predicciones
- [ ] Revisar alertas y recomendaciones
- [ ] Validar gráficos y métricas

---

## 📞 Soporte y Mejoras Futuras

### Mejoras Potenciales

- 🔧 Exportar alertas a email/Slack
- 🔧 Dashboard con histórico de drift
- 🔧 Predicción de cuándo retrainar
- 🔧 Integración con CI/CD para retraining automático
- 🔧 Métricas de desempeño del modelo (accuracy, AUC)

### Contacto

Para preguntas o mejoras, revisar la documentación o ajustar parámetros según necesidad.

---

**Última actualización:** 13 de Febrero, 2026  
**Estado:** ✅ Producción - Sistema de Monitoreo Activo
