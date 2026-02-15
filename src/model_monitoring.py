import os
import time
import pandas as pd
import numpy as np
import requests
import streamlit as st
from ft_engineering import ft_engineering_procesado

st.set_page_config(page_title="Monitoreo del modelo", layout="wide")

import plotly.express as px 
import plotly.graph_objects as go
from evidently import Report 
from evidently.presets import DataDriftPreset
from cargar_datos import cargar_datos
from scipy.stats import ks_2samp


##############
# 1) Configuracion
##############
API_URL= "http://localhost:8000/predict"
DATASET_PATH = "./Base_de_datos.xlsx" #dataset transformado
MONITOR_LOG = "./Base_de_datos.csv" #dataset para monitorear


##############
# 2) Carga de datos
##############

@st.cache_data
def load_data():
    X_train, X_test, y_train, y_test = ft_engineering_procesado()
    # acá X_train/X_test ya son DataFrames numéricos
    split_idx = int(len(X_train) * 0.8)
    # Últimos registros como datos actuales (orden cronológico)
    X_ref, X_new = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_ref, y_new = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    obj_cols = safe_df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        safe_df[col] = safe_df[col].astype("string")
    return safe_df

##############
# PSI (Population Stability Index)
##############

def calculate_psi(reference, current, bins=10):
    """
    Calcula el PSI entre dos distribuciones.
    
    PSI > 0.25: drift significativo
    PSI 0.1-0.25: drift moderado
    PSI < 0.1: sin drift
    """
    # Manejo de valores nulos
    reference = reference.dropna()
    current = current.dropna()
    
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    
    # Para variables numéricas
    if pd.api.types.is_numeric_dtype(reference):
        # Definir bins basados en el rango combinado
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        # Evitar divisiones por cero
        if min_val == max_val:
            return 0.0
        
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Histogramas
        ref_counts = np.histogram(reference, bins=bins_edges)[0]
        curr_counts = np.histogram(current, bins=bins_edges)[0]
    else:
        # Para variables categóricas
        categories = list(set(reference.unique()) | set(current.unique()))
        ref_counts = np.array([sum(reference == cat) for cat in categories])
        curr_counts = np.array([sum(current == cat) for cat in categories])
    
    # Normalizar para obtener proporciones
    ref_prop = ref_counts / ref_counts.sum()
    curr_prop = curr_counts / curr_counts.sum()
    
    # Evitar log(0)
    ref_prop = np.where(ref_prop == 0, 1e-10, ref_prop)
    curr_prop = np.where(curr_prop == 0, 1e-10, curr_prop)
    
    # PSI = sum((curr_prop - ref_prop) * ln(curr_prop / ref_prop))
    psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
    
    return psi


def calculate_temporal_drift(log_df, reference_df, window_size=50, exclude_cols=None):
    """
    Calcula el PSI máximo por ventana de tiempo.
    Permite ver cómo evoluciona el drift.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    temporal_metrics = []
    
    # Agrupar por ventanas de tamaño window_size
    num_windows = max(1, len(log_df) // window_size)
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(log_df))
        window_data = log_df.iloc[start_idx:end_idx].drop(
            columns=["prediction", "timestamp"], 
            errors="ignore"
        )
        
        if len(window_data) == 0:
            continue
        
        # Calcular PSI máximo en la ventana
        max_psi = 0
        max_psi_var = None
        for col in reference_df.columns:
            if col not in exclude_cols and col in window_data.columns:
                psi = calculate_psi(reference_df[col], window_data[col])
                if not np.isnan(psi) and psi >= max_psi:
                    max_psi = psi
                    max_psi_var = col
        
        # Timestamp medio de la ventana
        if "timestamp" in log_df.columns:
            mid_timestamp = pd.to_datetime(log_df.iloc[start_idx:end_idx]["timestamp"]).mean()
        else:
            mid_timestamp = end_idx
        
        temporal_metrics.append({
            "Ventana": i + 1,
            "Timestamp": mid_timestamp,
            "PSI_max": round(max_psi, 4),
            "Variable_max": max_psi_var,
            "Muestras": len(window_data)
        })
    
    return pd.DataFrame(temporal_metrics)


def calculate_drift_metrics(reference_df, current_df, exclude_cols=None):
    """
    Calcula PSI para todas las columnas numéricas.
    Retorna un DataFrame con los resultados.
    
    exclude_cols: lista de columnas a excluir del análisis
    """
    if exclude_cols is None:
        exclude_cols = []
    
    metrics = []
    
    for col in reference_df.columns:
        if col in exclude_cols or col not in current_df.columns:
            continue
            
        psi = calculate_psi(reference_df[col], current_df[col])
        
        # Clasificar drift
        if np.isnan(psi):
            drift_status = "N/A"
        elif psi > 0.25:
            drift_status = "🔴 Alto"
        elif psi > 0.1:
            drift_status = "🟡 Moderado"
        else:
            drift_status = "🟢 Bajo"
        
        metrics.append({
            "Variable": col,
            "PSI": round(psi, 4),
            "Estado": drift_status
        })
    
    return pd.DataFrame(metrics)


##############
# 3) API para predicciones
##############

def get_predictions(X_batch: pd.DataFrame, progress_bar=None):
    records = X_batch.to_dict(orient="records")
    preds = []
    total = len(records)
    for idx, record in enumerate(records, start=1):
        try:
            response = requests.post(API_URL, json=record)
            response.raise_for_status()
            preds.append(response.json()["Predicted_default"])
            if progress_bar is not None and total > 0:
                progress_bar.progress(idx / total)
        except Exception as e:
            st.error(f"Error conectando con la API: {e}")
            return None
    return preds
        
##############
# 4) Guardar logs con timestamp
##############

def log_predictions(X_batch, preds):
    log_df = X_batch.copy()
    log_df['prediction'] = preds
    log_df['timestamp'] = pd.Timestamp.now()
    
    if os.path.exists(MONITOR_LOG):
        log_df.to_csv(MONITOR_LOG, mode='a', header=False, index=False, quoting=1)  # quoting=csv.QUOTE_ALL
    else:
        log_df.to_csv(MONITOR_LOG, index=False, quoting=1)

##############
# 5) Reporte Evidently
##############

def generate_drift_report(ref_data, new_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data = ref_data, current_data = new_data)
    return report

##############
# 6) Alertas y Recomendaciones
##############

def generate_recommendations(drift_metrics):
    """
    Genera recomendaciones automáticas basadas en el drift detectado.
    """
    recommendations = []
    
    # Análisis de drift
    high_drift_vars = drift_metrics[drift_metrics["PSI"] > 0.25]["Variable"].tolist()
    moderate_drift_vars = drift_metrics[(drift_metrics["PSI"] > 0.1) & (drift_metrics["PSI"] <= 0.25)]["Variable"].tolist()
    
    if len(high_drift_vars) > 0:
        recommendations.append({
            "nivel": "🔴 CRÍTICO",
            "mensaje": f"Se detectó DRIFT ALTO en {len(high_drift_vars)} variable(s): {', '.join(high_drift_vars[:3])}",
            "accion": "RETRAINING URGENTE - Reentrenar el modelo inmediatamente"
        })
    
    if len(moderate_drift_vars) > 0 and len(high_drift_vars) == 0:
        recommendations.append({
            "nivel": "🟡 MODERADO",
            "mensaje": f"Se detectó DRIFT MODERADO en {len(moderate_drift_vars)} variable(s): {', '.join(moderate_drift_vars[:3])}",
            "accion": "MONITOREO INTENSIVO - Aumentar frecuencia de monitoreo. Considerar retraining en próximas horas"
        })
    
    if len(high_drift_vars) == 0 and len(moderate_drift_vars) == 0:
        recommendations.append({
            "nivel": "🟢 ESTABLE",
            "mensaje": "Todas las variables se encuentran ESTABLES",
            "accion": "Continuar con monitoreo rutinario normal"
        })
    
    return recommendations

##############
# 5) StreamLit UI con graficas
##############

st.title("Monitoreo del modelo en Producción")

# Metricas principales en la aparte superior
if os.path.exists(MONITOR_LOG):
    try:
        logged_data = pd.read_csv(MONITOR_LOG, on_bad_lines='skip', engine='python')
    except Exception as e:
        st.error(f"Error al leer logs: {e}")
        st.warning("Intenta borrar el archivo Base_de_datos.csv y reinicia.")
        logged_data = pd.DataFrame()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predicciones", len(logged_data))
    with col2:
        st.metric("Predicción promedio", f"{logged_data['prediction'].mean():.3f}")
    with col3:
        st.metric("Desviacion Estándar", f"{logged_data['prediction'].std():.3f}")

st.sidebar.header("Opciones")
sample_size = st.sidebar.slider("Tamaño de muestra para monitoreo:", 50, 500, 200)
if st.button("Generar nuevas predicciones y actualizar log"):
    sample = X_new.sample(n=sample_size, random_state=int(time.time()))
    with st.spinner("Generando predicciones..."):
        progress_bar = st.progress(0)
        preds = get_predictions(sample, progress_bar)
        progress_bar.progress(1.0)
    
    if preds:
        log_predictions(sample, preds)
        st.success("Nuevas predicciones agregadas al log")
        st.rerun()

# Mostrar datos y graficas (FUERA del if st.button)
if os.path.exists(MONITOR_LOG):
    
    logged_data = pd.read_csv(MONITOR_LOG, on_bad_lines='skip', engine='python')
        
    #Crear tabs para organizar mejor
    tab1, tab2, tab3, tab4 = st.tabs(["Graficas", "Data Drift", "Logs", "Predicciones por Lotes"])

    with tab1:
        col1, col2 = st.columns(2)
            
        with col1:
                
                # Distribución de predicciones 
                pred_counts = (
                    logged_data['prediction']
                    .round()
                    .astype(int)
                    .value_counts()
                    .reindex([0, 1], fill_value=0)
                    .reset_index()
                )
                pred_counts.columns = ["Predicción", "Cantidad"]
                pred_counts["Predicción"] = pred_counts["Predicción"].astype(str)

                fig_hist = px.bar(
                    pred_counts,
                    x="Predicción",
                    y="Cantidad",
                    title="Distribución de Predicciones",
                    color="Predicción",
                    category_orders={"Predicción": ["0", "1"]},
                    color_discrete_map={"0": "#E74C3C", "1": "#2ECC71"}
                )
                fig_hist.update_layout(
                    template="plotly_white",
                    title_x=0.5,
                    xaxis_title="Predicción",
                    yaxis_title="Cantidad",
                    xaxis=dict(type="category"),
                    showlegend=False
                )
                st.plotly_chart(fig_hist, width='content')

        with col2:
                
                #Grafico de linea temporal (si hay timestamp)
                if 'timestamp' in logged_data.columns:
                    logged_data['timestamp'] = pd.to_datetime(logged_data['timestamp'])
                    # Agrupar por minuto para mejor visualizacion
                    temporal_data = logged_data.groupby(
                        logged_data['timestamp'].dt.floor('min')
                    )['prediction'].mean().reset_index()

                    fig_time = px.line(
                        temporal_data,
                        x='timestamp',
                        y='prediction',
                        title = "Evolucion temporal de Predicciones",
                        color_discrete_sequence =['#ff7f0e']
                    )
                    st.plotly_chart(fig_time, width="stretch")
                    
                else:
                    # Box plot como alternativa
                    fig_box = px.box(
                        load_data,
                        y='prediction',
                        title = "Distribucion de Predicciones (Box Plot)"
                    )
                    st.plotly_chart(fig_box, width="stretch")
            
            #Grafico de comparacion con datos de referencia
        st.subheader("Comparacion con Datos de Referencia")        
            
            # Seleccionamos algunas columnas numericas para comparar
        current_data = logged_data.drop(columns=["prediction", "timestamp"], errors="ignore")
        numeric_cols = current_data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'prediction'][:8] #primeras 8 variables
            
        if len(numeric_cols) > 0: 
                # Crear gráficos en columnas de 2
                cols = st.columns(2)
                col_idx = 0
                
                for col in numeric_cols:
                    if col in X_ref.columns:
                        ref_mean = X_ref[col].mean()
                        actual_mean = current_data[col].mean()
                        
                        # Crear DataFrame para el gráfico
                        comp_data = pd.DataFrame({
                            'Dataset': ['Referencia', 'Actual'],
                            'Media': [ref_mean, actual_mean]
                        })
                        
                        # Crear gráfico de barras individual
                        fig = px.bar(
                            comp_data,
                            x='Dataset',
                            y='Media',
                            title=f'Variable: {col}',
                            color='Dataset',
                            color_discrete_map={'Referencia': 'lightblue', 'Actual': 'orange'}
                        )
                        
                        with cols[col_idx % 2]:
                            st.plotly_chart(fig, width='stretch')
                        
                        col_idx += 1
            
        with tab2:
            st.subheader("Reporte de Data Drift (PSI)")
            
            # Calcular PSI para todas las variables (excluyendo las temporales)
            try:
                current_data_for_drift = logged_data.drop(
                    columns=["prediction", "timestamp"], 
                    errors="ignore"
                )
                
                # Columnas a excluir (variables temporales que cambian naturalmente)
                exclude_temporal = ["mes_prestamo", "anio_prestamo", "dia_semana_prestamo", "fin_de_mes"]
                
                drift_metrics = calculate_drift_metrics(X_ref, current_data_for_drift, exclude_cols=exclude_temporal)
                
                # ========================
                # ALERTAS Y RECOMENDACIONES
                # ========================
                st.subheader("🚨 Alertas y Recomendaciones")
                recommendations = generate_recommendations(drift_metrics)
                
                for rec in recommendations:
                    if rec["nivel"].startswith("🔴"):
                        st.error(f"**{rec['nivel']}** {rec['mensaje']}\n\n✅ **Acción:** {rec['accion']}")
                    elif rec["nivel"].startswith("🟡"):
                        st.warning(f"**{rec['nivel']}** {rec['mensaje']}\n\n✅ **Acción:** {rec['accion']}")
                    else:
                        st.success(f"**{rec['nivel']}** {rec['mensaje']}\n\n✅ **Acción:** {rec['accion']}")
                
                st.divider()
                
                # Mostrar tabla con métricas
                st.dataframe(drift_metrics, width='stretch')
                
                # Resumen de drift
                high_drift = len(drift_metrics[drift_metrics["PSI"] > 0.25])
                moderate_drift = len(drift_metrics[(drift_metrics["PSI"] > 0.1) & (drift_metrics["PSI"] <= 0.25)])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Variables con Drift Alto", high_drift)
                with col2:
                    st.metric("Variables con Drift Moderado", moderate_drift)
                with col3:
                    st.metric("Variables Estables", len(drift_metrics) - high_drift - moderate_drift)
                
                # Gráfico de PSI por variable
                fig_psi = px.bar(
                    drift_metrics.sort_values("PSI", ascending=False),
                    x="Variable",
                    y="PSI",
                    color="PSI",
                    color_continuous_scale=["green", "yellow", "red"],
                    title="Índice de Estabilidad Poblacional (PSI) por Variable",
                    hover_data={"Estado": True}
                )
                fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                                 annotation_text="Umbral Moderado")
                fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", 
                                 annotation_text="Umbral Alto")
                st.plotly_chart(fig_psi, width='stretch')
                
            except Exception as e:
                st.error(f"Error calculando drift: {e}")
            
            st.divider()
            st.subheader("Reporte de Data Drift (Evidently)")
            drift_report = generate_drift_report(
                X_ref, current_data_for_drift
            )
            
            # Mostrar reporte
            try:
                st.components.v1.html(drift_report._repr_html_(), height=1000)
            except:
                st.write("Reporte de Data Drift generado exitosamente")            
                st.write(f"Datos de referencia: {X_ref.shape}, Datos actuales: {X_ref.shape}")

            # Análisis temporal dentro de Data Drift (al final)
            st.divider()
            st.subheader("Evolución Temporal del Data Drift")
            
            try:
                exclude_temporal = ["mes_prestamo", "anio_prestamo", "dia_semana_prestamo", "fin_de_mes"]
                window_size = st.sidebar.slider("Tamaño de ventana para análisis:", 20, 200, 50)
                
                temporal_drift = calculate_temporal_drift(
                    logged_data, 
                    X_ref, 
                    window_size=window_size,
                    exclude_cols=exclude_temporal
                )
                
                if len(temporal_drift) > 0:
                    # Gráfico de línea temporal de PSI máximo
                    fig_temporal = px.line(
                        temporal_drift,
                        x="Ventana",
                        y="PSI_max",
                        title="Evolución del PSI Máximo por Ventana Temporal",
                        markers=True,
                        color_discrete_sequence=["#636EFA"]
                    )
                    
                    # Agregar zonas de referencia
                    fig_temporal.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                                          annotation_text="Umbral Moderado (0.1)")
                    fig_temporal.add_hline(y=0.25, line_dash="dash", line_color="red", 
                                          annotation_text="Umbral Alto (0.25)")
                    
                    st.plotly_chart(fig_temporal, width='stretch')
                    
                    # Tabla de evolución
                    st.dataframe(temporal_drift, width='stretch')
                    
                    # Detección de tendencias
                    st.subheader("Análisis de Tendencias")
                    
                    if len(temporal_drift) > 2:
                        # Calcular pendiente (tendencia simple)
                        psi_values = temporal_drift["PSI_max"].values
                        ventanas = np.arange(len(psi_values))
                        
                        # Regresión lineal simple
                        slope = np.polyfit(ventanas, psi_values, 1)[0]
                        
                        # Interpretación
                        if slope > 0.01:
                            st.warning(f"📈 **Tendencia CRECIENTE** - El drift está aumentando (pendiente: {slope:.4f})")
                            st.write("Recomendación: Monitorea de cerca el comportamiento del modelo.")
                        elif slope < -0.01:
                            st.success(f"📉 **Tendencia DECRECIENTE** - El drift está disminuyendo (pendiente: {slope:.4f})")
                            st.write("Recomendación: Excelente, los datos se están estabilizando.")
                        else:
                            st.info(f"➡️ **Tendencia ESTABLE** - El drift se mantiene constante (pendiente: {slope:.4f})")
                            st.write("Recomendación: El modelo funciona de manera estable.")
                    
                else:
                    st.warning("Aún no hay suficientes predicciones para análisis temporal.")
                    
            except Exception as e:
                st.error(f"Error en análisis temporal: {e}")
        
        with tab3:
            st.subheader("Log de monitoreo")
            
            # Filtro para mostrar mas o menos filas
            show_rows = st.selectbox("Mostrar ultimas:", [10, 20, 50, 100])
            display_df = make_arrow_compatible(logged_data.tail(show_rows))
            st.dataframe(display_df, width="stretch")

            # boton de descarga
            csv = logged_data.to_csv(index=False)
            st.download_button(
                label="Descargar CSV completo",
                data = csv,
                file_name=f"monitoring_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime = "text/csv"
            )
        
        with tab4:
            st.subheader("Predicciones por Lotes")
            st.write("Carga un archivo CSV con múltiples registros para obtener predicciones en lote.")
            
            # Upload file
            uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Leer el archivo
                    df_batch = pd.read_csv(uploaded_file)
                    st.write(f"Registros cargados: {len(df_batch)}")
                    st.dataframe(df_batch.head(), width='stretch')
                    
                    if st.button("Hacer predicciones"):
                        with st.spinner("Procesando predicciones..."):
                            try:
                                # Convertir cada fila a diccionario y hacer request
                                records = df_batch.to_dict('records')
                                
                                # Request a la API de batch
                                response = requests.post(
                                    "http://localhost:8000/predict_batch",
                                    json=records,
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    predictions = response.json()["Predicted_default"]
                                    
                                    # Agregar predicciones al dataframe
                                    df_batch['Pago_atiempo'] = predictions
                                    df_batch['Pago_atiempo'] = df_batch['Pago_atiempo'].astype(str).map({'0': '❌ No', '1': '✅ Sí'})
                                    
                                    st.success("✅ Predicciones completadas")
                                    st.dataframe(df_batch, width='stretch')
                                    
                                    # Botón de descarga
                                    csv_results = df_batch.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Descargar resultados",
                                        data=csv_results,
                                        file_name=f"predicciones_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Estadísticas
                                    st.subheader("Resumen de Predicciones")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Total procesado", len(df_batch))
                                    with col2:
                                        default_count = predictions.count(0)
                                        st.metric("Predicciones de Riesgo", default_count)
                                else:
                                    st.error(f"Error en la API: {response.status_code}")
                                    st.write(response.text)
                            except requests.exceptions.ConnectionError:
                                st.error("❌ No se puede conectar con la API. ¿Está corriendo en el puerto 8000?")
                            except Exception as e:
                                st.error(f"❌ Error procesando predicciones: {str(e)}")
                except Exception as e:
                    st.error(f"Error cargando archivo: {str(e)}")
        
else:
    st.warning("No hay datos de monitoreo aún. Presiona el boton para iniciar.")
