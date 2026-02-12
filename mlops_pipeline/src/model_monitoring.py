import os
import time
import pandas as pd
import requests
import streamlit as st
from ft_engineering import ft_engineering_procesado

st.set_page_config(page_title="Monitoreo del modelo", layout="wide")

import plotly.express as px 
import plotly.graph_objects as go
from evidently import Report 
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split
from cargar_datos import cargar_datos


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
    X_ref, X_new, y_ref, y_new = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()


def make_arrow_compatible(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    obj_cols = safe_df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        safe_df[col] = safe_df[col].astype("string")
    return safe_df

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
        log_df.to_csv(MONITOR_LOG, mode='a', header=False, index = False)
    else:
        log_df.to_csv(MONITOR_LOG, index=False)

##############
# 5) Reporte Evidently
##############

def generate_drift_report(ref_data, new_data):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data = ref_data, current_data = new_data)
    return report

##############
# 5) StreamLit UI con graficas
##############

st.title("Monitoreo del modelo en Producción")

# Metricas principales en la aparte superior
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predicciones", len(logged_data))
    with col2:
        st.metric("Predicción promedio", f"{logged_data['prediction'].mean():.3f}")
    with col3:
        st.metric("Desviacion Estándar", f"{logged_data['prediction'].std():.3f}")
    with col4:
        positive_rate = (logged_data['prediction'] > 0.5).mean()*100
        st.metric("Tasa Positiva (%)", f"{positive_rate:.1f}%")

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
        
    # Mostrar datos y graficas
    if os.path.exists(MONITOR_LOG):
        
        logged_data = pd.read_csv(MONITOR_LOG)
        
        #Crear tabs para organizar mejor
        tab1, tab2, tab3 = st.tabs(["Graficas", "Data Drift", "Logs"])

        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                
                #Histograma de predicciones
                fig_hist = px.histogram(
                    logged_data,
                    x='prediction',
                    nbins=20,
                    title="Distribucion de Predicciones",
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig_hist, width="stretch")

            with col2:
                
                #Grafico de linea temporal (si hay timestamp)
                if 'timestamp' in logged_data.columns:
                    logged_data['timestamp'] = pd.to_datetime(logged_data['timestamp'])
                    # Agrupar por minuto para mejor visualizacion
                    temporal_data = logged_data.groupby(
                        logged_data['timestamp'].dt.floor('T')
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
            numeric_cols = [col for col in numeric_cols if col != 'prediction'][:4] #solo las primeras 4
            
            if len(numeric_cols) > 0: 
                comparison_data = []
                for col in numeric_cols:
                    if col in X_ref.columns:
                        comparison_data.append({
                            'Feature':col,
                            'Referencia': X_ref[col].mean(),
                            'Actual': current_data[col].mean(),
                            'Dataset': 'Comparacion'
                        })
            
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Bar(
                        name='Referencia',
                        x=comp_df['Feature'],
                        y=comp_df['Referencia'],
                        marker_color='lightblue'
                    ))
                    
                    fig_comp.add_trace(go.Bar(
                        name='Actual',
                        x=comp_df['Feature'],
                        y=comp_df['Actual'],
                        marker_color='orange'
                    ))
                    
                    fig_comp.update_layout(
                        title='Comparacion de Medias: Referencia vs Actual',
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig_comp, width="stretch")
            
        with tab2:
            st.subheader("Reporte de Data Drift")
            drift_report = generate_drift_report(
                X_ref, logged_data.drop(columns=["prediction", "timestamp"], errors="ignore")
            )
            
            # Mostrar reporte
            try:
                st.components.v1.html(drift_report._repr_html_(), height=1000)
            except:
                st.write("Reporte de Data Drift generado exitosamente")            
                st.write(f"Datos de referencia: {X_ref.shape}, Datos actuales: {X_new.shape}")            

                try:
                    drift_data = drift_report.as_dict()
                    if 'metrics' in drift_data and len (drift_data['metrics']):
                        dataset_drift = drift_data['metrics'][0]. get('result')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Dataset Drift Detectado", "Si" if dataset_drift else "No")
                        with col2:
                            # Contar cuantas features tienen drift
                            feature_drifts = drift_data['metrics'][0].get('result',{}).get('drift_by_columns', {}) 
                            drift_count = sum(1 for v in feature_drifts.values() if v) if feature_drifts else 0
                            st.metric("Features con Drift", f"{drift_count/len(feature_drifts)}" if feature_drifts else "0/0")
                
                except:
                    pass
        
        with tab3:
            st.subheader("Log de monitoreo")
            
            # Filtro para mostrar mas o menos filas
            shows_rows = st.selectbox("Mostrar ultimas:", [10, 20, 50, 100])
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

else:
    st.warning("No hay datos de monitoreo aún. Presiona el boton para iniciar.")
