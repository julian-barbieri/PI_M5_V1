# librerías
import pandas as pd
from cargar_datos import cargar_datos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder as FEOneHotEncoder, OrdinalEncoder as FEOrdinalEncoder



# funcion ára exportar ft_engineering()

def ft_engineering_procesado():
    
    #carga de datos
    df = cargar_datos()
    
    # 0) limpieza del dataset
    #tendencia_ingresos nulos, int o float -> "Sin_informacion"
    df['tendencia_ingresos'] = df['tendencia_ingresos'].apply(
        lambda x: 'Sin_informacion' if isinstance(x, (int, float)) or pd.isna(x) else x
    )

    #promedio_ingresos_datacredito nulos -> valor mediana
    mediana_promedio_ingresos_datacredito = df['promedio_ingresos_datacredito'].median()
    df['promedio_ingresos_datacredito'] = df['promedio_ingresos_datacredito'].fillna(mediana_promedio_ingresos_datacredito)

    #saldo_mora_codeudor  nulos -> valor 0
    df['saldo_mora_codeudor'] = df['saldo_mora_codeudor'].fillna(0)

    #saldo_principal  nulos -> valor 0
    df['saldo_principal'] = df['saldo_principal'].fillna(0)

    #saldo_mora  nulos -> valor 0
    df['saldo_mora'] = df['saldo_mora'].fillna(0)

    #saldo_total  nulos -> valor 0
    df['saldo_total'] = df['saldo_total'].fillna(0)

    #puntaje_datacredito  nulos -> valor mediana
    df['puntaje_datacredito'] = df['puntaje_datacredito'].fillna(df['puntaje_datacredito'].median())
    
    #Obtengo fecha actual
    hoy = pd.Timestamp.today().normalize()

    #eliminamos fechas mayores a la actual ya que son prestamos historicos
    df = df[df["fecha_prestamo"] <= hoy]
    
    #OUTLIERS
    #Eliminamos outliers de clientes con edades > 90 años
    df = df[df["edad_cliente"] <= 90]

    #Eliminamos outliers de clientes con Plazos > 60 meses
    df = df[df["plazo_meses"] <= 60]

    #Eliminamos outliers de clientes con salarios > al percentil 99 
    #MODIFICARLE LUEGO CON 0.95
    p99 = df["salario_cliente"].quantile(0.99)
    df = df[df["salario_cliente"] <= p99]

    #Eliminamos outliers de clientes con prestamos > al percentil 99 
    p99 = df["total_otros_prestamos"].quantile(0.99)
    df = df[df["total_otros_prestamos"] <= p99]
    
    # 1) realizamos ingenieria de features
    # division por edades
    df['grupoEdad'] = pd.cut(df['edad_cliente'],
                                    bins=[0, 30, 50, 90],
                                    labels=['Joven', 'Adulto', 'Mayor'])

    # tipo_laboral la transformo en binaria
    df["es_independiente"] = (df["tipo_laboral"] == "Independiente").astype(int)
    df = df.drop(columns='tipo_laboral')

    # fecha_prestamo dividimos por dia de la semana, mes y año
    df["anio_prestamo"] = df["fecha_prestamo"].dt.year
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    df["dia_semana_prestamo"] = df["fecha_prestamo"].dt.weekday

    # si es fin de mes, clásico en riesgo
    df["fin_de_mes"] = df["fecha_prestamo"].dt.is_month_end.astype(int)
    
    # Total de créditos activos
    df["total_creditos"] = (
        df["creditos_sectorFinanciero"] +
        df["creditos_sectorCooperativo"] +
        df["creditos_sectorReal"]
    )

    #proporcion de mora que tiene 
    df["ratio_mora_saldo"] = df["saldo_mora"] / (df["saldo_total"] + 1)
    
    # 2) Identificar variables categoricas, numericas y binarias
    categorical_cols = ['tipo_credito', 'tendencia_ingresos']
    
    ordinal_cols = ['grupoEdad']
    
    numerical_cols = ['salario_cliente', 'capital_prestado', 'edad_cliente', 'total_otros_prestamos', 'cuota_pactada', 'puntaje', 'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal', 
                    'promedio_ingresos_datacredito', 'anio_prestamo', 'mes_prestamo', 'dia_semana_prestamo', 'total_creditos']
    binary_cols = ['es_independiente', 'fin_de_mes']
    
    # 3) Realizamos las transformaciones con Feature-engine
    for col in categorical_cols + ordinal_cols:
        df[col] = df[col].astype("category")

    # binarios: que sean 0/1 numéricos
    for col in binary_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # numéricas: forzar a numeric
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    leak_cols = [
        "saldo_mora", "saldo_total", "saldo_principal",
        "saldo_mora_codeudor", "ratio_mora_saldo", "puntaje"
    ]

    # Orden temporal antes del split
    df = df.sort_values("fecha_prestamo").reset_index(drop=True)

    drop_cols = ["Pago_atiempo", "fecha_prestamo"] + leak_cols
    X = df.drop(columns=drop_cols)
    y = df["Pago_atiempo"]

    # Columnas para el preprocessor: solo las que siguen en X (sin leak_cols)
    numerical_cols_X = [c for c in numerical_cols if c in X.columns]
    categorical_cols_X = [c for c in categorical_cols if c in X.columns]
    ordinal_cols_X = [c for c in ordinal_cols if c in X.columns]
    binary_cols_X = [c for c in binary_cols if c in X.columns]

    preprocessor_fe = Pipeline(steps=[
        ("num_impute", MeanMedianImputer(variables=numerical_cols_X, imputation_method="median")),
        ("cat_impute", CategoricalImputer(variables=categorical_cols_X, imputation_method="frequent")),
        ("ord_impute", CategoricalImputer(variables=ordinal_cols_X, imputation_method="frequent")),
        ("ord_encode", FEOrdinalEncoder(variables=ordinal_cols_X, encoding_method="ordered")),
        ("cat_encode", FEOneHotEncoder(variables=categorical_cols_X)),
        ("bin_impute", MeanMedianImputer(variables=binary_cols_X, imputation_method="median")),
    ])

    # 5) Dividimos train-test de forma temporal (sin shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 6) aplicamos el preprocesamiento
    X_train_processed_fe = preprocessor_fe.fit_transform(X_train, y_train)
    X_test_processed_fe = preprocessor_fe.transform(X_test)

    # 7) Convertimos a DataFrame
    feature_names = preprocessor_fe.get_feature_names_out()

    X_train_processed_fe = pd.DataFrame(
        X_train_processed_fe,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed_fe = pd.DataFrame(
        X_test_processed_fe,
        columns=feature_names,
        index=X_test.index
    )

    #Visualizamos balanceo de clases (target)
    print(df['Pago_atiempo'].value_counts())
    return X_train_processed_fe, X_test_processed_fe, y_train, y_test