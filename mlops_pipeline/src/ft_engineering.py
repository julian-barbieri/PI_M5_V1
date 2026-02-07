# librerías
import pandas as pd
from cargar_datos import cargar_datos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime


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
    categorical_cols = ['tipo_credito', 'grupoEdad', 'tendencia_ingresos']
    numerical_cols = ['salario_cliente', 'capital_prestado', 'edad_cliente', 'total_otros_prestamos', 'cuota_pactada', 'puntaje', 'puntaje_datacredito', 'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor', 'creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal', 
                    'promedio_ingresos_datacredito', 'anio_prestamo', 'mes_prestamo', 'dia_semana_prestamo', 'total_creditos']
    binary_cols = ['es_independiente', 'fin_de_mes']

    # 3) creamos pipeline para cada ruta

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    bin_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    # 4) Realizamos las transformaciones con ColumnTransformer
    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numerical_cols),
            ("cat", cat_pipe, categorical_cols),
            ("bin", bin_pipe, binary_cols)
        ]
    )

    # 5) Definimos features y target
    X = df.drop('Pago_atiempo', axis=1)
    y = df['Pago_atiempo']

    # 6) Dividimos train-test (mismo que partes anteriores)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7) aplicamos el preprocesamiento
    X_train_processed = preprocess.fit_transform(X_train)
    X_test_processed = preprocess.transform(X_test)
    
    # 8) Convertimos a DataFrame
    feature_names = preprocess.get_feature_names_out()

    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )
    
    return X_train_df, X_test_df, y_train, y_test