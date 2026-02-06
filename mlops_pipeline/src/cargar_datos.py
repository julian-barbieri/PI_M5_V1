import pandas as pd
from pathlib import Path

def cargar_datos():
    ruta_excel = Path("C:\\Users\\julia\\Data Science\\ProyectoM5_JulianBarbieri\\PI_M5_V1\\Base_de_datos.xlsx")
    df = pd.read_excel(ruta_excel)
    return df