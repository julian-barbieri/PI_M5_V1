import pandas as pd
from pathlib import Path

def cargar_datos():

    # 1. Ruta absoluta del directorio donde está este script (src)
    ruta_actual = Path(__file__).resolve().parent
    # 2. Subir un nivel para llegar a la carpeta del proyecto (mlops_pipeline)
    ruta_proyecto = ruta_actual.parent

    # 3. Contruir la ruta completa al archivo Excel
    ruta_excel = ruta_proyecto / "Base_de_datos.xlsx"
    

    # 4. Leemos los datos y los imprimimos
    df = pd.read_excel(ruta_excel)
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
    return df

if __name__ == "__main__":
    datos = cargar_datos()