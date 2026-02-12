'''import pandas as pd
from pathlib import Path

def cargar_datos():
    ruta_excel = Path("C:\\Users\\julia\\Data Science\\ProyectoM5_JulianBarbieri\\PI_M5_V1\\Base_de_datos.xlsx")
    df = pd.read_excel(ruta_excel)
    return df

'''

import pandas as pd
import os

def cargar_datos():

    # 1. Ruta absoluta del directorio donde está este script (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    print(ruta_actual)

    # 2. Subir dos niveles para llegar a la carpeta donde está la base de datos
    ruta_proyecto = os.path.dirname(os.path.dirname(ruta_actual))
    print(ruta_proyecto)

    # 3. Contruir la ruta completa al archivo Excel
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")
    print(ruta_excel)
    

    # 4. Leemos los datos y los imprimimos
    df = pd.read_excel(ruta_excel)
    return df

if __name__ == "__main__":
    datos = cargar_datos()