import pandas as pd
import multiprocessing
from src.controllers.strategies.geometrica import GeometricSIA # Cambio aquí
from src.controllers.manager import Manager

# Lista de caracteres base
CARACTERES_BASE = "ABCDE"  # Asegúrate de que coincida con el número de nodos

def procesar_cadena(cadena):
    try:
        parte_t1, parte_t = cadena.split("|")
        parte_t1 = parte_t1.replace("_{t+1}", "")
        parte_t = parte_t.replace("_{t}", "")
        alcance = "".join("1" if c in parte_t1 else "0" for c in CARACTERES_BASE)
        mecanismo = "".join("1" if c in parte_t else "0" for c in CARACTERES_BASE)
        return alcance, mecanismo
    except Exception as e:
        print(f"Error procesando la cadena: {cadena}, Error: {e}")
        return None, None

def ejecutar_proceso(resultado_queue, condiciones, alcance, mecanismo):
    estado_inicio = "10000"  # Ajusta a la cantidad de nodos
    config_sistema = Manager(estado_inicial=estado_inicio)
    analizador = GeometricSIA(config_sistema)  # Cambio aquí
    resultado = analizador.aplicar_estrategia(condiciones, alcance, mecanismo)
    resultado_queue.put(resultado)

def ejecutar_con_tiempo_limite(condiciones, alcance, mecanismo, timeout=3600):
    resultado_queue = multiprocessing.Queue()
    proceso = multiprocessing.Process(target=ejecutar_proceso, args=(resultado_queue, condiciones, alcance, mecanismo))
    proceso.start()
    
    proceso.join(timeout)
    if not resultado_queue.empty():
        resultado = resultado_queue.get()
        if resultado is not None:
            proceso.terminate()
            proceso.join()
            return resultado
    
    if proceso.is_alive():
        print("Tiempo excedido. Terminando proceso...")
        proceso.terminate()
        proceso.join()
        return None
    
    return None

def iniciar(alcance, mecanismo):
    condiciones = "11111"  # Ajusta a la cantidad de nodos
    return ejecutar_con_tiempo_limite(condiciones, alcance, mecanismo)

def leer_columna_excel(ruta_archivo, nombre_columna):
    try:
        df = pd.read_excel(ruta_archivo, engine='openpyxl')

        if nombre_columna not in df.columns:
            print(f"Error: La columna '{nombre_columna}' no existe en el archivo.")
            return
        print("Perdida      Tiempo de ejecucion")
        
        for i, fila in df[nombre_columna].items():
            alcance, mecanismo = procesar_cadena(str(fila))
            if alcance is not None and mecanismo is not None:
                resultado = iniciar(alcance, mecanismo)
                if resultado is not None:
                    print(resultado.perdida, resultado.tiempo_ejecucion, sep="\t")
                    print(f"{resultado.particion}")
                else:
                    print("Tiempo excedido (60 minutos). Terminando proceso...")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
    ruta_excel = "c:\\Users\\mauri\\Downloads\\Prueba aotomatizador.xlsx"
    nombre_columna = "Prueba aotomatizador"
    leer_columna_excel(ruta_excel, nombre_columna)
