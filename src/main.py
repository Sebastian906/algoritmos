import pandas as pd
import multiprocessing
from src.controllers.strategies.phi import Phi
from src.controllers.manager import Manager
from src.controllers.methods.memory_reduction import MemoryReductionStrategy
from src.controllers.methods.block_processing import BlockProcessingStrategy
from src.controllers.methods.parallelization import ParallelizationStrategy

# Configuraciones predeterminadas con diferentes estrategias
ESTRATEGIAS_MEMORIA = {
    'rapido': {
        'reduccion': MemoryReductionStrategy(porcentaje=0.2),
        'bloques': BlockProcessingStrategy(),
        'paralelizacion': ParallelizationStrategy(num_procesos=4)
    },
    'equilibrado': {
        'reduccion': MemoryReductionStrategy(porcentaje=0.5),
        'bloques': BlockProcessingStrategy(),
        'paralelizacion': ParallelizationStrategy()
    },
    'preciso': {
        'reduccion': None,  # Sin reducción de memoria
        'bloques': BlockProcessingStrategy(tamaño_bloque=500),
        'paralelizacion': ParallelizationStrategy(num_procesos=None)  # Todos los núcleos
    }
}

# Lista de caracteres base
CARACTERES_BASE = "ABCDEFGHIJKLMNOPQRST" #cambiar por la cantidad de nodos que hay cada nodo expresa por un numero 

def procesar_cadena(cadena):
    try:
        # Separar la cadena en las dos partes (antes y después del "|")
        parte_t1, parte_t = cadena.split("|")

        # Eliminar la parte "{t+1}" y "{t}"
        parte_t1 = parte_t1.replace("_{t+1}", "")
        parte_t = parte_t.replace("_{t}", "")
        #print(parte_t1, parte_t, sep="\t") linea pos si quieres ver que subsistema se esta haciendo 

        # Crear las cadenas alcance y mecanismo basadas en la presencia de los caracteres
        alcance = "".join("1" if c in parte_t1 else "0" for c in CARACTERES_BASE)
        mecanismo = "".join("1" if c in parte_t else "0" for c in CARACTERES_BASE)

        return alcance, mecanismo
    except Exception as e:
        print(f"Error procesando la cadena: {cadena}, Error: {e}")
        return None, None

def ejecutar_proceso(
        resultado_queue, 
        condiciones, 
        alcance, 
        mecanismo, 
        estrategia_memoria='equilibrado'
    ):
    estado_inicio = "10000000000000000000"#agrega los ceros necesarios para la cantidad de nodos que vayas a utilizar 
    config_sistema = Manager(estado_inicial=estado_inicio)

    # Aplicar estrategias de memoria seleccionadas
    config_sistema.preparar_estrategias(ESTRATEGIAS_MEMORIA.get(estrategia_memoria, {}))

    analizador_fb = Phi(config_sistema)
    resultado = analizador_fb.aplicar_estrategia(
        condiciones, 
        alcance, 
        mecanismo,
        estrategia_muestreo=ESTRATEGIAS_MEMORIA[estrategia_memoria]['reduccion'].porcentaje
            if ESTRATEGIAS_MEMORIA[estrategia_memoria]['reduccion'] else None
    )
    resultado_queue.put(resultado)

def ejecutar_con_tiempo_limite(
        condiciones, 
        alcance, 
        mecanismo, 
        estrategia_memoria='equilibrado',
        timeout=3600
    ): #altera el timeout a la cantidad de segunos que quieras esperar que se ejecute el subsistema 
    resultado_queue = multiprocessing.Queue()
    proceso = multiprocessing.Process(
        target=ejecutar_proceso, 
        args=(resultado_queue, condiciones, alcance, mecanismo, estrategia_memoria)
    )
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

def iniciar(
        alcance, 
        mecanismo,
        estrategia_memoria='equilibrado'
    ):
    condiciones = "11111111111111111111"#agrega los 1 necesarios para la cantidad de nodos que vayas a utilizar 
    return ejecutar_con_tiempo_limite(
        condiciones, 
        alcance, 
        mecanismo,
        estrategia_memoria
    )

def leer_columna_excel(
        ruta_archivo, 
        nombre_columna,
        estrategia_memoria='equilibrado'
    ):
    try:
        df = pd.read_excel(ruta_archivo, engine='openpyxl')

        if nombre_columna not in df.columns:
            print(f"Error: La columna '{nombre_columna}' no existe en el archivo.")
            return
        print("Perdida      Tiempo de ejecucion")
        
        for i, fila in df[nombre_columna].items():
            alcance, mecanismo = procesar_cadena(str(fila))
            if alcance is not None and mecanismo is not None:
                resultado = iniciar(alcance, mecanismo, estrategia_memoria)
                if resultado is not None:
                    print(resultado.perdida, resultado.tiempo_ejecucion, sep="\t")
                else:
                    print("Tiempo excedido (60 minuto). Terminando proceso...")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")

if __name__ == "__main__":
    ruta_excel = "C:\\Users\\usuario\\Downloads\\Prueba aotomatizador.xlsx"
    nombre_columna = "Prueba aotomatizador"

    # Opciones de estrategias: 'rapido', 'equilibrado', 'preciso'
    estrategia_memoria = 'equilibrado'

    leer_columna_excel(ruta_excel, nombre_columna, estrategia_memoria)