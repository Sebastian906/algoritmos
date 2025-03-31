from abc import ABC, abstractmethod
import time
from typing import Optional

import numpy as np
import numpy.typing as NDArray

from src.constants.models import SIA_PREPARATION_TAG
from src.middlewares.slogger import SafeLogger
from src.controllers.manager import Manager
from src.models.core.system import System

from src.constants.base import (
    COLON_DELIM,
    FLOAT_ZERO,
    STR_ZERO,
)
from src.constants.error import (
    ERROR_INCOMPATIBLE_SIZES,
)

# Importar las estrategias
from src.controllers.methods.compact_representation import CompactRepresentationStrategy
from src.controllers.methods.block_processing import BlockProcessingStrategy
from src.controllers.methods.parallelization import ParallelizationStrategy
from src.controllers.methods.memory_reduction import MemoryReductionStrategy

class SIA(ABC):
    """
    La clase SIA es la encargada de albergar como madre todos los diferentes algoritmos desarrollados, planteando la base de la que con el método preparar_subsistema se obtendrá uno con características indicadas por el usuario.

    Args:
    ----
        - config (Loader): El cargador de la data desde las muestras con las matrices, es relevante recordar que este tiene el estado inicial como cadena, por lo que es crucial su transoformación a np.array(...) para capacidad de indexar datos.
        - sia_debug_observer (DebugObserver): Debugger que no afecte el rendimiento de la ejecución para un sistema.
        - sia_logger (Logger): Imprime datos de la ejecución en logs/<fecha>/<hora>/ asociando una hora específica por cada fecha del año, allí agrupa el resultado de la ejecución de los distintos loggers situados en aplicativo. De esta forma por hora se almacenará el último resultado de la ejecución.
        - sia_subsistema (System): El subsistema resultante de la preparación, es almacenado para tener una copia reutilizable en el proceso de particionamiento.
        - sia_dists_marginales (np.ndarray): Igualmente, una copia con fines de reutilización durante cálculos con la EMD.
    """

    def __init__(
        self, 
        gestor: Manager,
        estrategia_representacion: Optional[CompactRepresentationStrategy] = None,
        estrategia_bloques: Optional[BlockProcessingStrategy] = None,
        estrategia_paralelizacion: Optional[ParallelizationStrategy] = None,
        estrategia_reduccion: Optional[MemoryReductionStrategy] = None
    ) -> None:
        self.sia_gestor = gestor
        self.sia_logger = SafeLogger(SIA_PREPARATION_TAG)

        # Estrategias de gestión de memoria
        self.estrategia_representacion = estrategia_representacion or CompactRepresentationStrategy()
        self.estrategia_bloques = estrategia_bloques or BlockProcessingStrategy()
        self.estrategia_paralelizacion = estrategia_paralelizacion or ParallelizationStrategy()
        self.estrategia_reduccion = estrategia_reduccion or MemoryReductionStrategy()

        self.sia_subsistema: System
        self.sia_dists_marginales: NDArray[np.float32]
        self.sia_tiempo_inicio: float = FLOAT_ZERO

    @abstractmethod
    def aplicar_estrategia(self):
        """
        Método principal sobre el que las clases herederas implementarán su algoritmo de resolución del problema con una metodología determinada.
        """

    def sia_cargar_tpm(self) -> np.ndarray:
        """Carga TPM desde archivo con estrategia de representación compacta"""
        tpm_original = np.genfromtxt(self.sia_gestor.tpm_filename, delimiter=COLON_DELIM)
        return tpm_original

    def sia_preparar_subsistema(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        estrategia_muestreo: Optional[float] = None
    ):
        """Es en este método donde dada la entrada del usuario, vamos a generar un sistema completo, aplicamos condiciones de fondo (background conditions), loe substraemos partes para dejar un subsistema y es este el que retornamos pues este es el mínimo "sistema" útil para poder encontrar la bipartición que le genere la menor pérdida.

        Args:
            - condicion (str): Cadena de bits donde los bits en cero serán las dimensiones a condicionar.
            - alcance (str): Cadena de bits donde los bits en cero serán las dimensiones a substraer del alcance .
            - mecanismo (str): Cadena de bits donde los bits en cero serán las dimensiones a substraer del mecanismo.

        Raises:
            - Exception: Es crucial que todos tengan el mismo tamaño del estado inicial para correctamente identificar los índices y valor de cada variable rápidamente.
        
        Preparar subsistema con estrategias de gestión de memoria.

        Args:
            estrategia_muestreo (Optional[float]): Porcentaje de muestreo de estados para reducción de memoria.
        """
        if self.chequear_parametros(condicion, alcance, mecanismo):
            raise Exception(ERROR_INCOMPATIBLE_SIZES)

        dims_condicionadas = np.array(
            [ind for ind, bit in enumerate(condicion) if bit == STR_ZERO], dtype=np.int8
        )
        dims_alcance = np.array(
            [ind for ind, bit in enumerate(alcance) if bit == STR_ZERO], dtype=np.int8
        )
        dims_mecanismo = np.array(
            [ind for ind, bit in enumerate(mecanismo) if bit == STR_ZERO], dtype=np.int8
        )

        # Preparar directorio de salida
        self.sia_gestor.output_dir.mkdir(parents=True, exist_ok=True)

        # Cargar y preparar datos
        tpm = self.sia_cargar_tpm()
        estado_inicial = np.array(
            [canal for canal in self.sia_gestor.estado_inicial], dtype=np.int8
        )

        # Formación de datos con logs opcionales de ejemplificación
        completo = System(tpm, estado_inicial)
        self.sia_logger.critic("Original creado.")
        self.sia_logger.info(completo)

        candidato = completo.condicionar(dims_condicionadas)
        self.sia_logger.critic("Candidato creado.")
        self.sia_logger.info(f"{dims_condicionadas}")
        self.sia_logger.debug(candidato)

        subsistema = candidato.substraer(dims_alcance, dims_mecanismo)
        self.sia_logger.critic("Subsistema creado.")
        self.sia_logger.debug(f"{dims_alcance, dims_mecanismo=}")
        self.sia_logger.debug(subsistema)

        # Opcional: Aplicar estrategia de reducción de memoria por muestreo
        if estrategia_muestreo is not None:
            subsistema.estados = self.estrategia_reduccion.muestrear_estados(
                subsistema.estados, 
                porcentaje=estrategia_muestreo
            )

        self.sia_subsistema = subsistema
        self.sia_dists_marginales = subsistema.distribucion_marginal()
        self.sia_tiempo_inicio = time.time()

    def chequear_parametros(self, candidato: str, futuro: str, presente: str):
        """Valida que los datos enviados por el usuario sean correctos, donde no hay problema si tienen la misma longitud puesto se están asignando los valores correspondientes a cada variable.

        Args:
            candidato (str): Cadena de texto que representa la presencia o ausencia de un conjunto de variables que serán enviadas para condicionar el sistema original dejándolo como un Sistema candidato, si su bit asociado equivale a 0 se condiciona la variable, si equivale a 1 esta variable se mantendrá en el sistema durante toda la ejecución (hasta que un subsistema la marginalice).
            futuro (str): Cadena de texto que representa la presencia o ausencia de un conjunto de variables que serán enviadas para substraer en el alcance del Sistema candidato dejándo un Subsistema, si su bit asociado equivale a 0 la variable será marginalizada, si equivale a 1 la variable se mantendrá en el Sistema candidato durante toda la ejecución (hasta que una partición la marginalice).
            presente (str): Cadena de texto que representa la presencia o ausencia de un conjunto de variables que serán enviadas para substraer en el mecanismo del Sistema candidato dejándolo como un Subsistema, si su bit asociado equivale a 0 la variable será marginalizada, si equivale a 1 la variable se mantendrá en el Sistema candidato durante toda la ejecución (hasta que una partición la marginalice).

        Returns:
            bool: True si las dimensiones son diferentes, de otra forma los parámetros enviados son válidos (y depende si existe la red asociada).
        """
        return not (
            len(self.sia_gestor.estado_inicial)
            == len(candidato)
            == len(futuro)
            == len(presente)
        )
    
    def aplicar_procesamiento_paralelo(self, funcion_procesamiento):
        """
        Método para aplicar procesamiento paralelo a los estados del subsistema.
        
        Args:
            funcion_procesamiento (callable): Función a aplicar a cada estado
        
        Returns:
            list: Resultados del procesamiento paralelo
        """
        return self.estrategia_paralelizacion.aplicar_estrategia(
            self.sia_subsistema.estados, 
            funcion_procesamiento
        )

    def aplicar_procesamiento_bloques(
        self, 
        funcion_procesamiento, 
        tamaño_bloque: int = 1000
    ):
        """
        Método para aplicar procesamiento por bloques a los estados del subsistema.
        
        Args:
            funcion_procesamiento (callable): Función a aplicar a cada estado
            tamaño_bloque (int): Tamaño de cada bloque de procesamiento
        
        Returns:
            list: Resultados del procesamiento en bloques
        """
        return self.estrategia_bloques.aplicar_estrategia(
            self.sia_subsistema.estados, 
            funcion_procesamiento, 
            tamaño_bloque
        )