# Nueva estrategia: Paralelización: Uso de herramientas como 
# multiprocessing para dividir el trabajo entre múltiples núcleos.

import multiprocessing
from typing import List, Any, Callable
import numpy as np

class ParallelizationStrategy:
    """
    Estrategia de paralelización para procesamiento distribuido de estados.
    """
    def __init__(self, num_procesos: int = None):
        """
        Inicializa la estrategia de paralelización.
        
        Args:
            num_procesos (int, optional): Número de procesos a utilizar. 
                                          Por defecto usa todos los núcleos disponibles.
        """
        self.num_procesos = num_procesos or multiprocessing.cpu_count()

    def procesar_estado_paralelo(
        self, 
        funcion_procesamiento: Callable, 
        estados: List[Any]
    ) -> List[Any]:
        """
        Procesa estados en paralelo.
        
        Args:
            funcion_procesamiento (Callable): Función para procesar cada estado
            estados (List[Any]): Lista de estados a procesar
        
        Returns:
            List[Any]: Resultados del procesamiento
        """
        with multiprocessing.Pool(processes=self.num_procesos) as pool:
            resultados = pool.map(funcion_procesamiento, estados)
        return resultados

    def aplicar_estrategia(
        self, 
        estados: np.ndarray, 
        funcion_procesamiento: Callable
    ) -> List[Any]:
        """
        Método principal para aplicar procesamiento paralelo.
        
        Args:
            estados (np.ndarray): Espacio de estados a procesar
            funcion_procesamiento (Callable): Función para procesar cada estado
        
        Returns:
            List[Any]: Resultados combinados del procesamiento paralelo
        """
        # Convertir numpy array a lista si es necesario
        estados_lista = estados.tolist() if isinstance(estados, np.ndarray) else estados
        
        return self.procesar_estado_paralelo(funcion_procesamiento, estados_lista)