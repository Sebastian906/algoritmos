# Nueva estrategia: Técnicas de Reducción de memoria: Uso de algoritmos 
# heurísiticos para evitar la exploración completa de espacio de datos.

import numpy as np
import random
from typing import List, Any, Callable

class MemoryReductionStrategy:
    def __init__(self, porcentaje=0.2):
        self.porcentaje = porcentaje
    """
    Estrategia de reducción de memoria usando muestreo aleatorio.
    """
    @staticmethod
    def muestrear_estados(
        estados: np.ndarray, 
        num_muestras: int = None, 
        porcentaje: float = None
    ) -> np.ndarray:
        """
        Muestrea un subconjunto de estados usando Monte Carlo.
        
        Args:
            estados (np.ndarray): Espacio completo de estados
            num_muestras (int, optional): Número de muestras a seleccionar
            porcentaje (float, optional): Porcentaje de estados a muestrear
        
        Returns:
            np.ndarray: Estados muestreados
        """
        if num_muestras is None and porcentaje is None:
            raise ValueError("Debe especificar num_muestras o porcentaje")
        
        if porcentaje is not None:
            num_muestras = int(len(estados) * porcentaje)
        
        return np.array(random.sample(list(estados), num_muestras))

    def aplicar_reduccion_montecarlo(
        self, 
        estados: np.ndarray, 
        funcion_procesamiento: Callable, 
        num_muestras: int = None, 
        porcentaje: float = None
    ) -> List[Any]:
        """
        Aplica muestreo de Monte Carlo y procesamiento.
        
        Args:
            estados (np.ndarray): Espacio completo de estados
            funcion_procesamiento (Callable): Función para procesar cada estado
            num_muestras (int, optional): Número de muestras a seleccionar
            porcentaje (float, optional): Porcentaje de estados a muestrear
        
        Returns:
            List[Any]: Resultados del procesamiento de estados muestreados
        """
        estados_muestreados = self.muestrear_estados(
            estados, 
            num_muestras, 
            porcentaje
        )
        
        return [funcion_procesamiento(estado) for estado in estados_muestreados]