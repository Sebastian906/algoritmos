# Nueva estrategia: Procesamiento en bloques: Dividir el espacio de estados 
# en bloques más pequeños que puedan procesarse de manera independiente.

import numpy as np
from typing import List, Generator, Any

class BlockProcessingStrategy:
    """
    Estrategia de procesamiento por bloques para manejar grandes espacios de estados.
    """
    def __init__(self, tamaño_bloque=1000):
        """
        Inicializa la estrategia de procesamiento por bloques.
        
        Args:
            tamaño_bloque (int, optional): Tamaño predeterminado de los bloques. Por defecto 1000.
        """
        self.tamaño_bloque = tamaño_bloque
        
    @staticmethod
    def dividir_espacio_estados(estados: np.ndarray, tamaño_bloque: int) -> Generator[np.ndarray, None, None]:
        """
        Divide un espacio de estados en bloques más pequeños.
        
        Args:
            estados (np.ndarray): Espacio completo de estados
            tamaño_bloque (int): Tamaño de cada bloque
        
        Yields:
            np.ndarray: Bloques de estados
        """
        for i in range(0, len(estados), tamaño_bloque):
            yield estados[i:i + tamaño_bloque]

    def procesar_bloque(self, bloque: np.ndarray, funcion_procesamiento: callable) -> List[Any]:
        """
        Procesa un bloque de estados usando una función de procesamiento proporcionada.
        
        Args:
            bloque (np.ndarray): Bloque de estados a procesar
            funcion_procesamiento (callable): Función para procesar cada estado
        
        Returns:
            List[Any]: Resultados del procesamiento de cada estado en el bloque
        """
        return [funcion_procesamiento(estado) for estado in bloque]

    def aplicar_estrategia(
        self, 
        estados: np.ndarray, 
        funcion_procesamiento: callable, 
        tamaño_bloque: int = 1000
    ) -> List[Any]:
        """
        Método principal para aplicar procesamiento por bloques.
        
        Args:
            estados (np.ndarray): Espacio completo de estados
            funcion_procesamiento (callable): Función para procesar cada estado
            tamaño_bloque (int, optional): Tamaño de cada bloque. Por defecto 1000.
        
        Returns:
            List[Any]: Resultados combinados del procesamiento de todos los bloques
        """
        resultados_totales = []
        
        for bloque in self.dividir_espacio_estados(estados, tamaño_bloque):
            resultados_bloque = self.procesar_bloque(bloque, funcion_procesamiento)
            resultados_totales.extend(resultados_bloque)
        
        return resultados_totales