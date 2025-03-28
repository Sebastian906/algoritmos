# Nueva estrategia: Representación compacta de datos: Uso de estructuras de datos como 
# matrices dispersas (sparse matrices) para representar grandes espacios de estados.

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sparse

class CompactRepresentationStrategy:
    """
    Estrategia de representación compacta para manejo eficiente de matrices de probabilidad de transición.
    """
    @staticmethod
    def crear_tpm_dispersa(tpm: np.ndarray) -> csr_matrix:
        """
        Convierte una matriz de probabilidad de transición a una matriz dispersa.
        
        Args:
            tpm (np.ndarray): Matriz de probabilidad de transición original
        
        Returns:
            csr_matrix: Matriz dispersa representando la TPM
        """
        # Convertir a matriz dispersa conservando solo valores significativos
        umbral = 1e-6  # Umbral para considerar un valor como significativo
        tpm_sparse = csr_matrix(np.where(np.abs(tpm) > umbral, tpm, 0))
        return tpm_sparse

    @staticmethod
    def marginalizar_por_filas(tpm_sparse: csr_matrix, elementos_a_excluir: list) -> csr_matrix:
        """
        Marginaliza una matriz dispersa excluyendo filas específicas.
        
        Args:
            tpm_sparse (csr_matrix): Matriz de probabilidad de transición dispersa
            elementos_a_excluir (list): Índices de elementos a excluir
        
        Returns:
            csr_matrix: Matriz marginalizada
        """
        # Máscara para mantener solo las filas deseadas
        mascara = np.ones(tpm_sparse.shape[0], dtype=bool)
        mascara[elementos_a_excluir] = False
        
        # Extraer submatriz
        tpm_marginalizada = tpm_sparse[mascara]
        
        # Normalizar para mantener propiedades de probabilidad
        return CompactRepresentationStrategy._normalizar_matriz_sparse(tpm_marginalizada)

    @staticmethod
    def _normalizar_matriz_sparse(matriz: csr_matrix) -> csr_matrix:
        """
        Normaliza una matriz dispersa para que cada fila sume 1.
        
        Args:
            matriz (csr_matrix): Matriz a normalizar
        
        Returns:
            csr_matrix: Matriz normalizada
        """
        # Calcular suma de cada fila
        sumas_filas = np.array(matriz.sum(axis=1)).flatten()
        
        # Evitar división por cero
        sumas_filas[sumas_filas == 0] = 1
        
        # Crear matriz diagonal de inversos
        inv_sumas = sparse.diags(1.0 / sumas_filas)
        
        # Multiplicar matriz original por inverso de sumas
        return inv_sumas @ matriz

    def aplicar_estrategia(self, tpm: np.ndarray, elementos_a_excluir: list = None):
        """
        Método principal para aplicar la estrategia de representación compacta.
        
        Args:
            tpm (np.ndarray): Matriz de probabilidad de transición original
            elementos_a_excluir (list, optional): Elementos a excluir de la matriz
        
        Returns:
            csr_matrix: Matriz procesada
        """
        # Convertir a matriz dispersa
        tpm_sparse = self.crear_tpm_dispersa(tpm)
        
        # Marginalizar si se proporcionan elementos a excluir
        if elementos_a_excluir:
            tpm_sparse = self.marginalizar_por_filas(tpm_sparse, elementos_a_excluir)
        
        return tpm_sparse