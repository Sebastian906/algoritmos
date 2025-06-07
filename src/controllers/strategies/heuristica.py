import random
import math
from collections import deque
from scipy.linalg import eigh
import numpy as np

class Heuristicas:
    def __init__(self, seed, tabla_costos,sia_subsistema, mapa_global_a_local):
        self.sia_subsistema = sia_subsistema
        self.mapa_global_a_local = mapa_global_a_local
        self.tabla_costos = tabla_costos
        self.seed = seed
    def _binario_a_entero(self, binario):
        return int("".join(map(str, binario)), 2)

    def _distancia_hamming(self, v, u):
        return sum(b1 != b2 for b1, b2 in zip(v, u))

    def _valor_estado_variable(self, idx, v_idx):
        # Esta función dependerá de self.sia_subsistema, que ya se pasa en el constructor.
        try:
            # Asumiendo que sia_subsistema.ncubos es accesible y contiene los datos.
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx]
        except (IndexError, AttributeError):
            # Podría ser un logger en un entorno real
            # print(f"Error al acceder al valor del estado para v_idx={v_idx}, idx={idx}")
            return 0.0 # Valor por defecto si hay un error
    def _distancia_entre_tablas_costos(self, tabla1_idx_local, tabla2_idx_local):
        """
        Calcula una distancia entre dos tablas de costos (matrices T_v_idx).
        Se usa la norma de Frobenius de la diferencia.
        Asume que tabla1_idx_local y tabla2_idx_local son índices locales (v_idx)
        para acceder a self.tabla_costos.
        """
        if tabla1_idx_local not in self.tabla_costos or tabla2_idx_local not in self.tabla_costos:
            # Si alguna variable no tiene su tabla de costos, la distancia es "infinita".
            # Esto puede ocurrir si un 'global_idx' no está en el 'alcance' del subsistema.
            return float('inf')

        T1 = self.tabla_costos[tabla1_idx_local]
        T2 = self.tabla_costos[tabla2_idx_local]
        
        # Asegurarse de que las matrices tienen la misma forma para la resta
        if T1.shape != T2.shape:
            # Esto no debería pasar si todas las T_v_idx tienen la misma forma (N_estados, N_estados)
            # Pero es una buena verificación de robustez.
            return float('inf')
            
        # Norma de Frobenius de la diferencia
        return np.linalg.norm(T1 - T2, 'fro')
        
    def spectral_clustering_bipartition2(self, estados_bin, nodos_alcance, nodos_mecanismo, modo='aislado'):
        """
        Clustering espectral para bipartición de nodos (variables tipo/índice).
        La matriz de afinidad W se construye basándose en la distancia entre las tablas de costos
        asociadas a las variables de cada nodo.
        """
        # Construcción de la lista de nodos como tuplas (tipo, índice global)
        # nodos_alcance (alcance, futuro) y dims_ncubos (mecanismo, presente)
        presentes = [(0, np.int8(idx)) for idx in nodos_mecanismo]
        futuros = [(1, np.int8(idx)) for idx in nodos_alcance]
        todos_los_nodos = futuros + presentes # El orden es importante para la indexación de W
        n_nodos = len(todos_los_nodos)

        if n_nodos <= 1: # No se puede biparticionar con 0 o 1 nodo
            return ([], []), float("inf")

        # --- CONSTRUCCIÓN DE LA MATRIZ DE AFINIDAD W ---
        W = np.zeros((n_nodos, n_nodos))
        
        # Primero, obtener todas las distancias entre las tablas de costos para normalización
        all_distances = []
        for i in range(n_nodos):
            for j in range(i + 1, n_nodos): # Solo la mitad superior para simetría
                tipo_i, global_idx_i = todos_los_nodos[i]
                tipo_j, global_idx_j = todos_los_nodos[j]

                # Mapear global_idx a local_idx (v_idx)
                # Solo si el global_idx corresponde a una variable en el subsistema para la cual se calculó la tabla de costos
                if global_idx_i in self.mapa_global_a_local and global_idx_j in self.mapa_global_a_local:
                    local_idx_i = self.mapa_global_a_local[global_idx_i]
                    local_idx_j = self.mapa_global_a_local[global_idx_j]
                    dist = self._distancia_entre_tablas_costos(local_idx_i, local_idx_j)
                    if dist != float('inf'):
                        all_distances.append(dist)
                # Si una variable no tiene una tabla de costos (no está en el subsistema analizado),
                # se considera que la afinidad es 0 (o la distancia infinita).

        max_dist = max(all_distances) if all_distances else 1.0 # Evitar división por cero
        
        for i in range(n_nodos):
            for j in range(n_nodos):
                if i == j:
                    W[i, j] = 1.0 # Alta afinidad de un nodo consigo mismo
                    continue

                tipo_i, global_idx_i = todos_los_nodos[i]
                tipo_j, global_idx_j = todos_los_nodos[j]
                
                if global_idx_i in self.mapa_global_a_local and global_idx_j in self.mapa_global_a_local:
                    local_idx_i = self.mapa_global_a_local[global_idx_i]
                    local_idx_j = self.mapa_global_a_local[global_idx_j]
                    
                    dist_tablas = self._distancia_entre_tablas_costos(local_idx_i, local_idx_j)
                    
                    # Añadir una componente de distancia simple entre tipos (futuro/presente)
                    # y un peso para la distancia de Hamming entre los IDs (si es aplicable, pero mejor usar la distancia de tablas).
                    # La distancia tipo + índice es la que tenías originalmente.
                    # Mantendremos la afinidad basada principalmente en las tablas de costos,
                    # pero puedes ajustar el peso de la "distancia de identificadores" si es deseado.
                    dist_identificadores = abs(tipo_i - tipo_j) + abs(global_idx_i - global_idx_j) # Puedes ponderar esto
                    
                    # Normalizar la distancia de las tablas y combinarla
                    if dist_tablas != float('inf') and max_dist > 0:
                        normalized_dist_tablas = dist_tablas / max_dist
                        # Combinamos las distancias. Esto es una heurística.
                        # Una suma simple (o promedio) de distancias antes de exp.
                        # Puedes ajustar los pesos según la importancia de cada componente.
                        total_dist = normalized_dist_tablas + (dist_identificadores / (max(nodos_alcance + nodos_mecanismo) + 1)) # Normalizar identificadores
                        W[i, j] = np.exp(-total_dist)
                    else:
                        W[i, j] = 1e-10 # Afinidad muy baja si no se pudo calcular la distancia entre tablas.
                else:
                    W[i, j] = 1e-10 # Si no hay tablas de costos para los índices, afinidad muy baja

        # Asegurar simetría explícitamente, aunque la construcción ya debería serlo
        W = (W + W.T) / 2

        # --- Cálculo de la Laplaciana Normalizada ---
        D = np.sum(W, axis=1)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = np.diag(D) - W
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv

        # --- Descomposición espectral ---
        try:
            eigenvals, eigenvecs = eigh(L_norm)
            idx = eigenvals.argsort()
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            fiedler_vector = eigenvecs[:, 1] if n_nodos > 1 else np.random.randn(n_nodos)

        except Exception as e:
            print(f"Error durante la descomposición espectral: {e}")
            return ([], []), float("inf")

        # --- Asignación de nodos a grupos ---
        grupoA = []
        grupoB = []

        if modo == 'aislado':
            if n_nodos == 0:
                return ([], []), float("inf")
            # Ordenar nodos por el valor del vector de Fiedler
            indices_ordenados = np.argsort(fiedler_vector)
            
            # Aislar el nodo con el menor valor de Fiedler
            # `todos_los_nodos[indices_ordenados[0]]` es la tupla (tipo, idx)
            grupoA = [todos_los_nodos[indices_ordenados[0]]]
            grupoB = [todos_los_nodos[i] for i in indices_ordenados[1:]]
            
            # Asegurar que ambos grupos no estén vacíos. Si solo hay un nodo, grupo B estará vacío.
            if not grupoA or not grupoB:
                # Si solo hay un nodo, o si solo hay un grupo, se considera una bipartición trivial o inválida
                return ([], []), float("inf")

        elif modo == 'signo':
            for i, val in enumerate(fiedler_vector):
                if val < 0:
                    grupoA.append(todos_los_nodos[i])
                else:
                    grupoB.append(todos_los_nodos[i])
            
            # Asegurar que ambos grupos no estén vacíos
            if not grupoA or not grupoB:
                mediana = np.median(fiedler_vector)
                grupoA = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val < mediana]
                grupoB = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val >= mediana]

                if not grupoA or not grupoB:
                    print("Advertencia: No se pudo formar una bipartición no vacía usando el modo 'signo'.")
                    return ([], []), float("inf")
        else:
            raise ValueError("Modo no reconocido. Usa 'aislado' o 'signo'.")
        
        # --- Evaluación del costo ---
        # Ahora _evaluar_biparticion_corregida debe recibir los grupos como tuplas (tipo, idx)
        # y usar self.tabla_costos para calcular el costo.
        costo = self._evaluar_biparticion_corregida(grupoA, grupoB, estados_bin, nodos_alcance, nodos_mecanismo)
        
        return (grupoA, grupoB), costo


    def _evaluar_biparticion_corregida(self, grupoA, grupoB, estados_bin, nodos_alcance, nodos_mecanismo):
        """
        Evalúa el costo de una bipartición de nodos (variables tipo/índice).
        Ahora calcula el costo de "corte" de la bipartición de variables.
        """
        costo_total = 0.0
        # `grupoA` y `grupoB` son listas de tuplas (tipo, global_idx)
        
        # Para el costo del corte de la bipartición en el contexto de variables (nodos):
        # Podríamos definir el costo de un corte como la suma de las afinidades (o la suma de costos inversos)
        # entre nodos que están en grupos diferentes.
        
        # Utilizamos la misma lógica de afinidad inversa que en spectral_clustering_bipartition
        # para que la evaluación del costo sea consistente.

        # Calcular el costo de corte como la suma de las afinidades inversas
        # (es decir, las distancias/costos entre los nodos cortados).
        # Esto requiere una nueva matriz de "costo de corte" específica para la evaluación.
        
        # Construir una matriz de costos entre nodos (tipo, idx)
        costo_entre_nodos = np.zeros((len(grupoA) + len(grupoB), len(grupoA) + len(grupoB)))
        
        all_nodes_flat = grupoA + grupoB # Para facilitar la indexación si se desea una matriz
        
        for i_idx, node_i in enumerate(grupoA):
            for j_idx, node_j in enumerate(grupoB):
                tipo_i, global_idx_i = node_i
                tipo_j, global_idx_j = node_j
                
                # Mapear global_idx a local_idx (v_idx)
                if global_idx_i in self.mapa_global_a_local and global_idx_j in self.mapa_global_a_local:
                    local_idx_i = self.mapa_global_a_local[global_idx_i]
                    local_idx_j = self.mapa_global_a_local[global_idx_j]
                    
                    dist_tablas = self._distancia_entre_tablas_costos(local_idx_i, local_idx_j)
                    
                    # Si no se pudo calcular la distancia entre tablas (ej. variable no en subsistema)
                    # o si es infinito, el costo de corte es alto.
                    if dist_tablas == float('inf'):
                        costo_total += 1.0 # Contribución alta al costo
                    else:
                        # Si es una distancia (menor es mejor), el costo de corte es la distancia.
                        # Si la afinidad era exp(-dist), el costo es `dist`.
                        # Pero el Taller Final habla de "Función de Costo para Transiciones entre Estados".
                        # Aquí el costo de la bipartición sería la suma de "desafinidades" entre los grupos.
                        # Usaremos la distancia entre tablas directamente como el "costo" de la conexión.
                        # Puedes sumar una pequeña componente para la distancia de identificadores si se quiere.
                        costo_identificadores = abs(tipo_i - tipo_j) + abs(global_idx_i - global_idx_j)
                        costo_total += dist_tablas + costo_identificadores
                else:
                    costo_total += 1.0 # Costo alto si no se pueden evaluar las tablas de costos

        # Evitar división por cero si no hay conexiones cortadas
        if len(grupoA) == 0 or len(grupoB) == 0:
            return float("inf") # Un grupo vacío no es una bipartición válida

        # Para normalizar el costo, podríamos dividirlo por el número de posibles conexiones cortadas
        # o por el total de variables.
        # Una forma simple es el costo total del corte.
        return costo_total