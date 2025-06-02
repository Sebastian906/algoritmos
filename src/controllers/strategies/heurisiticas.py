import random
import math
from collections import deque

import numpy as np

class Heuristicas:
    def __init__(self, seed=42):
        self.sia_subsistema = None
        self.mapa_global_a_local = None
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def set_sia_context(self, sia_subsistema, mapa_global_a_local):
        """Establece el contexto necesario para los cálculos"""
        self.sia_subsistema = sia_subsistema
        self.mapa_global_a_local = mapa_global_a_local

    def simulated_annealing_bipartition(self, estados_bin, tabla_costos, indices_ncubos, seed=None):
        """
        Encuentra bipartición óptima usando recocido simulado
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        nodos_alcance = sorted(list(indices_ncubos))
        
        # Solución inicial aleatoria
        k = max(1, len(nodos_alcance) // 2)
        grupoA = nodos_alcance[:k]
        grupoB = nodos_alcance[k:]
        
        mejor_solucion = (grupoA, grupoB)
        mejor_costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)
        
        temperatura = 1000
        solucion_actual = mejor_solucion
        costo_actual = mejor_costo
        
        iteraciones = 0
        max_iteraciones = 1000
        
        while temperatura > 0.1 and iteraciones < max_iteraciones:
            # Generar vecino: intercambiar un nodo entre grupos
            nueva_grupoA, nueva_grupoB = self._generar_vecino(solucion_actual)
            nuevo_costo = self._evaluar_biparticion(nueva_grupoA, nueva_grupoB, estados_bin, tabla_costos, indices_ncubos)

            # Criterio de aceptación
            delta = nuevo_costo - costo_actual
            if delta < 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual = (nueva_grupoA, nueva_grupoB)
                costo_actual = nuevo_costo
                
                if nuevo_costo < mejor_costo:
                    mejor_solucion = solucion_actual
                    mejor_costo = nuevo_costo
            
            temperatura *= 0.95
            iteraciones += 1
        
        return mejor_solucion, mejor_costo

    def simulated_annealing_refinement(self, solucion_inicial, estados_bin, tabla_costos, indices_ncubos, temp_inicial=100, temp_final=0.01, alpha=0.9):
        """
        Refinamiento usando simulated annealing desde una solución inicial
        """
        temperatura = temp_inicial
        solucion_actual = solucion_inicial
        costo_actual = self._evaluar_biparticion(solucion_actual[0], solucion_actual[1], estados_bin, tabla_costos, indices_ncubos)
        
        mejor_solucion = solucion_actual
        mejor_costo = costo_actual
        
        while temperatura > temp_final:
            nueva_grupoA, nueva_grupoB = self._generar_vecino(solucion_actual)
            nuevo_costo = self._evaluar_biparticion(nueva_grupoA, nueva_grupoB, estados_bin, tabla_costos, indices_ncubos)
            
            delta = nuevo_costo - costo_actual
            if delta < 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual = (nueva_grupoA, nueva_grupoB)
                costo_actual = nuevo_costo
                
                if nuevo_costo < mejor_costo:
                    mejor_solucion = solucion_actual
                    mejor_costo = nuevo_costo
            
            temperatura *= alpha
        
        return mejor_solucion, mejor_costo

    def tabu_search_bipartition(self, estados_bin, tabla_costos, indices_ncubos, max_iter=1000, tabu_size=50):
        """
        Búsqueda tabú para bipartición óptima
        """
        nodos_alcance = list(indices_ncubos)
        
        # Solución inicial
        k = max(1, len(nodos_alcance) // 2)
        solucion_actual = (nodos_alcance[:k], nodos_alcance[k:])
        mejor_solucion = solucion_actual
        mejor_costo = self._evaluar_biparticion(*solucion_actual, estados_bin, tabla_costos, indices_ncubos)
        
        lista_tabu = deque(maxlen=tabu_size)
        
        for _ in range(max_iter):
            vecinos = self._generar_todos_vecinos(solucion_actual)
            mejor_vecino = None
            mejor_costo_vecino = float('inf')
            
            for vecino in vecinos:
                vecino_key = (tuple(sorted(vecino[0])), tuple(sorted(vecino[1])))
                if vecino_key not in lista_tabu:
                    costo = self._evaluar_biparticion(*vecino, estados_bin, tabla_costos, indices_ncubos)
                    if costo < mejor_costo_vecino:
                        mejor_vecino = vecino
                        mejor_costo_vecino = costo
            
            if mejor_vecino:
                solucion_actual = mejor_vecino
                vecino_key = (tuple(sorted(mejor_vecino[0])), tuple(sorted(mejor_vecino[1])))
                lista_tabu.append(vecino_key)
                
                if mejor_costo_vecino < mejor_costo:
                    mejor_solucion = mejor_vecino
                    mejor_costo = mejor_costo_vecino
            else:
                break
        
        return mejor_solucion, mejor_costo

    def genetic_algorithm_bipartition(self, estados_bin, tabla_costos, indices_ncubos, pop_size=100, generations=500):
        """
        Algoritmo genético para encontrar bipartición óptima
        """
        nodos_alcance = list(indices_ncubos)
        n_nodos = len(nodos_alcance)
        
        # Población inicial
        poblacion = []
        for _ in range(pop_size):
            individuo = [random.randint(0, 1) for _ in range(n_nodos)]
            if sum(individuo) == 0:
                individuo[0] = 1
            elif sum(individuo) == n_nodos:
                individuo[0] = 0
            poblacion.append(individuo)
        
        for _ in range(generations):
            fitness = []
            for individuo in poblacion:
                grupoA = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 1]
                grupoB = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 0]
                costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)
                fitness.append(1.0 / (1.0 + costo))
            
            nueva_poblacion = []
            for _ in range(pop_size):
                padre1 = self._seleccion_torneo(poblacion, fitness)
                padre2 = self._seleccion_torneo(poblacion, fitness)
                hijo = self._cruce_uniforme(padre1, padre2)
                hijo = self._mutacion(hijo, prob_mut=0.1)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        fitness_final = []
        for individuo in poblacion:
            grupoA = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 1]
            grupoB = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 0]
            costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)
            fitness_final.append(costo)
        
        mejor_idx = fitness_final.index(min(fitness_final))
        mejor_individuo = poblacion[mejor_idx]
        grupoA = [nodos_alcance[i] for i, bit in enumerate(mejor_individuo) if bit == 1]
        grupoB = [nodos_alcance[i] for i, bit in enumerate(mejor_individuo) if bit == 0]
        
        return (grupoA, grupoB), min(fitness_final)

    def spectral_clustering_bipartition(self, estados_bin, tabla_costos, indices_ncubos):
        """
        Usa clustering espectral basado en la geometría del hipercubo
        """
        try:
            from scipy.linalg import eigh
        except ImportError:
            from numpy.linalg import eigh
        
        nodos_alcance = list(indices_ncubos)
        n_nodos = len(nodos_alcance)
        
        W = np.zeros((n_nodos, n_nodos))
        for i in range(n_nodos):
            for j in range(n_nodos):
                if i != j:
                    dist_hamming = bin(i ^ j).count('1')
                    W[i, j] = np.exp(-dist_hamming)
        
        D = np.sum(W, axis=1)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = np.diag(D) - W
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
        
        eigenvals, eigenvecs = eigh(L_norm)
        
        if eigenvecs.shape[1] > 1:
            fiedler_vector = eigenvecs[:, 1]
        else:
            fiedler_vector = np.random.randn(n_nodos)
        
        grupoA = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] >= 0]
        grupoB = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] < 0]
        
        if not grupoA or not grupoB:
            mediana = np.median(fiedler_vector)
            grupoA = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] >= mediana]
            grupoB = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] < mediana]
        
        if not grupoA:
            grupoA = [grupoB.pop()]
        elif not grupoB:
            grupoB = [grupoA.pop()]
        
        costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)
        return (grupoA, grupoB), costo

    def random_search_bipartition(self, estados_bin, tabla_costos, indices_ncubos, max_iter=1000):
        """
        Búsqueda aleatoria como algoritmo de fallback
        """
        nodos_alcance = list(indices_ncubos)
        mejor_solucion = None
        mejor_costo = float('inf')
        
        for _ in range(max_iter):
            k = random.randint(1, len(nodos_alcance) - 1)
            grupoA = random.sample(nodos_alcance, k)
            grupoB = [n for n in nodos_alcance if n not in grupoA]
            
            costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_solucion = (grupoA, grupoB)
        
        return mejor_solucion, mejor_costo

    # Funciones auxiliares
    def _evaluar_biparticion(self, grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos):
        """Evalúa el costo de una bipartición específica"""
        indices_globales = sorted(list(indices_ncubos))
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}

        indicesA = sorted([mapa_global_a_local[n] for n in grupoA if n in mapa_global_a_local])  # ORDENAR
        num_estados = len(estados_bin)
        
        costo_total = 0.0
        contador = 0
        costos_por_variable = {}
        
        # Evaluar cada variable por separado y tomar el mínimo
        for v in sorted(tabla_costos.keys()):
            tabla = tabla_costos[v]
            costo_variable = 0.0
            contador_variable = 0
            
            for i in range(num_estados):
                for j in range(num_estados):
                    # Verificar si los estados difieren en los índices del grupo A
                    if any(estados_bin[i][idx] != estados_bin[j][idx] for idx in indicesA):
                        costo_variable += tabla[i][j]
                        contador_variable += 1
            
            if contador_variable > 0:
                costos_por_variable[v] = costo_variable / contador_variable
            else:
                costos_por_variable[v] = float("inf")
        
        # Retornar el costo mínimo entre todas las variables
        if costos_por_variable:
            return min(costos_por_variable.values())
        else:
            return float("inf")

    def _generar_vecino(self, solucion):
        """Genera un vecino intercambiando un nodo entre grupos"""
        grupoA, grupoB = solucion
        
        if not grupoA or not grupoB:
            return solucion
        
        if random.choice([True, False]) and len(grupoA) > 1:
            nodo = random.choice(grupoA)
            nueva_grupoA = [n for n in grupoA if n != nodo]
            nueva_grupoB = grupoB + [nodo]
        elif len(grupoB) > 1:
            nodo = random.choice(grupoB)
            nueva_grupoA = grupoA + [nodo]
            nueva_grupoB = [n for n in grupoB if n != nodo]
        else:
            return solucion
        
        return nueva_grupoA, nueva_grupoB

    def _generar_todos_vecinos(self, solucion):
        """Genera todos los vecinos posibles de una solución"""
        grupoA, grupoB = solucion
        vecinos = []
        
        if len(grupoA) > 1:
            for nodo in grupoA:
                nueva_grupoA = [n for n in grupoA if n != nodo]
                nueva_grupoB = grupoB + [nodo]
                vecinos.append((nueva_grupoA, nueva_grupoB))
        
        if len(grupoB) > 1:
            for nodo in grupoB:
                nueva_grupoA = grupoA + [nodo]
                nueva_grupoB = [n for n in grupoB if n != nodo]
                vecinos.append((nueva_grupoA, nueva_grupoB))
        
        return vecinos

    def _seleccion_torneo(self, poblacion, fitness, k=3):
        """Selección por torneo para algoritmo genético"""
        indices = random.sample(range(len(poblacion)), min(k, len(poblacion)))
        mejor_idx = max(indices, key=lambda i: fitness[i])
        return poblacion[mejor_idx]

    def _cruce_uniforme(self, padre1, padre2):
        """Cruce uniforme para algoritmo genético"""
        hijo = []
        for i in range(len(padre1)):
            if random.random() < 0.5:
                hijo.append(padre1[i])
            else:
                hijo.append(padre2[i])
        
        if sum(hijo) == 0:
            hijo[0] = 1
        elif sum(hijo) == len(hijo):
            hijo[0] = 0
        
        return hijo

    def _mutacion(self, individuo, prob_mut=0.1):
        """Mutación para algoritmo genético"""
        hijo = individuo.copy()
        for i in range(len(hijo)):
            if random.random() < prob_mut:
                hijo[i] = 1 - hijo[i]
        
        if sum(hijo) == 0:
            hijo[random.randint(0, len(hijo) - 1)] = 1
        elif sum(hijo) == len(hijo):
            hijo[random.randint(0, len(hijo) - 1)] = 0
        
        return hijo

    def _binario_a_entero(self, binario):
        """Convierte una representación binaria a entero"""
        return int("".join(str(b) for b in binario), 2)

    def calcular_tabla_costos_variable(self, estados_bin, v_idx):
        """
        Cálculo de la Tabla de Costos T mediante BFS modificado,
        siguiendo fielmente el Algoritmo 1 proporcionado.
        """
        if not self.sia_subsistema:
            raise ValueError("Contexto SIA no establecido. Llama set_sia_context() primero.")
            
        n = len(estados_bin)
        T = np.zeros((n, n))

        # Valores X[v] para cada estado binario
        val_estado = [self._valor_estado_variable(self._binario_a_entero(e), v_idx) for e in estados_bin]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                d = self._distancia_hamming(estados_bin[i], estados_bin[j])
                gamma = 2.0 ** -d

                # Paso 7: contribución directa
                T[i][j] = abs(val_estado[i] - val_estado[j])

                if d > 1:
                    Q = deque([i])
                    visited = set([i])
                    level = 0

                    while level < d and Q:
                        nextQ = deque()
                        for u in Q:
                            for v in range(n):
                                if self._distancia_hamming(estados_bin[u], estados_bin[v]) == 1 and \
                                self._distancia_hamming(estados_bin[v], estados_bin[j]) < self._distancia_hamming(estados_bin[u], estados_bin[j]):
                                    if v not in visited:
                                        # Línea 18: Acumulación del costo
                                        T[i][j] += gamma * T[i][v]
                                        visited.add(v)
                                        nextQ.append(v)
                        Q = nextQ
                        level += 1

                # Aplicar factor gamma al total acumulado
                T[i][j] *= gamma

        return T

    def _valor_estado_variable(self, idx, v_idx):
        """Obtiene el valor de una variable en un estado específico"""
        try:
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx]
        except (IndexError, AttributeError):
            return 0.0

    def _distancia_hamming(self, v, u):
        """Calcula la distancia de Hamming entre dos vectores binarios"""
        return sum(b1 != b2 for b1, b2 in zip(v, u))