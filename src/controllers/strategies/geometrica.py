import time
import numpy as np
from itertools import combinations
import random
import math
from collections import deque, defaultdict

from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile

class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger("GEOMETRIC")

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """Estrategia original con búsqueda exhaustiva"""
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        nodos_mecanismo = list(self.sia_subsistema.dims_ncubos)
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)

        indices_globales = list(self.sia_subsistema.indices_ncubos)
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}
        estados_bin = self.sia_subsistema.estados() if callable(self.sia_subsistema.estados) else self.sia_subsistema.estados
        num_estados = len(estados_bin)

        # Tabla de costos por variable
        tablas_costos = {v: self.calcular_tabla_costos_variable(estados_bin, v) for v in range(len(self.sia_subsistema.ncubos))}
        mejores = None
        mejor_costo = float("inf")

        def todas_biparticiones(nodos):
            for r in range(1, len(nodos)):
                for grupoA in combinations(nodos, r):
                    grupoB = [n for n in nodos if n not in grupoA]
                    yield grupoA, grupoB

        for grupoA, grupoB in todas_biparticiones(nodos_alcance):
            indicesA = [mapa_global_a_local[n] for n in grupoA]
            indicesB = [mapa_global_a_local[n] for n in grupoB]

            costo_total = 0.0
            contador = 0.0
            costos_por_variable = {}
            for v, tabla in tablas_costos.items():
                for i in range(num_estados):
                    for j in range(num_estados):
                        if any(estados_bin[i][idx] != estados_bin[j][idx] for idx in indicesA):
                            costo_total += tabla[i][j]
                            contador += 1.0
                costos_por_variable[v] = costo_total/contador if contador > 0 else float("inf")
            costo_total = min(costos_por_variable.values())
                
            if costo_total < mejor_costo:
                mejor_costo = costo_total
                mejores = (
                    [(1, n) for n in grupoA],
                    [(1, n) for n in grupoB] + [(0, n) for n in nodos_mecanismo]
                )

        fmt_mip = fmt_biparte_q(mejores[0], mejores[1]) if mejores else "No se encontró partición válida"

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_costo,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
    
    def simulated_annealing_bipartition(self, estados_bin, tabla_costos, temp_inicial=1000, temp_final=0.1, alpha=0.95):
        """
        Encuentra bipartición óptima usando recocido simulado
        """
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)
        
        # Solución inicial aleatoria
        k = max(1, len(nodos_alcance) // 2)
        grupoA = random.sample(nodos_alcance, k)
        grupoB = [n for n in nodos_alcance if n not in grupoA]
        
        mejor_solucion = (grupoA, grupoB)
        mejor_costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos)
        
        temperatura = temp_inicial
        solucion_actual = mejor_solucion
        costo_actual = mejor_costo
        
        while temperatura > temp_final:
            # Generar vecino: intercambiar un nodo entre grupos
            nueva_grupoA, nueva_grupoB = self._generar_vecino(solucion_actual)
            nuevo_costo = self._evaluar_biparticion(nueva_grupoA, nueva_grupoB, estados_bin, tabla_costos)
            
            # Criterio de aceptación
            delta = nuevo_costo - costo_actual
            if delta < 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual = (nueva_grupoA, nueva_grupoB)
                costo_actual = nuevo_costo
                
                if nuevo_costo < mejor_costo:
                    mejor_solucion = solucion_actual
                    mejor_costo = nuevo_costo
            
            temperatura *= alpha
        
        return mejor_solucion, mejor_costo

    def simulated_annealing_refinement(self, solucion_inicial, estados_bin, tabla_costos, temp_inicial=100, temp_final=0.01, alpha=0.9):
        """
        Refinamiento usando simulated annealing desde una solución inicial
        """
        temperatura = temp_inicial
        solucion_actual = solucion_inicial
        costo_actual = self._evaluar_biparticion(solucion_actual[0], solucion_actual[1], estados_bin, tabla_costos)
        
        mejor_solucion = solucion_actual
        mejor_costo = costo_actual
        
        while temperatura > temp_final:
            nueva_grupoA, nueva_grupoB = self._generar_vecino(solucion_actual)
            nuevo_costo = self._evaluar_biparticion(nueva_grupoA, nueva_grupoB, estados_bin, tabla_costos)
            
            delta = nuevo_costo - costo_actual
            if delta < 0 or random.random() < math.exp(-delta / temperatura):
                solucion_actual = (nueva_grupoA, nueva_grupoB)
                costo_actual = nuevo_costo
                
                if nuevo_costo < mejor_costo:
                    mejor_solucion = solucion_actual
                    mejor_costo = nuevo_costo
            
            temperatura *= alpha
        
        return mejor_solucion, mejor_costo

    def tabu_search_bipartition(self, estados_bin, tabla_costos, max_iter=1000, tabu_size=50):
        """
        Búsqueda tabú para bipartición óptima
        """
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)
        
        # Solución inicial
        k = max(1, len(nodos_alcance) // 2)
        solucion_actual = (nodos_alcance[:k], nodos_alcance[k:])
        mejor_solucion = solucion_actual
        mejor_costo = self._evaluar_biparticion(*solucion_actual, estados_bin, tabla_costos)
        
        lista_tabu = deque(maxlen=tabu_size)
        
        for iteracion in range(max_iter):
            vecinos = self._generar_todos_vecinos(solucion_actual)
            mejor_vecino = None
            mejor_costo_vecino = float('inf')
            
            for vecino in vecinos:
                # Crear una representación hasheable del vecino
                vecino_key = (tuple(sorted(vecino[0])), tuple(sorted(vecino[1])))
                if vecino_key not in lista_tabu:
                    costo = self._evaluar_biparticion(*vecino, estados_bin, tabla_costos)
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
                # Si no hay vecinos válidos, reiniciar
                break
        
        return mejor_solucion, mejor_costo

    def genetic_algorithm_bipartition(self, estados_bin, tabla_costos, pop_size=100, generations=500):
        """
        Algoritmo genético para encontrar bipartición óptima
        """
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)
        n_nodos = len(nodos_alcance)
        
        # Población inicial
        poblacion = []
        for _ in range(pop_size):
            # Representación: bitstring donde 1 = grupoA, 0 = grupoB
            individuo = [random.randint(0, 1) for _ in range(n_nodos)]
            # Asegurar que ambos grupos tengan al menos un elemento
            if sum(individuo) == 0:
                individuo[0] = 1
            elif sum(individuo) == n_nodos:
                individuo[0] = 0
            poblacion.append(individuo)
        
        for generacion in range(generations):
            # Evaluación
            fitness = []
            for individuo in poblacion:
                grupoA = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 1]
                grupoB = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 0]
                costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos)
                fitness.append(1.0 / (1.0 + costo))  # Maximizar fitness
            
            # Selección por torneo
            nueva_poblacion = []
            for _ in range(pop_size):
                padre1 = self._seleccion_torneo(poblacion, fitness)
                padre2 = self._seleccion_torneo(poblacion, fitness)
                hijo = self._cruce_uniforme(padre1, padre2)
                hijo = self._mutacion(hijo, prob_mut=0.1)
                nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        # Evaluación final para encontrar el mejor
        fitness_final = []
        for individuo in poblacion:
            grupoA = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 1]
            grupoB = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit == 0]
            costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos)
            fitness_final.append(costo)
        
        # Retornar mejor solución
        mejor_idx = fitness_final.index(min(fitness_final))
        mejor_individuo = poblacion[mejor_idx]
        grupoA = [nodos_alcance[i] for i, bit in enumerate(mejor_individuo) if bit == 1]
        grupoB = [nodos_alcance[i] for i, bit in enumerate(mejor_individuo) if bit == 0]
        
        return (grupoA, grupoB), min(fitness_final)

    def spectral_clustering_bipartition(self, estados_bin, tabla_costos):
        """
        Usa clustering espectral basado en la geometría del hipercubo
        """
        try:
            from scipy.linalg import eigh
        except ImportError:
            # Fallback usando numpy
            from numpy.linalg import eigh
        
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)
        n_nodos = len(nodos_alcance)
        
        # Construir matriz de afinidad basada en distancias de Hamming
        W = np.zeros((n_nodos, n_nodos))
        for i in range(n_nodos):
            for j in range(n_nodos):
                if i != j:
                    # Calcular afinidad basada en propiedades del hipercubo
                    dist_hamming = bin(i ^ j).count('1')  # XOR para distancia de Hamming
                    W[i, j] = np.exp(-dist_hamming)
        
        # Matriz laplaciana normalizada
        D = np.sum(W, axis=1)
        # Evitar división por cero
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = np.diag(D) - W
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
        
        # Eigendecomposición
        eigenvals, eigenvecs = eigh(L_norm)
        
        # Usar segundo eigenvector (Fiedler vector) para bipartición
        if eigenvecs.shape[1] > 1:
            fiedler_vector = eigenvecs[:, 1]
        else:
            # Fallback: usar vector aleatorio
            fiedler_vector = np.random.randn(n_nodos)
        
        # Bipartición basada en el signo del vector de Fiedler
        grupoA = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] >= 0]
        grupoB = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] < 0]
        
        if not grupoA or not grupoB:
            # Fallback: usar mediana
            mediana = np.median(fiedler_vector)
            grupoA = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] >= mediana]
            grupoB = [nodos_alcance[i] for i in range(n_nodos) if fiedler_vector[i] < mediana]
        
        # Asegurar que ambos grupos tengan al menos un elemento
        if not grupoA:
            grupoA = [grupoB.pop()]
        elif not grupoB:
            grupoB = [grupoA.pop()]
        
        costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos)
        return (grupoA, grupoB), costo

    def random_search_bipartition(self, estados_bin, tabla_costos, max_iter=1000):
        """
        Búsqueda aleatoria como algoritmo de fallback
        """
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)
        mejor_solucion = None
        mejor_costo = float('inf')
        
        for _ in range(max_iter):
            k = random.randint(1, len(nodos_alcance) - 1)
            grupoA = random.sample(nodos_alcance, k)
            grupoB = [n for n in nodos_alcance if n not in grupoA]
            
            costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos)
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_solucion = (grupoA, grupoB)
        
        return mejor_solucion, mejor_costo

    # Funciones auxiliares
    def _evaluar_biparticion(self, grupoA, grupoB, estados_bin, tabla_costos):
        """Evalúa el costo de una bipartición específica"""
        indices_globales = list(self.sia_subsistema.indices_ncubos)
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}
        
        indicesA = [mapa_global_a_local[n] for n in grupoA if n in mapa_global_a_local]
        num_estados = len(estados_bin)
        
        costo_total = 0.0
        contador = 0
        
        for i in range(num_estados):
            for j in range(num_estados):
                if any(estados_bin[i][idx] != estados_bin[j][idx] for idx in indicesA):
                    int_i = self._binario_a_entero(estados_bin[i])
                    int_j = self._binario_a_entero(estados_bin[j])
                    key = (int_i, int_j)
                    if key in tabla_costos:
                        costo_total += tabla_costos[key]
                        contador += 1
        
        return costo_total / max(contador, 1)

    def _generar_vecino(self, solucion):
        """Genera un vecino intercambiando un nodo entre grupos"""
        grupoA, grupoB = solucion
        
        if not grupoA or not grupoB:
            return solucion
        
        if random.choice([True, False]) and len(grupoA) > 1:
            # Mover de A a B
            nodo = random.choice(grupoA)
            nueva_grupoA = [n for n in grupoA if n != nodo]
            nueva_grupoB = grupoB + [nodo]
        elif len(grupoB) > 1:
            # Mover de B a A
            nodo = random.choice(grupoB)
            nueva_grupoA = grupoA + [nodo]
            nueva_grupoB = [n for n in grupoB if n != nodo]
        else:
            return solucion  # No se puede generar vecino válido
        
        return nueva_grupoA, nueva_grupoB

    def _generar_todos_vecinos(self, solucion):
        """Genera todos los vecinos posibles de una solución"""
        grupoA, grupoB = solucion
        vecinos = []
        
        # Mover cada nodo de A a B (si A tiene más de un elemento)
        if len(grupoA) > 1:
            for nodo in grupoA:
                nueva_grupoA = [n for n in grupoA if n != nodo]
                nueva_grupoB = grupoB + [nodo]
                vecinos.append((nueva_grupoA, nueva_grupoB))
        
        # Mover cada nodo de B a A (si B tiene más de un elemento)
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
        
        # Asegurar que ambos grupos tengan al menos un elemento
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
        
        # Asegurar que ambos grupos tengan al menos un elemento
        if sum(hijo) == 0:
            hijo[random.randint(0, len(hijo) - 1)] = 1
        elif sum(hijo) == len(hijo):
            hijo[random.randint(0, len(hijo) - 1)] = 0
        
        return hijo

    def calcular_tabla_costos_unificada(self, estados_bin):
        """
        Calcula una tabla de costos unificada para todos los pares de estados
        """
        n_estados = len(estados_bin)
        tabla_costos = {}
        
        # Calcular costos mínimos entre todas las variables
        for v in range(len(self.sia_subsistema.ncubos)):
            tabla_v = self.calcular_tabla_costos_variable(estados_bin, v)
            
            for i in range(n_estados):
                for j in range(n_estados):
                    int_i = self._binario_a_entero(estados_bin[i])
                    int_j = self._binario_a_entero(estados_bin[j])
                    key = (int_i, int_j)
                    
                    if key not in tabla_costos:
                        tabla_costos[key] = tabla_v[i][j]
                    else:
                        tabla_costos[key] = min(tabla_costos[key], tabla_v[i][j])
        
        return tabla_costos

    def calcular_tabla_costos_variable(self, estados_bin, v_idx):
        """
        Cálculo de la Tabla de Costos T mediante BFS modificado,
        siguiendo fielmente el Algoritmo 1 proporcionado.
        """
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
        try:
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx]
        except (IndexError, AttributeError):
            return 0.0

    def _binario_a_entero(self, binario):
        return int("".join(str(b) for b in binario), 2)

    def _distancia_hamming(self, v, u):
        return sum(b1 != b2 for b1, b2 in zip(v, u))