import random
import math
from collections import deque

import numpy as np

class Heuristicas:
    def __init__(self, seed):
        self.sia_subsistema = None
        self.mapa_global_a_local = None
        self.seed = seed

    def set_sia_context(self, sia_subsistema, mapa_global_a_local):
        """Establece el contexto necesario para los cálculos"""
        self.sia_subsistema = sia_subsistema
        self.mapa_global_a_local = mapa_global_a_local

    def simulated_annealing_bipartition(self, estados_bin, tabla_costos, indices_ncubos, use_corrected_evaluation=False):
        """
        Algoritmo metaheurístico de optimización global que se inspira en el proceso de recocido de metalurgia, donde su metal se calienta y luego se enfría lentamente para mejorar su estrcutura cristalina.
        Busca la solución óptima a un problema de optimización de manera probabilistica, explorando el espacio de soluciones y, a medida que la "temperatura" del sistema disminuye, aceptando cada vez menos soluciones que no mejoran la función objetivo
        Encuentra bipartición óptima usando recocido simulado
        Heuristica principal implementada
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        nodos_alcance = sorted(list(indices_ncubos))
        
        # Solución inicial aleatoria
        k = max(1, len(nodos_alcance) // 2)
        if len(nodos_alcance) > 3:
            k = random.randint(max(1, len(nodos_alcance) // 3), 
                             min(len(nodos_alcance) - 1, 2 * len(nodos_alcance) // 3))
            
        grupoA = nodos_alcance[:k]
        grupoB = nodos_alcance[k:]
        
        mejor_solucion = (grupoA, grupoB)
        mejor_costo = self._evaluar_biparticion_corregida(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos) if use_corrected_evaluation else self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)

        temperatura = 1000
        solucion_actual = mejor_solucion
        costo_actual = mejor_costo
        
        iteraciones = 0
        max_iteraciones = 1000
        
        while temperatura > 0.1 and iteraciones < max_iteraciones:
            # Generar vecino: intercambiar un nodo entre grupos
            nueva_grupoA, nueva_grupoB = self._generar_vecino_balanceado(solucion_actual)
            
            if use_corrected_evaluation:
                nuevo_costo = self._evaluar_biparticion_corregida(nueva_grupoA, nueva_grupoB, estados_bin, tabla_costos, indices_ncubos)
            else:
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
        Método de optimización matemática, perteneciente a la clase de técnicas de búsqueda local. Aumenta el rendimiento del método de búsqueda local mediante el uso de estructuras de memoria: una vez que una potencial solución es determinada, se la marca como "tabú" de modo que el algoritmo no vuelva a visitar esa posible solución
        """
        nodos_alcance = list(indices_ncubos)
        
        # Solución inicial
        solucion_actual = ([nodos_alcance[0]], nodos_alcance[1:])
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

    def genetic_algorithm_bipartition(self, estados_bin, tabla_costos,  indices_ncubos, pop_size=100, generations=500):
        """
        Algoritmo genético para encontrar bipartición óptima
        Técnica de optimización inspirada en la selección natural y evolución   biológica. Se utiliza para encontrar soluciones a problemas complejos, buscando   la mejor solución dentro de una población de posibles soluciones mediante la  modificación iterativa de la población a través de procesos como la selección,   el cruce y la mutación
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
                grupoA = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit]
                grupoB = [nodos_alcance[i] for i, bit in enumerate(individuo) if not    bit]
                costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin,  tabla_costos, indices_ncubos)
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
            grupoA = [nodos_alcance[i] for i, bit in enumerate(individuo) if bit]
            grupoB = [nodos_alcance[i] for i, bit in enumerate(individuo) if not bit]
            costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin,  tabla_costos, indices_ncubos)
            fitness_final.append(costo)

        mejor_idx = fitness_final.index(min(fitness_final))
        mejor_individuo = poblacion[mejor_idx]
        grupoA = [nodos_alcance[i] for i, bit in enumerate(mejor_individuo) if bit]
        grupoB = [nodos_alcance[i] for i, bit in enumerate(mejor_individuo) if not bit]

        return (grupoA, grupoB), min(fitness_final)

    def spectral_clustering_bipartition(self, estados_bin, tabla_costos, indices_ncubos, dims_ncubos, modo='aislado'):
        """
        Clustering espectral para bipartición de nodos basada en la estructura del hipercubo.
        Permite dos modos:
            - 'aislado': aísla un único nodo (original).
            - 'signo': usa el signo del vector de Fiedler para bipartición completa.
        """
        try:
            from scipy.linalg import eigh
        except ImportError:
            from numpy.linalg import eigh

        # Construcción de la lista de nodos como tuplas (tipo, índice)
        presentes = [(0, np.int8(idx)) for idx in dims_ncubos]
        futuros = [(1, np.int8(idx)) for idx in indices_ncubos]
        todos_los_nodos = futuros + presentes
        n_nodos = len(todos_los_nodos)

        # Matriz de afinidad W basada en distancia de Hamming modificada
        W = np.zeros((n_nodos, n_nodos))
        for i in range(n_nodos):
            for j in range(n_nodos):
                if i != j:
                    tipo_i, idx_i = todos_los_nodos[i]
                    tipo_j, idx_j = todos_los_nodos[j]
                    dist_hamming = abs(tipo_i - tipo_j) + abs(idx_i - idx_j)
                    W[i, j] = np.exp(-dist_hamming)

        # Laplaciana normalizada
        D = np.sum(W, axis=1)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = np.diag(D) - W
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv

        # Descomposición espectral
        eigenvals, eigenvecs = eigh(L_norm)
        fiedler_vector = eigenvecs[:, 1] if eigenvecs.shape[1] > 1 else np.random.randn(n_nodos)
        print("Fiedler vector:", fiedler_vector)

        # Asignación de nodos a grupos según el modo elegido
        if modo == 'aislado':
            # Ordenar nodos y aislar el primero
            indices_ordenados = np.argsort(fiedler_vector)
            grupoA = [todos_los_nodos[indices_ordenados[0]]]
            grupoB = [todos_los_nodos[i] for i in indices_ordenados[1:]]
        elif modo == 'signo':
            grupoA = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val < 0]
            grupoB = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val >= 0]
            # En caso de que un grupo quede vacío (raro), usa la mediana como fallback
            if len(grupoA) == 0 or len(grupoB) == 0:
                mediana = np.median(fiedler_vector)
                grupoA = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val < mediana]
                grupoB = [todos_los_nodos[i] for i in range(n_nodos) if todos_los_nodos[i] not in grupoA]
        else:
            raise ValueError("Modo no reconocido. Usa 'aislado' o 'signo'.")
        # Evaluación del costo usando la función corregida
        costo = self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos, dims_ncubos)
        return (grupoA, grupoB), costo 


    def random_search_bipartition(self, estados_bin, tabla_costos, indices_ncubos, max_iter=1000):
        """
        Búsqueda aleatoria como algoritmo de fallback
        Métodos de optimización numérica qe no requieren el gradiente del problema de optimización, por lo que puede utilizarse en funciones no continuas ni diferenciables. También se conocen como métodos de búsqueda directa, sin derivada o caja negra
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
            print(mejor_costo)
        
        return mejor_solucion, mejor_costo

    # Funciones auxiliares
    def _evaluar_biparticion_corregida(self, grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos, dims_ncubos):
        """
        Evalúa el costo de una bipartición considerando nodos presentes y futuros.
        grupoA y grupoB: listas de tuplas (tipo, idx), donde tipo=1 es futuro, tipo=0 es presente.
        """
        # Compatibilidad con listas de enteros
        if grupoA and isinstance(grupoA[0], (int, np.integer)):
            return self._evaluar_biparticion(grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos)

        indices_ncubos_set = set(int(idx) for idx in indices_ncubos)

        # Extraer índices de nodos futuros en cada grupo
        grupoA_futuros = [int(idx) for tipo, idx in grupoA if tipo == 1 and int(idx) in indices_ncubos_set]
        grupoB_futuros = [int(idx) for tipo, idx in grupoB if tipo == 1 and int(idx) in indices_ncubos_set]

        # Si ambos grupos no tienen nodos futuros, costo infinito
        if not grupoA_futuros and not grupoB_futuros:
            return float("inf")

        # Construir la lista de nodos futuros para el mapeo local
        indices_globales = sorted(list(indices_ncubos))
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}

        # Indices locales de nodos futuros en grupoA
        indicesA = sorted([mapa_global_a_local[n] for n in grupoA_futuros if n in mapa_global_a_local])
        num_estados = len(estados_bin)

        costo_total = 0.0
        total_variables = 0

        for v in sorted(tabla_costos.keys()):
            tabla = tabla_costos[v]
            costo_variable = 0.0
            contador_variable = 0

            for i in range(num_estados):
                for j in range(num_estados):
                    # Si grupoA tiene futuros, evalúa sobre ellos
                    # Si grupoA no tiene futuros (solo presentes), no hay restricción, pero igual se evalúa el costo
                    if indicesA:
                        if any(estados_bin[i][idx] != estados_bin[j][idx] for idx in indicesA):
                            costo_variable += tabla[i][j]
                            contador_variable += 1
                    else:
                        # Si grupoA no tiene futuros, evalúa el costo sobre todos los estados (sin restricción)
                        costo_variable += tabla[i][j]
                        contador_variable += 1

            if contador_variable > 0:
                costo_total += costo_variable / contador_variable
                total_variables += 1

        if total_variables > 0:
            return costo_total / total_variables
        else:
            return float("inf")
        
    def _evaluar_biparticion(self, grupoA, grupoB, estados_bin, tabla_costos, indices_ncubos, dims_ncubos=None):
        """Evaluación original que toma el mínimo costo entre variables"""
        indices_globales = sorted(list(indices_ncubos))
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}

        indicesA = sorted([mapa_global_a_local[n] for n in grupoA if n in mapa_global_a_local])
        num_estados = len(estados_bin)
        
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

    def _generar_vecino_balanceado(self, solucion):
        """
        Genera un vecino evitando biparticiones extremas que aíslen pocos nodos
        """
        grupoA, grupoB = solucion
        
        if not grupoA or not grupoB:
            return solucion
        
        # Evitar que grupos queden con muy pocos elementos
        min_size = max(1, len(grupoA + grupoB) // 4)  # Al menos 25% en cada grupo si es posible
        
        if random.choice([True, False]) and len(grupoA) > min_size:
            nodo = random.choice(grupoA)
            nueva_grupoA = [n for n in grupoA if n != nodo]
            nueva_grupoB = grupoB + [nodo]
        elif len(grupoB) > min_size:
            nodo = random.choice(grupoB)
            nueva_grupoA = grupoA + [nodo]
            nueva_grupoB = [n for n in grupoB if n != nodo]
        else:
            return solucion
        
        return nueva_grupoA, nueva_grupoB
    
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