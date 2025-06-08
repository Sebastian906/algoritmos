import random
import math
from collections import deque
from scipy.linalg import eigh
import numpy as np

class Heuristicas:
    def __init__(self, seed, tabla_costos, sia_subsistema, mapa_global_a_local):
        self.sia_subsistema = sia_subsistema
        self.mapa_global_a_local = mapa_global_a_local
        self.tabla_costos = tabla_costos
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _binario_a_entero(self, binario):
        return int("".join(map(str, binario)), 2)

    def _distancia_hamming(self, v, u):
        return sum(b1 != b2 for b1, b2 in zip(v, u))

    def _valor_estado_variable(self, idx, v_idx):
        try:
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx]
        except (IndexError, AttributeError):
            return 0.0

    def _distancia_entre_tablas_costos_reformulada(self, tabla1_idx_local, tabla2_idx_local):
        """
        Calcula distancia entre tablas de costos considerando la estructura geométrica
        del hipercubo y los factores de decrecimiento exponencial γ = 2^(-d).
        """
        if tabla1_idx_local not in self.tabla_costos or tabla2_idx_local not in self.tabla_costos:
            return float('inf')

        T1 = self.tabla_costos[tabla1_idx_local]
        T2 = self.tabla_costos[tabla2_idx_local]
        
        if T1.shape != T2.shape:
            return float('inf')
        
        # Distancia ponderada considerando la estructura del hipercubo
        # Las diferencias en transiciones cercanas (distancia Hamming pequeña) pesan más
        distancia_ponderada = 0.0
        n_estados = T1.shape[0]
        
        for i in range(n_estados):
            for j in range(n_estados):
                if i != j:
                    # Calcular peso basado en distancia topológica
                    # Aproximamos la distancia Hamming usando los índices de estado
                    peso_geometrico = 2.0 ** (-abs(i - j))  # Factor γ simplificado
                    diferencia = abs(T1[i, j] - T2[i, j])
                    distancia_ponderada += peso_geometrico * diferencia
        
        return distancia_ponderada

    def clustering_geometrico_biparticion(self, estados_bin, nodos_alcance, nodos_mecanismo, 
                                        distribuciones_marginales=None, modo='proyecciones'):
        """
        Clustering basado en la metodología geométrica del PDF, usando proyecciones
        marginales y discrepancia tensorial en lugar de clustering espectral tradicional.
        """
        presentes = [(0, np.int8(idx)) for idx in nodos_mecanismo]
        futuros = [(1, np.int8(idx)) for idx in nodos_alcance]
        todos_los_nodos = futuros + presentes
        n_nodos = len(todos_los_nodos)

        if n_nodos <= 1:
            return ([], []), float("inf")

        if modo == 'proyecciones':
            return self._biparticion_por_proyecciones_marginales(
                todos_los_nodos, estados_bin, distribuciones_marginales
            )
        elif modo == 'discrepancia_tensorial':
            return self._biparticion_por_discrepancia_minimal(
                todos_los_nodos, estados_bin
            )
        else:
            # Fallback al método espectral modificado
            return self._clustering_espectral_geometrico(todos_los_nodos, estados_bin)

    def _biparticion_por_proyecciones_marginales(self, todos_los_nodos, estados_bin, distribuciones_marginales):
        """
        Bipartición basada en análisis de proyecciones marginales como se describe
        en la sección 3.2.1 del PDF.
        """
        if distribuciones_marginales is None:
            return self._clustering_espectral_geometrico(todos_los_nodos, estados_bin)
        
        # Estrategia 1: Separación por independencia geométrica
        independencias = self._calcular_independencias_desde_marginales(distribuciones_marginales)
        
        if len(independencias) > 0:
            # Encontrar el par de variables más independientes para hacer el corte
            par_mas_independiente = max(independencias.items(), key=lambda x: x[1])
            (var1, var2), _ = par_mas_independiente
            
            # Formar grupos basados en esta separación
            grupo_a = []
            grupo_b = []
            
            for nodo in todos_los_nodos:
                tipo, idx = nodo
                if idx == var1 or (tipo == 0 and idx in self._get_variables_correlacionadas(var1, independencias)):
                    grupo_a.append(nodo)
                elif idx == var2 or (tipo == 1 and idx in self._get_variables_correlacionadas(var2, independencias)):
                    grupo_b.append(nodo)
                else:
                    # Asignar al grupo más pequeño para balance
                    if len(grupo_a) <= len(grupo_b):
                        grupo_a.append(nodo)
                    else:
                        grupo_b.append(nodo)
            
            if len(grupo_a) > 0 and len(grupo_b) > 0:
                costo = self._evaluar_biparticion_discrepancia_tensorial(grupo_a, grupo_b, estados_bin)
                return (grupo_a, grupo_b), costo
        
        # Fallback: separación por tipo
        return self._separacion_por_tipo(todos_los_nodos, estados_bin)

    def _calcular_independencias_desde_marginales(self, distribuciones_marginales):
        """
        Calcula medidas de independencia entre variables basándose en las distribuciones marginales.
        """
        independencias = {}
        
        if 'proyecciones_pares' not in distribuciones_marginales:
            return independencias
        
        for (i, j), dist_conjunta in distribuciones_marginales['proyecciones_pares'].items():
            # Calcular independencia como desviación del producto de marginales
            if i in distribuciones_marginales and j in distribuciones_marginales:
                marginal_i = distribuciones_marginales[i]
                marginal_j = distribuciones_marginales[j]
                
                producto_marginales = np.outer(marginal_i, marginal_j)
                discrepancia = np.linalg.norm(dist_conjunta - producto_marginales, 'fro')
                
                independencias[(i, j)] = discrepancia
        
        return independencias

    def _get_variables_correlacionadas(self, variable, independencias):
        """
        Obtiene variables correlacionadas con la variable dada basándose en independencias.
        """
        correlacionadas = set()
        umbral_correlacion = 0.1  # Ajustable
        
        for (var1, var2), independencia in independencias.items():
            if independencia < umbral_correlacion:  # Baja independencia = alta correlación
                if var1 == variable:
                    correlacionadas.add(var2)
                elif var2 == variable:
                    correlacionadas.add(var1)
        
        return correlacionadas

    def _biparticion_por_discrepancia_minimal(self, todos_los_nodos, estados_bin):
        """
        Encuentra la bipartición que minimiza la discrepancia tensorial
        mediante búsqueda heurística inteligente.
        """
        mejor_biparticion = None
        menor_discrepancia = float('inf')
        
        # Generar biparticiones candidatas de forma inteligente
        candidatas = self._generar_candidatas_inteligentes(todos_los_nodos)
        
        for grupo_a, grupo_b in candidatas:
            if len(grupo_a) == 0 or len(grupo_b) == 0:
                continue
                
            discrepancia = self._evaluar_biparticion_discrepancia_tensorial(
                grupo_a, grupo_b, estados_bin
            )
            
            if discrepancia < menor_discrepancia:
                menor_discrepancia = discrepancia
                mejor_biparticion = (grupo_a, grupo_b)
        
        if mejor_biparticion is None:
            return self._separacion_por_tipo(todos_los_nodos, estados_bin)
        
        return mejor_biparticion, menor_discrepancia

    def _generar_candidatas_inteligentes(self, todos_los_nodos):
        """
        Genera biparticiones candidatas usando estrategias heurísticas inteligentes.
        """
        candidatas = []
        n_nodos = len(todos_los_nodos)
        
        # Estrategia 1: Separación por tipo (presente/futuro)
        presentes = [nodo for nodo in todos_los_nodos if nodo[0] == 0]
        futuros = [nodo for nodo in todos_los_nodos if nodo[0] == 1]
        
        if len(presentes) > 0 and len(futuros) > 0:
            candidatas.append((presentes, futuros))
        
        # Estrategia 2: Separación por índices pares/impares
        pares = [nodo for nodo in todos_los_nodos if nodo[1] % 2 == 0]
        impares = [nodo for nodo in todos_los_nodos if nodo[1] % 2 == 1]
        
        if len(pares) > 0 and len(impares) > 0:
            candidatas.append((pares, impares))
        
        # Estrategia 3: Separación basada en similitud de tablas de costos
        if len(self.tabla_costos) > 1:
            grupos_similares = self._agrupar_por_similitud_tablas()
            if len(grupos_similares) >= 2:
                candidatas.append((grupos_similares[0], grupos_similares[1]))
        
        # Estrategia 4: Bipartición aleatoria balanceada
        nodos_shuffled = todos_los_nodos.copy()
        random.shuffle(nodos_shuffled)
        mid = n_nodos // 2
        candidatas.append((nodos_shuffled[:mid], nodos_shuffled[mid:]))
        
        return candidatas

    def _agrupar_por_similitud_tablas(self):
        """
        Agrupa nodos basándose en la similitud de sus tablas de costos.
        """
        # Calcular matriz de similitud entre todas las tablas
        indices_locales = list(self.tabla_costos.keys())
        n_tablas = len(indices_locales)
        
        if n_tablas < 2:
            return []
        
        similitudes = np.zeros((n_tablas, n_tablas))
        
        for i, idx_i in enumerate(indices_locales):
            for j, idx_j in enumerate(indices_locales):
                if i != j:
                    dist = self._distancia_entre_tablas_costos_reformulada(idx_i, idx_j)
                    similitudes[i, j] = 1.0 / (1.0 + dist) if dist != float('inf') else 0.0
        
        # Clustering simple basado en similitudes
        visitados = set()
        grupos = []
        
        for i in range(n_tablas):
            if i not in visitados:
                grupo_actual = []
                cola = [i]
                
                while cola:
                    actual = cola.pop(0)
                    if actual not in visitados:
                        visitados.add(actual)
                        grupo_actual.append(actual)
                        
                        # Agregar vecinos similares
                        for j in range(n_tablas):
                            if j not in visitados and similitudes[actual, j] > 0.5:
                                cola.append(j)
                
                if grupo_actual:
                    # Convertir índices locales a nodos (tipo, idx_global)
                    nodos_grupo = []
                    for idx_local in grupo_actual:
                        # Buscar el índice global correspondiente
                        for global_idx, local_idx in self.mapa_global_a_local.items():
                            if local_idx == indices_locales[idx_local]:
                                # Determinar tipo basándose en algún criterio (simplificado)
                                tipo = 0 if global_idx < max(self.mapa_global_a_local.keys()) // 2 else 1
                                nodos_grupo.append((tipo, global_idx))
                                break
                    
                    if nodos_grupo:
                        grupos.append(nodos_grupo)
        
        return grupos

    def _clustering_espectral_geometrico(self, todos_los_nodos, estados_bin):
        """
        Versión modificada del clustering espectral que incorpora la geometría
        del hipercubo y los factores de decrecimiento exponencial.
        """
        n_nodos = len(todos_los_nodos)
        
        # Construcción de matriz de afinidad con pesos geométricos
        W = np.zeros((n_nodos, n_nodos))
        
        for i in range(n_nodos):
            for j in range(n_nodos):
                if i == j:
                    W[i, j] = 1.0
                    continue
                
                tipo_i, global_idx_i = todos_los_nodos[i]
                tipo_j, global_idx_j = todos_los_nodos[j]
                
                # Calcular afinidad basada en tablas de costos reformuladas
                if (global_idx_i in self.mapa_global_a_local and 
                    global_idx_j in self.mapa_global_a_local):
                    
                    local_idx_i = self.mapa_global_a_local[global_idx_i]
                    local_idx_j = self.mapa_global_a_local[global_idx_j]
                    
                    dist_tablas = self._distancia_entre_tablas_costos_reformulada(local_idx_i, local_idx_j)
                    
                    if dist_tablas != float('inf'):
                        # Factor de decrecimiento exponencial aplicado a la afinidad
                        distancia_topologica = abs(global_idx_i - global_idx_j)
                        gamma = 2.0 ** (-distancia_topologica)
                        
                        # Afinidad combinando tabla de costos y geometría
                        afinidad_geometrica = gamma * np.exp(-dist_tablas / (dist_tablas + 1.0))
                        W[i, j] = afinidad_geometrica
                    else:
                        W[i, j] = 1e-10
                else:
                    W[i, j] = 1e-10
        
        # Asegurar simetría
        W = (W + W.T) / 2
        
        # Clustering espectral estándar
        D = np.sum(W, axis=1)
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = np.diag(D) - W
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
        
        try:
            eigenvals, eigenvecs = eigh(L_norm)
            idx = eigenvals.argsort()
            fiedler_vector = eigenvecs[:, idx[1]] if n_nodos > 1 else np.random.randn(n_nodos)
        except Exception:
            return self._separacion_por_tipo(todos_los_nodos, estados_bin)
        
        # Bipartición basada en el vector de Fiedler
        mediana = np.median(fiedler_vector)
        grupo_a = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val < mediana]
        grupo_b = [todos_los_nodos[i] for i, val in enumerate(fiedler_vector) if val >= mediana]
        
        if not grupo_a or not grupo_b:
            return self._separacion_por_tipo(todos_los_nodos, estados_bin)
        
        costo = self._evaluar_biparticion_discrepancia_tensorial(grupo_a, grupo_b, estados_bin)
        return (grupo_a, grupo_b), costo

    def _separacion_por_tipo(self, todos_los_nodos, estados_bin):
        """
        Separación simple por tipo de nodo (presente/futuro).
        """
        presentes = [nodo for nodo in todos_los_nodos if nodo[0] == 0]
        futuros = [nodo for nodo in todos_los_nodos if nodo[0] == 1]
        
        if not presentes or not futuros:
            # Si no hay separación por tipo, hacer división por la mitad
            mid = len(todos_los_nodos) // 2
            grupo_a = todos_los_nodos[:mid] if mid > 0 else [todos_los_nodos[0]]
            grupo_b = todos_los_nodos[mid:] if mid < len(todos_los_nodos) else []
            if not grupo_b:
                grupo_b = [todos_los_nodos[-1]]
                grupo_a = todos_los_nodos[:-1]
        else:
            grupo_a = presentes
            grupo_b = futuros
        
        costo = self._evaluar_biparticion_discrepancia_tensorial(grupo_a, grupo_b, estados_bin)
        return (grupo_a, grupo_b), costo

    def _evaluar_biparticion_discrepancia_tensorial(self, grupo_a, grupo_b, estados_bin):
        """
        Evalúa la bipartición usando discrepancia tensorial según la metodología del PDF.
        """
        discrepancia_total = 0.0
        
        # 1. Discrepancia por pérdida de información en proyecciones
        for nodo_a in grupo_a:
            for nodo_b in grupo_b:
                tipo_a, idx_a = nodo_a
                tipo_b, idx_b = nodo_b
                
                # Si ambos índices están en el mapa local, usar tabla de costos
                if (idx_a in self.mapa_global_a_local and 
                    idx_b in self.mapa_global_a_local):
                    
                    local_a = self.mapa_global_a_local[idx_a]
                    local_b = self.mapa_global_a_local[idx_b]
                    
                    if local_a in self.tabla_costos and local_b in self.tabla_costos:
                        # Usar diferencia entre tablas de costos como medida de discrepancia
                        diff_tablas = self._distancia_entre_tablas_costos_reformulada(local_a, local_b)
                        if diff_tablas != float('inf'):
                            discrepancia_total += diff_tablas
                        else:
                            discrepancia_total += 10.0  # Penalización por no poder calcular
        
        # 2. Penalización por desequilibrio en la bipartición
        ratio_grupos = min(len(grupo_a), len(grupo_b)) / max(len(grupo_a), len(grupo_b))
        penalizacion_desequilibrio = (1.0 - ratio_grupos) * 5.0
        
        # 3. Bonificación por coherencia geométrica (variables del mismo tipo juntas)
        bonus_coherencia = 0.0
        tipos_a = set(nodo[0] for nodo in grupo_a)
        tipos_b = set(nodo[0] for nodo in grupo_b)
        
        if len(tipos_a) == 1 or len(tipos_b) == 1:  # Al menos un grupo es homogéneo
            bonus_coherencia = -2.0  # Reducir discrepancia
        
        return discrepancia_total + penalizacion_desequilibrio + bonus_coherencia

    # Método de compatibilidad con la versión anterior
    def spectral_clustering_bipartition2(self, estados_bin, nodos_alcance, nodos_mecanismo, modo='proyecciones'):
        """
        Método de compatibilidad que redirige al nuevo clustering geométrico.
        """
        return self.clustering_geometrico_biparticion(
            estados_bin, nodos_alcance, nodos_mecanismo, 
            distribuciones_marginales=None, modo=modo
        )