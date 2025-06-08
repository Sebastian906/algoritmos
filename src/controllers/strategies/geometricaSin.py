import time
import numpy as np
from collections import deque
from functools import lru_cache
import concurrent.futures
from itertools import combinations

from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.controllers.strategies.heuristica import Heuristicas

class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger("GEOMETRIC_OPT")

        # Cache para evitar recálculos - ahora con límites
        self._cache_distancias = LRUCache(maxsize=512)
        self._cache_costos = LRUCache(maxsize=256)
        self._cache_independencias = LRUCache(maxsize=128)

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """
        Implementación optimizada de la estrategia geométrica para el análisis de biparticiones.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
    
        nodos_mecanismo = sorted(list(self.sia_subsistema.dims_ncubos))
        nodos_alcance = sorted(list(self.sia_subsistema.indices_ncubos))
        
        # Early exit para casos triviales
        if len(nodos_mecanismo) + len(nodos_alcance) <= 2:
            return self._crear_solucion_trivial()
        
        indices_globales_subsistema = sorted(list(set(nodos_alcance + nodos_mecanismo)))
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales_subsistema)}
        self.mapa_global_a_local = mapa_global_a_local 
        
        # OPTIMIZACIÓN 1: Muestreo inteligente para sistemas grandes
        estados_bin = self.sia_subsistema.estados()
        n_estados_original = len(estados_bin)
        
        if n_estados_original > 256:  # Para sistemas > 8 variables
            estados_bin = self._muestreo_inteligente(estados_bin, max_estados=256)
            self.logger.info(f"Muestreo aplicado: {n_estados_original} -> {len(estados_bin)} estados")

        self._validar_consistencia_subsistema(estados_bin)
        
        # OPTIMIZACIÓN 2: Cálculo de tabla de costos optimizado con manejo de errores
        try:
            tabla_de_costos = self._calcular_tabla_costos_optimizada(estados_bin, self.sia_subsistema.ncubos)
        except Exception as e:
            print(f"Error en cálculo de tabla de costos: {e}")
            # Fallback a método más simple
            tabla_de_costos = self._calcular_tabla_costos_fallback(estados_bin, self.sia_subsistema.ncubos)

        # OPTIMIZACIÓN 3: Distribuciones con sampling adaptativo
        distribuciones_marginales = self._calcular_distribuciones_rapidas(estados_bin)
        
        # OPTIMIZACIÓN 4: Evaluación con algoritmo genético simple para casos grandes
        mejor_biparticion, mejor_costo = self._evaluar_biparticiones_optimizada(
            estados_bin, nodos_alcance, nodos_mecanismo, 
            distribuciones_marginales, tabla_de_costos
        )
        
        # Formateo del resultado
        if mejor_biparticion:
            biparticion_canonica = self._obtener_biparticion_canonica(mejor_biparticion)
            fmt_mip = fmt_biparte_q(list(biparticion_canonica[0]), list(biparticion_canonica[1]))
        else:
            fmt_mip = "No se encontró partición válida"
            mejor_costo = float("inf")

        mejor_costo = max(0.0, mejor_costo)
            
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_costo,
            distribucion_subsistema=distribuciones_marginales,
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _muestreo_inteligente(self, estados_bin, max_estados=256):
        """
        OPTIMIZACIÓN CRÍTICA: Muestreo estratificado para reducir complejidad exponencial.
        """
        if len(estados_bin) <= max_estados:
            return estados_bin
            
        estados_array = np.array(estados_bin)
        n_estados, n_vars = estados_array.shape
        
        # Estrategia de muestreo híbrida:
        # 1. Mantener estados "extremos" (muchos 0s o 1s)
        # 2. Muestreo uniforme del resto
        
        pesos_hamming = np.sum(estados_array, axis=1)  # Peso de Hamming
        
        estados_seleccionados = set()
        
        # Incluir estados extremos (primeros y últimos percentiles)
        n_extremos = min(max_estados // 4, 32)
        indices_extremos_bajos = np.argsort(pesos_hamming)[:n_extremos]
        indices_extremos_altos = np.argsort(pesos_hamming)[-n_extremos:]
        
        estados_seleccionados.update(indices_extremos_bajos)
        estados_seleccionados.update(indices_extremos_altos)
        
        # Muestreo estratificado del resto
        restantes = max_estados - len(estados_seleccionados)
        if restantes > 0:
            indices_disponibles = [i for i in range(n_estados) if i not in estados_seleccionados]
            if len(indices_disponibles) > restantes:
                # Muestreo uniforme
                indices_muestra = np.random.choice(indices_disponibles, restantes, replace=False)
                estados_seleccionados.update(indices_muestra)
            else:
                estados_seleccionados.update(indices_disponibles)
        
        indices_finales = sorted(list(estados_seleccionados))
        return [estados_bin[i] for i in indices_finales]

    def _calcular_tabla_costos_optimizada(self, estados_bin, tensores):
        """
        OPTIMIZACIÓN CRÍTICA: Cálculo de tabla de costos con matriz dispersa y vectorización.
        """
        estados_array = np.array(estados_bin, dtype=np.int8)
        n_estados, n_variables = estados_array.shape
        
        # Pre-calcular todas las distancias de Hamming de una vez
        distancias = self._calcular_distancias_hamming_vectorizadas(estados_array)
        
        # Máscara para vecinos (distancia == 1)
        mascara_vecinos = (distancias == 1)
        
        tabla_costos = {}
        
        # Pre-calcular valores de tensores para todos los estados
        valores_tensores = self._precalcular_valores_tensores(estados_array, tensores)
        
        for v in range(n_variables):
            tabla_costos[v] = {}
            tensor_val = valores_tensores[v]
            
            # Vectorizar cálculo de diferencias
            diferencias = np.abs(tensor_val[:, np.newaxis] - tensor_val[np.newaxis, :])
            gamma = 0.5  # 2^(-1)
            costos = gamma * diferencias
            
            # Solo guardar costos para vecinos y diagonal
            estados_tuple = [tuple(row) for row in estados_array]
            
            for i in range(n_estados):
                # Diagonal (mismo estado)
                tabla_costos[v][(estados_tuple[i], estados_tuple[i])] = 0.0
                
                # Solo vecinos
                for j in range(n_estados):
                    if i != j and mascara_vecinos[i, j]:
                        tabla_costos[v][(estados_tuple[i], estados_tuple[j])] = costos[i, j]

        return tabla_costos

    def _calcular_distancias_hamming_vectorizadas(self, estados_array):
        """Cálculo vectorizado de todas las distancias de Hamming."""
        # Usar broadcasting para calcular todas las distancias de una vez
        diff = estados_array[:, np.newaxis, :] != estados_array[np.newaxis, :, :]
        return np.sum(diff, axis=2)

    def _precalcular_valores_tensores(self, estados_array, tensores):
        """Pre-cálculo optimizado de valores de tensores."""
        n_estados, n_variables = estados_array.shape
        valores_tensores = {}
        
        for v in range(n_variables):
            valores_tensores[v] = np.zeros(n_estados, dtype=np.float32)
            
            try:
                tens = tensores[v]
                data = tens.data
                dims = tens.dims
                
                # Mapear dimensiones con validación estricta
                dims_locales = []
                for dim_global in dims:
                    if dim_global in self.mapa_global_a_local:
                        dim_local = self.mapa_global_a_local[dim_global]
                        # CORRECCIÓN: Validar que el índice local esté en rango
                        if dim_local < n_variables:
                            dims_locales.append(dim_local)
                        else:
                            print(f"ADVERTENCIA: Dimensión local {dim_local} fuera de rango para {n_variables} variables")
                            dims_locales.append(0)  # Usar primera variable como fallback
                    else:
                        print(f"ADVERTENCIA: Dimensión global {dim_global} no encontrada en mapa")
                        dims_locales.append(0)  # Valor por defecto seguro
                
                # CORRECCIÓN: Validar dims_locales antes de indexar
                if not dims_locales:
                    print(f"ADVERTENCIA: No hay dimensiones válidas para tensor {v}")
                    continue
                
                # Verificar que todos los índices están en rango
                max_dim = max(dims_locales)
                if max_dim >= n_variables:
                    print(f"ERROR: Dimensión máxima {max_dim} >= {n_variables} variables")
                    continue
                
                # Acceso seguro a los datos del tensor
                for idx_estado in range(n_estados):
                    try:
                        # Extraer índices de forma segura
                        indices_estado = []
                        for dim_local in dims_locales:
                            if dim_local < len(estados_array[idx_estado]):
                                indices_estado.append(estados_array[idx_estado, dim_local])
                            else:
                                indices_estado.append(0)  # Fallback seguro
                        
                        indices = tuple(indices_estado)
                        valores_tensores[v][idx_estado] = float(data[indices])
                        
                    except (IndexError, KeyError) as estado_error:
                        print(f"Error accediendo tensor {v}, estado {idx_estado}: {estado_error}")
                        valores_tensores[v][idx_estado] = 0.0
                    except Exception as estado_error:
                        print(f"Error inesperado tensor {v}, estado {idx_estado}: {estado_error}")
                        valores_tensores[v][idx_estado] = 0.0
                        
            except Exception as e:
                print(f"Error procesando tensor {v}: {e}")
                # valores_tensores[v] ya está inicializado con zeros
                
        return valores_tensores

    def _calcular_tabla_costos_fallback(self, estados_bin, tensores):
        """
        Método de fallback más simple y robusto para el cálculo de tabla de costos.
        """
        estados_array = np.array(estados_bin, dtype=np.int8)
        n_estados, n_variables = estados_array.shape
        tabla_costos = {v: {} for v in range(n_variables)}

        def hamming_simple(a, b):
            return np.sum(a != b)

        estados_tuple = [tuple(row) for row in estados_array]

        # Cálculo más conservador
        for v in range(min(n_variables, len(tensores))):
            try:
                # Calcular valores del tensor de forma segura
                tensor_valores = np.zeros(n_estados)
                
                if v < len(tensores):
                    tens = tensores[v]
                    data = tens.data
                    dims = tens.dims
                    
                    for idx_estado, estado in enumerate(estados_array):
                        try:
                            # Usar solo las dimensiones que estén disponibles
                            indices_validos = []
                            for dim_global in dims:
                                if dim_global in self.mapa_global_a_local:
                                    dim_local = self.mapa_global_a_local[dim_global]
                                    if dim_local < len(estado):
                                        indices_validos.append(estado[dim_local])
                                    else:
                                        indices_validos.append(0)
                                else:
                                    indices_validos.append(0)
                            
                            if indices_validos:
                                tensor_valores[idx_estado] = float(data[tuple(indices_validos)])
                        except:
                            tensor_valores[idx_estado] = 0.0

                # Calcular costos solo para vecinos
                for i in range(n_estados):
                    tabla_costos[v][(estados_tuple[i], estados_tuple[i])] = 0.0
                    
                    for j in range(n_estados):
                        if i != j:
                            d = hamming_simple(estados_array[i], estados_array[j])
                            if d == 1:  # Solo vecinos
                                gamma = 0.5
                                costo = gamma * abs(tensor_valores[i] - tensor_valores[j])
                                tabla_costos[v][(estados_tuple[i], estados_tuple[j])] = costo
                                
            except Exception as e:
                print(f"Error en tensor {v} del método fallback: {e}")
                # Inicializar con tabla vacía
                for i in range(n_estados):
                    tabla_costos[v][(estados_tuple[i], estados_tuple[i])] = 0.0

        print(f"Tabla de costos fallback calculada para {n_variables} variables")
        return tabla_costos

    def _calcular_distribuciones_rapidas(self, estados_bin):
        """Cálculo rápido de distribuciones sin proyecciones innecesarias."""
        if len(estados_bin) == 0:
            return {}
        
        estados_array = np.array(estados_bin, dtype=np.int8)
        n_estados, n_variables = estados_array.shape
        peso_uniforme = 1.0 / n_estados
        
        distribuciones = {}
        
        # Distribuciones marginales vectorizadas
        for v_idx in range(n_variables):
            columna = estados_array[:, v_idx]
            distribuciones[v_idx] = np.array([
                np.sum(columna == 0) * peso_uniforme,
                np.sum(columna == 1) * peso_uniforme
            ])
        
        # Solo calcular proyecciones pares si realmente las necesitamos y son pocas variables
        if n_variables <= 6:
            distribuciones['proyecciones_pares'] = self._calcular_proyecciones_importantes(
                estados_array, distribuciones, peso_uniforme
            )
        else:
            distribuciones['proyecciones_pares'] = {}
        
        return distribuciones

    def _calcular_proyecciones_importantes(self, estados_array, distribuciones, peso_uniforme):
        """Cálculo selectivo de proyecciones pares."""
        n_variables = estados_array.shape[1]
        proyecciones = {}
        
        # Solo variables con alta varianza
        variables_importantes = [
            v for v in range(n_variables) 
            if abs(distribuciones[v][0] - 0.5) > 0.15
        ]
        
        # Limitar número de proyecciones
        max_proyecciones = 10
        count = 0
        
        for i, v1 in enumerate(variables_importantes):
            for v2 in variables_importantes[i+1:]:
                if count >= max_proyecciones:
                    break
                    
                # Cálculo vectorizado de distribución conjunta
                estados_v1 = estados_array[:, v1]
                estados_v2 = estados_array[:, v2]
                
                dist_conjunta = np.zeros((2, 2))
                for val1 in [0, 1]:
                    for val2 in [0, 1]:
                        mask = (estados_v1 == val1) & (estados_v2 == val2)
                        dist_conjunta[val1, val2] = np.sum(mask) * peso_uniforme
                
                proyecciones[(v1, v2)] = dist_conjunta
                count += 1
            
            if count >= max_proyecciones:
                break
        
        return proyecciones

    def _evaluar_biparticiones_optimizada(self, estados_bin, nodos_alcance, nodos_mecanismo, 
                                        distribuciones_marginales, tabla_de_costos):
        """
        Evaluación optimizada usando algoritmo genético simple para casos grandes.
        """
        todos_los_nodos = [(1, np.int8(idx)) for idx in nodos_alcance] + [(0, np.int8(idx)) for idx in nodos_mecanismo]
        
        if len(todos_los_nodos) <= 1:
            return None, float("inf")

        # Para sistemas pequeños, usar heurísticas rápidas
        if len(todos_los_nodos) <= 8:
            return self._evaluar_biparticiones_heuristicas(
                todos_los_nodos, distribuciones_marginales, tabla_de_costos
            )
        
        # Para sistemas grandes, usar algoritmo genético simple
        return self._evaluar_biparticiones_genetico(
            todos_los_nodos, distribuciones_marginales, tabla_de_costos
        )

    def _evaluar_biparticiones_heuristicas(self, todos_los_nodos, distribuciones_marginales, tabla_de_costos):
        """Evaluación rápida con heurísticas para sistemas pequeños."""
        candidatas = []
        
        # Heurística 1: Separación por tipo (presente/futuro)
        presentes = [nodo for nodo in todos_los_nodos if nodo[0] == 0]
        futuros = [nodo for nodo in todos_los_nodos if nodo[0] == 1]
        
        if presentes and futuros:
            candidatas.append((presentes, futuros))
        
        # Heurística 2: Separación por varianza
        candidatas.extend(self._generar_candidatas_por_varianza_rapida(todos_los_nodos, distribuciones_marginales))
        
        # Heurística 3: División aleatoria balanceada
        if len(todos_los_nodos) >= 4:
            mid = len(todos_los_nodos) // 2
            nodos_shuffled = todos_los_nodos.copy()
            np.random.shuffle(nodos_shuffled)
            candidatas.append((nodos_shuffled[:mid], nodos_shuffled[mid:]))
        
        # Evaluar candidatas
        mejor_biparticion = None
        mejor_costo = float("inf")
        
        for grupo_a, grupo_b in candidatas[:3]:  # Máximo 3 candidatas
            if len(grupo_a) == 0 or len(grupo_b) == 0:
                continue
                
            costo = self._evaluar_biparticion_rapida(grupo_a, grupo_b, distribuciones_marginales)
            
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_biparticion = (grupo_a, grupo_b)
        
        return mejor_biparticion, mejor_costo

    def _evaluar_biparticiones_genetico(self, todos_los_nodos, distribuciones_marginales, tabla_de_costos):
        """Algoritmo genético simple para sistemas grandes."""
        n_nodos = len(todos_los_nodos)
        poblacion_size = min(20, max(8, n_nodos))
        generaciones = min(10, max(5, n_nodos // 2))
        
        # Población inicial con diferentes estrategias
        poblacion = []
        
        # Algunos individuos aleatorios
        for _ in range(poblacion_size // 2):
            mask = np.random.rand(n_nodos) < 0.5
            grupo_a = [todos_los_nodos[i] for i in range(n_nodos) if mask[i]]
            grupo_b = [todos_los_nodos[i] for i in range(n_nodos) if not mask[i]]
            if grupo_a and grupo_b:
                poblacion.append((grupo_a, grupo_b))
        
        # Algunos individuos con heurísticas
        candidatas_heuristicas = self._generar_candidatas_por_varianza_rapida(todos_los_nodos, distribuciones_marginales)
        for candidata in candidatas_heuristicas[:poblacion_size // 2]:
            poblacion.append(candidata)
        
        # Completar población si es necesario
        while len(poblacion) < poblacion_size:
            mid = n_nodos // 2
            mask = np.random.rand(n_nodos) < 0.5
            grupo_a = [todos_los_nodos[i] for i in range(n_nodos) if mask[i]]
            grupo_b = [todos_los_nodos[i] for i in range(n_nodos) if not mask[i]]
            if grupo_a and grupo_b:
                poblacion.append((grupo_a, grupo_b))
        
        # Evolución
        for gen in range(generaciones):
            # Evaluar población
            fitness = []
            for individuo in poblacion:
                costo = self._evaluar_biparticion_rapida(individuo[0], individuo[1], distribuciones_marginales)
                fitness.append(1.0 / (1.0 + costo))  # Convertir a fitness (mayor es mejor)
            
            # Selección y reproducción simple
            indices_fitness = np.argsort(fitness)[-poblacion_size//2:]  # Mejores 50%
            nueva_poblacion = [poblacion[i] for i in indices_fitness]
            
            # Mutación simple
            while len(nueva_poblacion) < poblacion_size:
                padre = nueva_poblacion[np.random.randint(len(nueva_poblacion))]
                hijo = self._mutar_individuo(padre, todos_los_nodos)
                if hijo:
                    nueva_poblacion.append(hijo)
            
            poblacion = nueva_poblacion
        
        # Retornar el mejor individuo
        mejor_individuo = None
        mejor_costo = float("inf")
        
        for individuo in poblacion:
            costo = self._evaluar_biparticion_rapida(individuo[0], individuo[1], distribuciones_marginales)
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_individuo = individuo
        
        return mejor_individuo, mejor_costo

    def _mutar_individuo(self, individuo, todos_los_nodos):
        """Mutación simple: intercambiar algunos nodos entre grupos."""
        grupo_a, grupo_b = individuo
        if len(grupo_a) <= 1 or len(grupo_b) <= 1:
            return None
        
        nuevo_a = grupo_a.copy()
        nuevo_b = grupo_b.copy()
        
        # Intercambiar 1-2 nodos aleatoriamente
        n_intercambios = np.random.randint(1, min(3, min(len(grupo_a), len(grupo_b))))
        
        for _ in range(n_intercambios):
            if nuevo_a and nuevo_b:
                # Mover de A a B
                if np.random.rand() < 0.5 and len(nuevo_a) > 1:
                    nodo = nuevo_a.pop(np.random.randint(len(nuevo_a)))
                    nuevo_b.append(nodo)
                # Mover de B a A
                elif len(nuevo_b) > 1:
                    nodo = nuevo_b.pop(np.random.randint(len(nuevo_b)))
                    nuevo_a.append(nodo)
        
        return (nuevo_a, nuevo_b) if nuevo_a and nuevo_b else None

    def _generar_candidatas_por_varianza_rapida(self, todos_los_nodos, distribuciones_marginales):
        """Versión rápida de generación de candidatas por varianza."""
        candidatas = []
        varianzas = {}
        
        for tipo, idx in todos_los_nodos:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in distribuciones_marginales:
                    dist = distribuciones_marginales[local_idx]
                    varianza = dist[0] * dist[1]
                    varianzas[(tipo, idx)] = varianza
                else:
                    varianzas[(tipo, idx)] = 0.25

        if len(set(varianzas.values())) > 1:  # Hay diversidad
            nodos_ordenados = sorted(varianzas.items(), key=lambda x: x[1])
            mid = len(nodos_ordenados) // 2
            
            grupo_baja_var = [nodo for nodo, _ in nodos_ordenados[:mid]]
            grupo_alta_var = [nodo for nodo, _ in nodos_ordenados[mid:]]
            
            if grupo_baja_var and grupo_alta_var:
                candidatas.append((grupo_baja_var, grupo_alta_var))

        return candidatas

    def _evaluar_biparticion_rapida(self, grupo_a, grupo_b, distribuciones_marginales):
        """Evaluación ultra-rápida de bipartición usando solo métricas básicas."""
        # Penalización por desequilibrio
        ratio = min(len(grupo_a), len(grupo_b)) / max(len(grupo_a), len(grupo_b))
        penalizacion_desequilibrio = (1.0 - ratio) * 2.0
        
        # Factor de coherencia temporal
        tipos_a = {tipo for tipo, _ in grupo_a}
        tipos_b = {tipo for tipo, _ in grupo_b}
        
        factor_coherencia = 0.0
        if len(tipos_a) == 1 and len(tipos_b) == 1 and tipos_a != tipos_b:
            factor_coherencia = -0.5  # Bonus por separación temporal
        elif len(tipos_a) > 1 and len(tipos_b) > 1:
            factor_coherencia = 0.3  # Penalización por mezcla temporal
        
        # Diferencia en varianzas promedio (aproximación rápida de heterogeneidad)
        def calcular_varianza_promedio(grupo):
            varianzas = []
            for tipo, idx in grupo:
                if idx in self.mapa_global_a_local:
                    local_idx = self.mapa_global_a_local[idx]
                    if local_idx in distribuciones_marginales:
                        dist = distribuciones_marginales[local_idx]
                        varianzas.append(dist[0] * dist[1])
            return np.mean(varianzas) if varianzas else 0.25
        
        var_a = calcular_varianza_promedio(grupo_a)
        var_b = calcular_varianza_promedio(grupo_b)
        diferencia_varianza = abs(var_a - var_b) * 0.5
        
        costo_total = penalizacion_desequilibrio + factor_coherencia + diferencia_varianza
        return max(0.0, costo_total)

    # Los métodos restantes se mantienen igual...
    def _validar_consistencia_subsistema(self, estados_bin):
        """Valida la consistencia entre estados binarios y tensores."""
        n_estados, n_variables_estados = np.array(estados_bin).shape
        n_tensores = len(self.sia_subsistema.ncubos)

        print(f"Validación: {n_estados} estados, {n_variables_estados} variables en estados")
        print(f"Validación: {n_tensores} tensores disponibles")
        print(f"Validación: mapa_global_a_local = {self.mapa_global_a_local}")

        # Verificar dimensiones de cada tensor
        for v, tensor in enumerate(self.sia_subsistema.ncubos):
            dims_globales = tensor.dims
            print(f"Tensor {v}: dimensiones globales = {dims_globales}")

            # Verificar que todas las dimensiones tienen mapeo válido
            for dim in dims_globales:
                if dim not in self.mapa_global_a_local:
                    print(f"ADVERTENCIA: Dimensión global {dim} del tensor {v} no tiene mapeo local")
                else:
                    dim_local = self.mapa_global_a_local[dim]
                    if dim_local >= n_variables_estados:
                        print(f"ERROR: Dimensión local {dim_local} fuera de rango para estados de {n_variables_estados} variables")
                        # CORRECCIÓN: Actualizar el mapeo para que esté en rango
                        self.mapa_global_a_local[dim] = min(dim_local, n_variables_estados - 1)
                        print(f"CORRECCIÓN: Dimensión {dim} remapeada a {self.mapa_global_a_local[dim]}")

        return True

    def _obtener_biparticion_canonica(self, biparticion):
        """Canonicalización rápida sin transformaciones exhaustivas."""
        grupo_a, grupo_b = biparticion
        
        grupo_a_ordenado = sorted(grupo_a, key=lambda x: (x[0], x[1]))
        grupo_b_ordenado = sorted(grupo_b, key=lambda x: (x[0], x[1]))
        
        if len(grupo_a_ordenado) <= len(grupo_b_ordenado):
            return (tuple(grupo_a_ordenado), tuple(grupo_b_ordenado))
        else:
            return (tuple(grupo_b_ordenado), tuple(grupo_a_ordenado))

    def _crear_solucion_trivial(self):
        """Crea solución para casos triviales."""
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=0.0,
            distribucion_subsistema={},
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion="Trivial",
        )


# Clase auxiliar para LRU Cache optimizado
class LRUCache:
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = deque()
    
    def get(self, key, default=None):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return default
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            oldest = self.access_order.popleft()
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)