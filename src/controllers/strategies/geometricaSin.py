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

        # Cache para evitar recálculos
        self._cache_distancias = {}
        self._cache_costos = {}
        self._cache_independencias = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """
        Implementación de la estrategia geométrica para el análisis de biparticiones,
        incorporando el cálculo de tabla de costos, heurísticas y la normalización
        a la forma canónica de la bipartición final.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
    
        nodos_mecanismo = sorted(list(self.sia_subsistema.dims_ncubos)) # global_idx para el mecanismo
        nodos_alcance = sorted(list(self.sia_subsistema.indices_ncubos)) # global_idx para el alcance
        
        # Early exit para casos triviales
        if len(nodos_mecanismo) + len(nodos_alcance) <= 2:
            return self._crear_solucion_trivial()
        
        # indices_globales son todos los índices de las variables del subsistema, ordenados.
        # Es decir, los índices globales que tienen un n-cubo asociado en el subsistema.
        indices_globales_subsistema = sorted(list(set(nodos_alcance + nodos_mecanismo)))
        
        # mapa_global_a_local es crucial para acceder a tabla_costos_por_variable
        # donde la clave es el índice LOCAL (v_idx) de la variable dentro de self.sia_subsistema.ncubos
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales_subsistema)}
        
        # Asignar como atributo de la instancia
        self.mapa_global_a_local = mapa_global_a_local
        
        # Obtener los estados binarios del subsistema. Estos son los estados del hipercubo.
        # Estados optimizados - evitar reconstrucción innecesaria
        estados_bin = self._obtener_estados_optimizado()
        
        if len(estados_bin) == 0:
            return self._crear_solucion_fallback()
        
        # 1. Cálculo de tabla de costos con paralelización y cache
        self.logger.info("Calculando tabla de costos optimizada")
        tabla_costos_por_variable = self._calcular_costos_paralelo(estados_bin)
        
        # 2. Distribuciones marginales con cálculo lazy
        distribuciones_marginales = self._calcular_distribuciones_lazy(estados_bin)
        
        # 3. Evaluación ultra-rápida con heurísticas avanzadas
        mejor_biparticion, mejor_costo = self._evaluar_biparticiones_ultra_rapido(
            estados_bin, nodos_alcance, nodos_mecanismo, 
            distribuciones_marginales, tabla_costos_por_variable
        )
        
        # 4. Formateo del resultado
        if mejor_biparticion:
            biparticion_canonica = self._obtener_biparticion_canonica_rapida(mejor_biparticion)
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

    def _obtener_estados_optimizado(self):
        """Obtiene estados de manera optimizada, evitando reconstrucciones innecesarias."""
        try:
            if hasattr(self.sia_subsistema, 'estados') and callable(self.sia_subsistema.estados):
                return self.sia_subsistema.estados()
            elif hasattr(self.sia_subsistema, 'estados'):
                return self.sia_subsistema.estados
            else:
                # Fallback: generar estados mínimos necesarios
                n_vars = getattr(self.sia_subsistema, 'n_variables', 0)
                if n_vars > 0:
                    return [tuple(int(b) for b in format(i, f'0{n_vars}b')) for i in range(2**min(n_vars, 10))]
                return []
        except Exception as e:
            self.logger.warning(f"Error obteniendo estados, usando fallback: {e}")
            return []

    def _calcular_costos_paralelo(self, estados_bin):
        """Cálculo paralelo de tabla de costos con optimizaciones."""
        if len(self.sia_subsistema.ncubos) == 0:
            return {}
        
        variables_ordenadas = list(range(len(self.sia_subsistema.ncubos)))
        tabla_costos_por_variable = {}
        
        # Para pocos estados, no usar paralelización (overhead no vale la pena)
        if len(estados_bin) < 50 or len(variables_ordenadas) < 4:
            for v_idx in variables_ordenadas:
                tabla_costos_por_variable[v_idx] = self._calcular_tabla_costos_optimizada(estados_bin, v_idx)
        else:
            # Paralelización para casos grandes
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(variables_ordenadas))) as executor:
                future_to_var = {
                    executor.submit(self._calcular_tabla_costos_optimizada, estados_bin, v_idx): v_idx 
                    for v_idx in variables_ordenadas
                }
                
                for future in concurrent.futures.as_completed(future_to_var):
                    v_idx = future_to_var[future]
                    try:
                        tabla_costos_por_variable[v_idx] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error calculando costos para variable {v_idx}: {e}")
                        tabla_costos_por_variable[v_idx] = np.zeros((len(estados_bin), len(estados_bin)))
        
        return tabla_costos_por_variable

    @lru_cache(maxsize=1000)
    def _distancia_hamming_optimizada(self, estado1, estado2):
        """Distancia de Hamming optimizada sin cache."""
        # Si son arrays de numpy, usar operaciones vectorizadas
        if hasattr(estado1, 'shape') and hasattr(estado2, 'shape'):
            return np.sum(estado1 != estado2)
        else:
            return sum(b1 != b2 for b1, b2 in zip(estado1, estado2))

    def _calcular_tabla_costos_optimizada(self, estados_bin, v_idx):
        """Versión optimizada del cálculo de tabla de costos."""
        n = len(estados_bin)
        T = np.zeros((n, n))

        # Precalcular todos los valores de estado una sola vez
        valores_estado = np.array([self._valor_estado_variable_optimizado(estado, v_idx) for estado in estados_bin])

        # Convertir estados a array de numpy para operaciones vectorizadas
        if len(estados_bin) > 0:
            try:
                estados_array = np.array(estados_bin)
                # Usar broadcasting para calcular todas las distancias de una vez
                if estados_array.ndim == 2:
                    # Calcular distancias usando broadcasting
                    distancias = np.sum(estados_array[:, None, :] != estados_array[None, :, :], axis=2)
                else:
                    # Fallback para casos complejos
                    distancias = np.zeros((n, n), dtype=np.int8)
                    for i in range(n):
                        for j in range(i + 1, n):
                            dist = self._distancia_hamming_optimizada(estados_bin[i], estados_bin[j])
                            distancias[i, j] = dist
                            distancias[j, i] = dist
            except:
                # Fallback si no se puede vectorizar
                distancias = np.zeros((n, n), dtype=np.int8)
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = self._distancia_hamming_optimizada(estados_bin[i], estados_bin[j])
                        distancias[i, j] = dist
                        distancias[j, i] = dist

        # Cálculo vectorizado cuando es posible
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                d = distancias[i, j]
                gamma = 2.0 ** (-d)
                diferencia_valores = abs(valores_estado[i] - valores_estado[j])

                T[i, j] = gamma * diferencia_valores

                # Solo calcular recursión para distancias > 1 y casos importantes
                if d > 1 and diferencia_valores > 1e-6:  # Umbral de significancia
                    contribucion = self._calcular_contribucion_bfs_optimizada(
                        i, j, estados_bin, valores_estado, distancias, gamma
                    )
                    T[i, j] += contribucion

        return T

    def _calcular_contribucion_bfs_optimizada(self, origen, destino, estados_bin, valores_estado, distancias, gamma):
        """BFS optimizado con early stopping y límites de profundidad."""
        if distancias[origen, destino] <= 1:
            return 0.0
        
        contribucion = 0.0
        visitados = {origen}
        cola = deque([origen])
        nivel = 0
        max_nivel = min(3, distancias[origen, destino])  # Limitar profundidad
        
        while cola and nivel < max_nivel:
            siguiente_cola = deque()
            nivel += 1
            
            while cola:
                actual = cola.popleft()
                
                # Buscar vecinos que nos acerquen al destino
                for candidato in range(len(estados_bin)):
                    if (candidato not in visitados and 
                        distancias[actual, candidato] == 1 and
                        distancias[candidato, destino] < distancias[actual, destino]):
                        
                        # Contribución ponderada por distancia
                        peso_distancia = 2.0 ** (-distancias[origen, candidato])
                        contribucion_local = peso_distancia * abs(valores_estado[origen] - valores_estado[candidato])
                        contribucion += contribucion_local * 0.5 ** nivel  # Decrecimiento por nivel
                        
                        visitados.add(candidato)
                        siguiente_cola.append(candidato)
            
            cola = siguiente_cola
        
        return contribucion

    def _calcular_distribuciones_lazy(self, estados_bin):
        """Cálculo lazy de distribuciones marginales."""
        if len(estados_bin) == 0:
            return {}
        
        n_variables = len(estados_bin[0])
        distribuciones = {}
        peso_uniforme = 1.0 / len(estados_bin)
        
        # Distribuciones marginales básicas
        estados_array = np.array(estados_bin)
        for v_idx in range(n_variables):
            columna = estados_array[:, v_idx]
            distribuciones[v_idx] = np.array([
                np.sum(columna == 0) * peso_uniforme,
                np.sum(columna == 1) * peso_uniforme
            ])
        
        # Solo calcular proyecciones pares para variables "importantes"
        # (aquellas con distribuciones no uniformes)
        variables_importantes = [
            v for v in range(n_variables) 
            if abs(distribuciones[v][0] - 0.5) > 0.1  # Umbral de no-uniformidad
        ]
        
        distribuciones['proyecciones_pares'] = {}
        for i, v1 in enumerate(variables_importantes):
            for v2 in variables_importantes[i+1:]:
                dist_conjunta = np.zeros((2, 2))
                for estado in estados_bin:
                    dist_conjunta[estado[v1], estado[v2]] += peso_uniforme
                distribuciones['proyecciones_pares'][(v1, v2)] = dist_conjunta
        
        return distribuciones

    def _evaluar_biparticiones_ultra_rapido(self, estados_bin, nodos_alcance, nodos_mecanismo, 
                                          distribuciones_marginales, tabla_costos_por_variable):
        """Evaluación ultra-rápida usando múltiples heurísticas simultáneas."""
        presentes = [(0, np.int8(idx)) for idx in nodos_mecanismo]
        futuros = [(1, np.int8(idx)) for idx in nodos_alcance]
        todos_los_nodos = futuros + presentes

        if len(todos_los_nodos) <= 1:
            return None, float("inf")

        # Estrategia múltiple: evaluar varias heurísticas en paralelo
        candidatas = []
        
        # 1. Separación temporal (más probable que sea óptima)
        if presentes and futuros:
            candidatas.append((presentes, futuros))
        
        # 2. Separación por varianza (variables con comportamiento similar)
        candidatas.extend(self._generar_candidatas_por_varianza(todos_los_nodos, distribuciones_marginales))
        
        # 3. Separación por correlación (solo si tenemos suficientes variables)
        if len(todos_los_nodos) >= 6:
            candidatas.extend(self._generar_candidatas_por_correlacion(todos_los_nodos, tabla_costos_por_variable))
        
        # Evaluar candidatas con early stopping
        mejor_biparticion = None
        mejor_costo = float("inf")
        umbral_aceptable = self._calcular_umbral_aceptable(len(todos_los_nodos))
        
        for i, (grupo_a, grupo_b) in enumerate(candidatas[:5]):  # Limitar a 5 mejores candidatas
            if len(grupo_a) == 0 or len(grupo_b) == 0:
                continue
            
            costo = self._evaluar_biparticion_rapida(grupo_a, grupo_b, distribuciones_marginales, tabla_costos_por_variable)
            
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_biparticion = (grupo_a, grupo_b)
                
                # Early stopping si encontramos una solución suficientemente buena
                if costo < umbral_aceptable:
                    self.logger.info(f"Early stopping en candidata {i+1}, costo: {costo:.4f}")
                    break
        
        return mejor_biparticion, mejor_costo

    def _generar_candidatas_por_varianza(self, todos_los_nodos, distribuciones_marginales):
        """Genera candidatas agrupando por varianza similar."""
        candidatas = []
        
        # Calcular varianza para cada nodo
        varianzas = {}
        for tipo, idx in todos_los_nodos:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in distribuciones_marginales:
                    dist = distribuciones_marginales[local_idx]
                    varianza = dist[0] * dist[1]  # p(1-p) para distribución binaria
                    varianzas[(tipo, idx)] = varianza
                else:
                    varianzas[(tipo, idx)] = 0.25  # Uniforme por defecto
        
        # Ordenar por varianza y crear bipartición
        nodos_ordenados = sorted(varianzas.items(), key=lambda x: x[1])
        mid = len(nodos_ordenados) // 2
        
        grupo_baja_var = [nodo for nodo, _ in nodos_ordenados[:mid]]
        grupo_alta_var = [nodo for nodo, _ in nodos_ordenados[mid:]]
        
        if grupo_baja_var and grupo_alta_var:
            candidatas.append((grupo_baja_var, grupo_alta_var))
        
        return candidatas

    def _generar_candidatas_por_correlacion(self, todos_los_nodos, tabla_costos_por_variable):
        """Genera candidatas basadas en correlaciones entre tablas de costos."""
        candidatas = []
        
        # Calcular matriz de correlación entre variables
        variables_locales = []
        for tipo, idx in todos_los_nodos:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in tabla_costos_por_variable:
                    variables_locales.append((tipo, idx, local_idx))
        
        if len(variables_locales) < 4:
            return candidatas
        
        # Calcular correlaciones usando norma de Frobenius
        correlaciones = {}
        for i, (t1, idx1, l1) in enumerate(variables_locales):
            for t2, idx2, l2 in variables_locales[i+1:]:
                tabla1 = tabla_costos_por_variable[l1]
                tabla2 = tabla_costos_por_variable[l2]
                
                # Correlación aproximada usando diferencia de normas
                norma_diff = np.linalg.norm(tabla1 - tabla2, 'fro')
                correlaciones[((t1, idx1), (t2, idx2))] = norma_diff
        
        # Encontrar el par más correlacionado y menos correlacionado
        if correlaciones:
            par_mas_correlacionado = min(correlaciones.items(), key=lambda x: x[1])
            par_menos_correlacionado = max(correlaciones.items(), key=lambda x: x[1])
            
            # Crear candidata basada en correlación alta
            nodo1, nodo2 = par_mas_correlacionado[0]
            grupo_correlacionado = [nodo1, nodo2]
            grupo_resto = [(tipo, idx) for tipo, idx in todos_los_nodos if (tipo, idx) not in grupo_correlacionado]
            
            if grupo_resto:
                candidatas.append((grupo_correlacionado, grupo_resto))
        
        return candidatas

    def _calcular_umbral_aceptable(self, n_nodos):
        """Calcula un umbral de costo aceptable para early stopping."""
        # Heurística: para problemas pequeños ser más exigente
        if n_nodos <= 4:
            return 0.01
        elif n_nodos <= 8:
            return 0.05
        else:
            return 0.1

    def _evaluar_biparticion_rapida(self, grupo_a, grupo_b, distribuciones_marginales, tabla_costos_por_variable):
        """Evaluación rápida de bipartición usando aproximaciones."""
        costo_total = 0.0
        
        # 1. Costo por diferencias en tablas (aproximación rápida)
        tablas_a = []
        tablas_b = []
        
        for tipo, idx in grupo_a:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in tabla_costos_por_variable:
                    tablas_a.append(tabla_costos_por_variable[local_idx])
        
        for tipo, idx in grupo_b:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in tabla_costos_por_variable:
                    tablas_b.append(tabla_costos_por_variable[local_idx])
        
        # Aproximación: usar promedios de tablas en lugar de comparaciones exhaustivas
        if tablas_a and tablas_b:
            promedio_a = np.mean(tablas_a, axis=0)
            promedio_b = np.mean(tablas_b, axis=0)
            costo_total += np.linalg.norm(promedio_a - promedio_b, 'fro') * 0.1
        
        # 2. Penalización por desequilibrio (rápida)
        ratio = min(len(grupo_a), len(grupo_b)) / max(len(grupo_a), len(grupo_b))
        penalizacion = (1.0 - ratio) * 5.0
        
        # CORRECCIÓN: Eliminar bonificación negativa y usar solo factores positivos
        # 3. Factor de coherencia temporal (ahora como penalización o neutro)
        tipos_a = {tipo for tipo, _ in grupo_a}
        tipos_b = {tipo for tipo, _ in grupo_b}
        
        # Si hay separación temporal perfecta, reducir penalización en lugar de dar bonus negativo
        factor_coherencia = 0.0
        if len(tipos_a) == 1 and len(tipos_b) == 1 and tipos_a != tipos_b:
            # Separación temporal perfecta: reducir penalización actual
            factor_coherencia = -min(penalizacion * 0.3, 1.0)  # Máximo 30% de reducción, cap a 1
        elif len(tipos_a) > 1 and len(tipos_b) > 1:
            # Mezcla temporal: penalización adicional
            factor_coherencia = 0.5
        
        costo_final = costo_total + penalizacion + factor_coherencia
        
        return max(0.0, costo_final)

    def _obtener_biparticion_canonica_rapida(self, biparticion):
        """Canonicalización rápida sin transformaciones exhaustivas."""
        grupo_a, grupo_b = biparticion
        
        # Ordenar grupos por criterio simple pero consistente
        grupo_a_ordenado = sorted(grupo_a, key=lambda x: (x[0], x[1]))
        grupo_b_ordenado = sorted(grupo_b, key=lambda x: (x[0], x[1]))
        
        # Forma canónica: grupo lexicográficamente menor primero
        if grupo_a_ordenado <= grupo_b_ordenado:
            return (tuple(grupo_a_ordenado), tuple(grupo_b_ordenado))
        else:
            return (tuple(grupo_b_ordenado), tuple(grupo_a_ordenado))

    def _valor_estado_variable_optimizado(self, estado_binario, v_idx):
        """Versión optimizada del cálculo de valor de estado."""
        try:
            idx_entero = int("".join(map(str, estado_binario)), 2)
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx_entero]
        except (IndexError, AttributeError, ValueError):
            return 0.0

    def _crear_solucion_trivial(self):
        """Crea solución para casos triviales."""
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=0.0,  # Ya es no negativa
            distribucion_subsistema={},
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion="Trivial",
        )

    def _crear_solucion_fallback(self):
        """Crea solución fallback cuando no hay estados válidos."""
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=0.0,  
            distribucion_subsistema={},
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion="Fallback - Sin estados válidos",
        )