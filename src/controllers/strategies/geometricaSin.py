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
        
        # mapa_global_a_local es crucial para acceder a tabla_de_costos
        # donde la clave es el índice LOCAL (v_idx) de la variable dentro de self.sia_subsistema.ncubos
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales_subsistema)}
        print(f"Mapa global a local: {mapa_global_a_local}")  # Debugging
        # Asignar como atributo de la instancia
        self.mapa_global_a_local = mapa_global_a_local
        
        # Obtener los estados binarios del subsistema. Estos son los estados del hipercubo.
        estados_bin = self.sia_subsistema.estados()
        
        # 1. Cálculo de tabla de costos con paralelización y cache
        tabla_de_costos = self._calcular_tabla_costos(estados_bin, self.sia_subsistema.ncubos)

        # 2. Distribuciones marginales con cálculo lazy
        distribuciones_marginales = self._calcular_distribuciones_lazy(estados_bin)
        
        # 3. Evaluación ultra-rápida con heurísticas avanzadas
        mejor_biparticion, mejor_costo = self._evaluar_biparticiones(
            estados_bin, nodos_alcance, nodos_mecanismo, 
            distribuciones_marginales, tabla_de_costos
        )
        
        # 4. Formateo del resultado
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

    def _calcular_costos_paralelo(self, estados_bin):
        """Cálculo paralelo de tabla de costos con optimizaciones."""
        if len(self.sia_subsistema.ncubos) == 0:
            return {}
        
        variables_ordenadas = list(range(len(self.sia_subsistema.ncubos)))
        tabla_de_costos = {}
        
        # Para pocos estados, no usar paralelización (overhead no vale la pena)
        if len(estados_bin) < 50 or len(variables_ordenadas) < 4:
            for v_idx in variables_ordenadas:
                tabla_de_costos[v_idx] = self._calcular_tabla_costos(estados_bin, v_idx)
        else:
            # Paralelización para casos grandes
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(variables_ordenadas))) as executor:
                future_to_var = {
                    executor.submit(self._calcular_tabla_costos, estados_bin, v_idx): v_idx 
                    for v_idx in variables_ordenadas
                }
                
                for future in concurrent.futures.as_completed(future_to_var):
                    v_idx = future_to_var[future]
                    try:
                        tabla_de_costos[v_idx] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error calculando costos para variable {v_idx}: {e}")
                        tabla_de_costos[v_idx] = np.zeros((len(estados_bin), len(estados_bin)))
        
        return tabla_de_costos

    @lru_cache(maxsize=1000)
    def _distancia_hamming(self, estado1, estado2):
        """Distancia de Hamming optimizada sin cache."""
        # Si son arrays de numpy, usar operaciones vectorizadas
        if hasattr(estado1, 'shape') and hasattr(estado2, 'shape'):
            return np.sum(estado1 != estado2)
        else:
            return sum(b1 != b2 for b1, b2 in zip(estado1, estado2))

    def _calcular_tabla_costos(self, estados_bin, tensores):
        """
        Calcula la tabla de costos t(i, j) SOLO para pares de estados vecinos (diferencia de un bit).
        tabla_costos[variable][(estado_i, estado_j)] = costo, solo si d(i, j) == 1 o i == j

        - estados_bin: np.ndarray de forma (2^n, n)
        - tensores: lista de NCubes, uno por cada variable
        """
        estados_array = (np.array(estados_bin) > 0).astype(int)
        n_estados, n_variables = estados_array.shape
        tabla_costos = {v: {} for v in range(n_variables)}

        def hamming(a, b):
            return np.sum(a != b)

        estados_tuple = [tuple(row) for row in estados_array]

        # Pre-cálculo: obtener valor del tensor para cada estado
        valores_tensor = {}
        for v in range(n_variables):
            tens = tensores[v]
            data = tens.data
            dims = tens.dims
            valores = [
                float(data[tuple(estado[d] for d in dims)])
                for estado in estados_array
            ]
            valores_tensor[v] = np.array(valores)

        # Solo calcular para vecinos (d == 1) y para i == j
        for v in range(n_variables):
            tensor_val = valores_tensor[v]
            for i in range(n_estados):
                tabla_costos[v][(estados_tuple[i], estados_tuple[i])] = 0.0  # Costo cero para el mismo estado
                for j in range(n_estados):
                    if i == j:
                        continue
                    d = hamming(estados_array[i], estados_array[j])
                    if d == 1:
                        gamma = 2.0 ** -d  # gamma = 0.5
                        costo_directo = abs(tensor_val[i] - tensor_val[j])
                        tabla_costos[v][(estados_tuple[i], estados_tuple[j])] = gamma * costo_directo
        print(f"Tabla de costos calculada para {n_variables} variables y {n_estados} estados.")
        print(f"tabla_costos: {tabla_costos}")  # Debugging
        return tabla_costos


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

    def _evaluar_biparticiones(self, estados_bin, nodos_alcance, nodos_mecanismo, 
                                          distribuciones_marginales, tabla_de_costos):
        """Evaluación ultra-rápida usando múltiples heurísticas simultáneas."""
        presentes = [(0, np.int8(idx)) for idx in nodos_mecanismo]
        futuros = [(1, np.int8(idx)) for idx in nodos_alcance]
        todos_los_nodos = futuros + presentes

        if len(todos_los_nodos) <= 1:
            return None, float("inf")

        # Estrategia múltiple: evaluar varias heurísticas en paralelo
        candidatas = []
        
        # 1. Separación por varianza (variables con comportamiento similar)
        candidatas.extend(self._generar_candidatas_por_varianza(todos_los_nodos, distribuciones_marginales))
        
        # 2. Separación por correlación (solo si tenemos suficientes variables)
        if len(todos_los_nodos) >= 6:
            candidatas.extend(self.s_generar_candidatas_por_correlacion(todos_los_nodos, tabla_de_costos))
        
        # Evaluar candidatas con early stopping
        mejor_biparticion = None
        mejor_costo = float("inf")
        umbral_aceptable = self._calcular_umbral_aceptable(len(todos_los_nodos))
        for i, (grupo_a, grupo_b) in enumerate(candidatas[:5]):  # Limitar a 5 mejores candidatas
            if len(grupo_a) == 0 or len(grupo_b) == 0:
                continue
        
            costo = self._evaluar_biparticion(grupo_a, grupo_b, distribuciones_marginales, tabla_de_costos)
            
            if costo < mejor_costo:
                mejor_costo = costo
                mejor_biparticion = (grupo_a, grupo_b)
            
            # Early stopping si encontramos una solución suficientemente buena
            if costo < umbral_aceptable:
                self.logger.info(f"Early stopping en candidata {i+1}, costo: {costo:.4f}")
                break
        
        return mejor_biparticion, mejor_costo

    def _generar_candidatas_por_varianza(self, todos_los_nodos, distribuciones_marginales):
        """Genera candidatas agrupando por varianza similar, solo si hay diversidad suficiente."""
        candidatas = []# Calcular varianza para cada nodo
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

        # Verificar diversidad de varianza
        valores_varianza = list(varianzas.values())
        if len(valores_varianza) > 1:
            max_var = max(valores_varianza)
            min_var = min(valores_varianza)
            if abs(max_var - min_var) < 1e-6:
                return candidatas  # No hay suficiente diversidad, no generamos biparticiones
        else:
            return candidatas

        # Ordenar por varianza y crear bipartición
        nodos_ordenados = sorted(varianzas.items(), key=lambda x: x[1])
        mid = len(nodos_ordenados) // 2

        grupo_baja_var = [nodo for nodo, _ in nodos_ordenados[:mid]]
        grupo_alta_var = [nodo for nodo, _ in nodos_ordenados[mid:]]

        if grupo_baja_var and grupo_alta_var:
            candidatas.append((grupo_baja_var, grupo_alta_var))

        return candidatas

    def s_generar_candidatas_por_correlacion(self, todos_los_nodos, tabla_de_costos):
        """Genera candidatas basadas en correlaciones entre tablas de costos."""
        candidatas = []
        # Calcular matriz de correlación entre variables
        variables_locales = []
        for tipo, idx in todos_los_nodos:
            if tipo == 1:
                if idx in self.mapa_global_a_local:
                    local_idx = self.mapa_global_a_local[idx]
                    if local_idx in tabla_de_costos:
                        variables_locales.append((tipo, idx, local_idx))

        if len(variables_locales) < 4:
            return candidatas

        # Convertir cada tabla de costos (diccionario) a matriz para comparación
        def tabla_dict_a_matriz(tabla_dict):
            # Obtener todos los pares de estados ordenados
            claves = sorted(tabla_dict.keys())
            return np.array([tabla_dict[k] for k in claves])

        correlaciones = {}
        for i, (t1, idx1, l1) in enumerate(variables_locales):
            for t2, idx2, l2 in variables_locales[i+1:]:
                tabla1 = tabla_dict_a_matriz(tabla_de_costos[l1])
                tabla2 = tabla_dict_a_matriz(tabla_de_costos[l2])
                # Correlación aproximada usando diferencia de normas
                norma_diff = np.linalg.norm(tabla1 - tabla2)
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

    def _evaluar_biparticion(self, grupo_a, grupo_b, distribuciones_marginales, tabla_de_costos):
        """Evaluación rápida de bipartición usando aproximaciones."""
        costo_total = 0.0

        # 1. Costo por diferencias en tablas (aproximación rápida)
        tablas_a = []
        tablas_b = []

        def tabla_dict_a_matriz(tabla_dict):
            claves = sorted(tabla_dict.keys())
            return np.array([tabla_dict[k] for k in claves])

        for tipo, idx in grupo_a:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in tabla_de_costos:
                    tablas_a.append(tabla_dict_a_matriz(tabla_de_costos[local_idx]))

        for tipo, idx in grupo_b:
            if idx in self.mapa_global_a_local:
                local_idx = self.mapa_global_a_local[idx]
                if local_idx in tabla_de_costos:
                    tablas_b.append(tabla_dict_a_matriz(tabla_de_costos[local_idx]))

        # Aproximación: usar promedios de tablas en lugar de comparaciones exhaustivas
        if tablas_a and tablas_b:
            promedio_a = np.mean(tablas_a, axis=0)
            promedio_b = np.mean(tablas_b, axis=0)
            costo_total += np.linalg.norm(promedio_a - promedio_b) * 0.1

        # 2. Penalización por desequilibrio (rápida)
        ratio = min(len(grupo_a), len(grupo_b)) / max(len(grupo_a), len(grupo_b))
        penalizacion = (1.0 - ratio) * 5.0

        # 3. Factor de coherencia temporal (ahora como penalización o neutro)
        tipos_a = {tipo for tipo, _ in grupo_a}
        tipos_b = {tipo for tipo, _ in grupo_b}

        factor_coherencia = 0.0
        if len(tipos_a) == 1 and len(tipos_b) == 1 and tipos_a != tipos_b:
            factor_coherencia = -min(penalizacion * 0.3, 1.0)
        elif len(tipos_a) > 1 and len(tipos_b) > 1:
            factor_coherencia = 0.5

        costo_final = costo_total + penalizacion + factor_coherencia

        return max(0.0, costo_final)

    def _obtener_biparticion_canonica(self, biparticion):
        """Canonicalización rápida sin transformaciones exhaustivas."""
        grupo_a, grupo_b = biparticion
        
        # Ordenar grupos por criterio simple pero consistente
        grupo_a_ordenado = sorted(grupo_a, key=lambda x: (x[0], x[1]))
        grupo_b_ordenado = sorted(grupo_b, key=lambda x: (x[0], x[1]))
        # Forma canónica: grupo lexicográficamente menor primero
        if len(grupo_a_ordenado) <= len(grupo_b_ordenado):
            return (tuple(grupo_a_ordenado), tuple(grupo_b_ordenado))
        else:
            return (tuple(grupo_b_ordenado), tuple(grupo_a_ordenado))

    def _valor_estado_variable(self, estado_binario, v_idx):
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