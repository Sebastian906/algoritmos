import time
import numpy as np
from collections import deque
import multiprocessing # Asumiendo que se re-incorporará la paralelización para calcular_tabla_costos_variable
from itertools import permutations, product

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
        self.logger = SafeLogger("GEOMETRIC")

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
        
        # indices_globales son todos los índices de las variables del subsistema, ordenados.
        # Es decir, los índices globales que tienen un n-cubo asociado en el subsistema.
        indices_globales_subsistema = sorted(list(set(nodos_alcance + nodos_mecanismo)))
        
        # mapa_global_a_local es crucial para acceder a tabla_costos_por_variable
        # donde la clave es el índice LOCAL (v_idx) de la variable dentro de self.sia_subsistema.ncubos
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales_subsistema)}
        
        # AGREGAR ESTA LÍNEA - Asignar como atributo de la instancia
        self.mapa_global_a_local = mapa_global_a_local
        
        # Obtener los estados binarios del subsistema. Estos son los estados del hipercubo.
        estados_bin = self.sia_subsistema.estados() if callable(self.sia_subsistema.estados) else self.sia_subsistema.estados
        
        # El número de dimensiones del hipercubo es la longitud de un estado binario
        num_dimensiones_hipercubo = len(estados_bin[0]) if len(estados_bin) > 0 else 0
    
        # 1. Cálculo de la Tabla de Costos con función t(i,j) reformulada
        self.logger.info("Calculando tabla de costos con función reformulada t(i,j)")
        variables_ordenadas_local = sorted(range(len(self.sia_subsistema.ncubos)))
        tabla_costos_por_variable = {}
        
        for v_idx_local in variables_ordenadas_local:
            tabla_costos_por_variable[v_idx_local] = self.calcular_tabla_costos_reformulada(estados_bin, v_idx_local)
    
        # 2. Cálculo de distribuciones marginales como proyecciones geométricas
        self.logger.info("Calculando distribuciones marginales como proyecciones geométricas")
        distribuciones_marginales = self.calcular_distribuciones_marginales(estados_bin)
        
        # 3. Aplicación de heurística con evaluación por discrepancia tensorial
        seed = 42
        heuristica = Heuristicas(seed, tabla_costos_por_variable, self.sia_subsistema, mapa_global_a_local)
        
        # Usar evaluación basada en discrepancia tensorial en lugar de clustering espectral
        (grupo_a_nodos, grupo_b_nodos), mejor_costo_heur = self.evaluar_biparticiones_discrepancia_tensorial(
            estados_bin, nodos_alcance, nodos_mecanismo, distribuciones_marginales, tabla_costos_por_variable, mapa_global_a_local
        )
        
        mejores = None
        mejor_costo = float("inf")
        
        if grupo_a_nodos and grupo_b_nodos:
            # 4. Normalización a forma canónica usando transformaciones del hipercubo
            biparticion_canonica = self.obtener_biparticion_canonica_geometrica(
                grupo_a_nodos, grupo_b_nodos, num_dimensiones_hipercubo
            )
            mejores = biparticion_canonica
            mejor_costo = mejor_costo_heur
            self.logger.info(f"Mejor solución geométrica: Grupo A: {mejores[0]} vs Grupo B: {mejores[1]}")
        else:
            self.logger.warning("No se encontró solución válida con la metodología geométrica reformulada.")
    
        # 5. Formateo y retorno
        if mejores:
            fmt_mip = fmt_biparte_q(list(mejores[0]), list(mejores[1]))
        else:
            fmt_mip = "No se encontró partición válida"
            mejor_costo = float("inf")
            
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_costo,
            distribucion_subsistema=distribuciones_marginales,
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def calcular_tabla_costos_reformulada(self, estados_bin, v_idx):
        """
        Implementación de la función de costo t(i,j) según la ecuación 3.1 del PDF:
        t_x(i,j) = γ · |X[i] - X[j]| + Σ_{k∈N(i,j)} {t(k,j)}
        donde γ = 2^(-d(i,j)) y d(i,j) es la distancia de Hamming.
        """
        n = len(estados_bin)
        T = np.zeros((n, n))

        # Calcular valores X[v] para cada estado
        val_estado = [self._valor_estado_variable(self._binario_a_entero(e), v_idx) for e in estados_bin]

        for i in range(n):
            for j in range(n):
                if i == j:
                    T[i][j] = 0.0
                    continue

                # Distancia de Hamming entre estados i y j
                d = self._distancia_hamming(estados_bin[i], estados_bin[j])
                
                # Factor de decrecimiento exponencial γ = 2^(-d)
                gamma = 2.0 ** (-d)
                
                # Contribución directa: γ · |X[i] - X[j]|
                T[i][j] = gamma * abs(val_estado[i] - val_estado[j])
                
                # Si no son vecinos inmediatos, agregar contribuciones recursivas
                if d > 1:
                    contribucion_recursiva = self._calcular_contribucion_recursiva_bfs(
                        i, j, estados_bin, val_estado, v_idx, gamma
                    )
                    T[i][j] += contribucion_recursiva

        return T

    def _calcular_contribucion_recursiva_bfs(self, origen, destino, estados_bin, val_estado, v_idx, gamma):
        """
        Implementación del Algoritmo 1 del PDF: BFS modificado para exploración recursiva
        del hipercubo con acumulación ponderada de costos.
        """
        contribucion_total = 0.0
        n = len(estados_bin)
        
        # Inicialización BFS
        Q = deque([origen])
        visited = set([origen])
        level = 0
        d_total = self._distancia_hamming(estados_bin[origen], estados_bin[destino])
        
        while level < d_total and Q:
            level += 1
            nextQ = deque()
            
            for u in Q:
                # Encontrar vecinos que nos acerquen al destino
                for v in range(n):
                    if (v not in visited and 
                        self._distancia_hamming(estados_bin[u], estados_bin[v]) == 1 and
                        self._distancia_hamming(estados_bin[v], estados_bin[destino]) < 
                        self._distancia_hamming(estados_bin[u], estados_bin[destino])):
                        
                        # Calcular contribución de este camino intermedio
                        d_intermedio = self._distancia_hamming(estados_bin[origen], estados_bin[v])
                        gamma_intermedio = 2.0 ** (-d_intermedio)
                        
                        # Acumulación del costo según línea 18 del Algoritmo 1
                        contribucion_intermedia = gamma_intermedio * abs(val_estado[origen] - val_estado[v])
                        contribucion_total += contribucion_intermedia
                        
                        visited.add(v)
                        nextQ.append(v)
            
            Q = nextQ
        
        return contribucion_total

    def calcular_distribuciones_marginales(self, estados_bin):
        """
        Calcula las distribuciones marginales como proyecciones geométricas
        del hipercubo n-dimensional según la sección 3.2.1 del PDF.
        """
        n_estados = len(estados_bin)
        n_variables = len(estados_bin[0]) if n_estados > 0 else 0
        
        distribuciones = {}
        
        # Para cada variable, calcular su distribución marginal
        for v_idx in range(n_variables):
            dist_marginal = np.zeros(2)  # Binaria: [P(X=0), P(X=1)]
            
            for estado in estados_bin:
                valor_variable = estado[v_idx]
                # Asumir distribución uniforme o usar pesos del subsistema
                peso_estado = 1.0 / n_estados  # Simplificación
                dist_marginal[valor_variable] += peso_estado
            
            distribuciones[v_idx] = dist_marginal
        
        # Calcular proyecciones conjuntas para pares de variables
        distribuciones['proyecciones_pares'] = {}
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                dist_conjunta = np.zeros((2, 2))  # P(Xi, Xj)
                
                for estado in estados_bin:
                    xi, xj = estado[i], estado[j]
                    peso_estado = 1.0 / n_estados
                    dist_conjunta[xi][xj] += peso_estado
                
                distribuciones['proyecciones_pares'][(i, j)] = dist_conjunta
        
        return distribuciones

    def evaluar_biparticiones_discrepancia_tensorial(self, estados_bin, nodos_alcance, nodos_mecanismo, 
                                               distribuciones_marginales, tabla_costos_por_variable, mapa_global_a_local=None):
        """
        Evaluación de biparticiones mediante discrepancia tensorial según sección 3.2 del PDF.
        En lugar de reconstruir el sistema completo, usa propiedades geométricas y marginales.
        """
        # Si no se pasa mapa_global_a_local, usar el atributo de la instancia
        if mapa_global_a_local is None:
            mapa_global_a_local = getattr(self, 'mapa_global_a_local', {})

        presentes = [(0, np.int8(idx)) for idx in nodos_mecanismo]
        futuros = [(1, np.int8(idx)) for idx in nodos_alcance]
        todos_los_nodos = futuros + presentes

        if len(todos_los_nodos) <= 1:
            return ([], []), float("inf")

        mejor_biparticion = None
        mejor_discrepancia = float("inf")

        # Estrategia de exploración inteligente basada en proyecciones marginales
        # En lugar de evaluar todas las biparticiones, usar heurística geométrica

        # 1. Identificar variables con mayor "independencia geométrica"
        independencias = self._calcular_independencias_geometricas(distribuciones_marginales, tabla_costos_por_variable)

        # 2. Generar biparticiones candidatas basadas en independencias
        biparticiones_candidatas = self._generar_biparticiones_candidatas(todos_los_nodos, independencias)

        # 3. Evaluar cada candidata usando discrepancia tensorial
        for grupo_a, grupo_b in biparticiones_candidatas:
            if len(grupo_a) == 0 or len(grupo_b) == 0:
                continue

            discrepancia = self._calcular_discrepancia_tensorial(
                grupo_a, grupo_b, distribuciones_marginales, tabla_costos_por_variable, mapa_global_a_local
            )

            if discrepancia < mejor_discrepancia:
                mejor_discrepancia = discrepancia
                mejor_biparticion = (grupo_a, grupo_b)

        if mejor_biparticion is None:
            return ([], []), float("inf")

        return mejor_biparticion, mejor_discrepancia

    def _calcular_independencias_geometricas(self, distribuciones_marginales, tabla_costos):
        """
        Calcula medidas de independencia geométrica entre variables basadas en
        las proyecciones marginales y la estructura del hipercubo.
        """
        independencias = {}
        n_variables = len([k for k in distribuciones_marginales.keys() if isinstance(k, int)])
        
        # Usar proyecciones conjuntas para medir independencia
        if 'proyecciones_pares' in distribuciones_marginales:
            for (i, j), dist_conjunta in distribuciones_marginales['proyecciones_pares'].items():
                # Calcular independencia como desviación del producto de marginales
                marginal_i = distribuciones_marginales[i]
                marginal_j = distribuciones_marginales[j]
                
                producto_marginales = np.outer(marginal_i, marginal_j)
                discrepancia = np.linalg.norm(dist_conjunta - producto_marginales, 'fro')
                
                independencias[(i, j)] = discrepancia
        
        return independencias

    def _generar_biparticiones_candidatas(self, todos_los_nodos, independencias):
        """
        Genera biparticiones candidatas inteligentemente basándose en las independencias geométricas.
        """
        candidatas = []
        n_nodos = len(todos_los_nodos)
        
        # Estrategia 1: Separar por tipo (presente/futuro) primero
        presentes = [nodo for nodo in todos_los_nodos if nodo[0] == 0]
        futuros = [nodo for nodo in todos_los_nodos if nodo[0] == 1]
        
        if len(presentes) > 0 and len(futuros) > 0:
            candidatas.append((presentes, futuros))
        
        # Estrategia 2: Usar independencias para agrupar variables similares
        if len(independencias) > 0:
            # Ordenar pares por independencia (menor = más dependientes)
            pares_ordenados = sorted(independencias.items(), key=lambda x: x[1])
            
            for i in range(min(3, len(pares_ordenados))):  # Limitar número de candidatas
                (var1, var2), _ = pares_ordenados[i]
                
                grupo_a = [nodo for nodo in todos_los_nodos if nodo[1] in [var1, var2]]
                grupo_b = [nodo for nodo in todos_los_nodos if nodo not in grupo_a]
                
                if len(grupo_a) > 0 and len(grupo_b) > 0:
                    candidatas.append((grupo_a, grupo_b))
        
        # Estrategia 3: Bipartición aleatoria controlada si no hay suficientes candidatas
        if len(candidatas) < 2:
            mid = n_nodos // 2
            candidatas.append((todos_los_nodos[:mid], todos_los_nodos[mid:]))
        
        return candidatas

    def _calcular_discrepancia_tensorial(self, grupo_a, grupo_b, distribuciones_marginales, tabla_costos, mapa_global_a_local):
        """
        Calcula la discrepancia tensorial de una bipartición según la metodología del PDF.
        Mide qué tan bien las proyecciones marginales de cada grupo preservan la información
        del sistema original sin necesidad de reconstrucción tensorial completa.
        """
        discrepancia_total = 0.0

        # 1. Discrepancia por pérdida de información en proyecciones
        for nodo_a in grupo_a:
            for nodo_b in grupo_b:
                tipo_a, idx_a = nodo_a
                tipo_b, idx_b = nodo_b

                # Si ambos índices están en el mapa local, usar tabla de costos
                if (idx_a in mapa_global_a_local and 
                    idx_b in mapa_global_a_local):

                    local_a = mapa_global_a_local[idx_a]
                    local_b = mapa_global_a_local[idx_b]

                    if local_a in tabla_costos and local_b in tabla_costos:
                        # Usar diferencia entre tablas de costos como medida de discrepancia
                        diff_tablas = np.linalg.norm(
                            tabla_costos[local_a] - tabla_costos[local_b], 'fro'
                        )
                        discrepancia_total += diff_tablas

        # 2. Penalización por desequilibrio en la bipartición
        ratio_grupos = min(len(grupo_a), len(grupo_b)) / max(len(grupo_a), len(grupo_b))
        penalizacion_desequilibrio = (1.0 - ratio_grupos) * 10.0  # Factor ajustable

        # 3. Bonificación por coherencia geométrica (variables del mismo tipo juntas)
        bonus_coherencia = 0.0
        tipos_a = set(nodo[0] for nodo in grupo_a)
        tipos_b = set(nodo[0] for nodo in grupo_b)

        if len(tipos_a) == 1 or len(tipos_b) == 1:  # Al menos un grupo es homogéneo
            bonus_coherencia = -2.0  # Reducir discrepancia

        return discrepancia_total + penalizacion_desequilibrio + bonus_coherencia

    def obtener_biparticion_canonica_geometrica(self, grupo_a, grupo_b, n_dimensiones):
        """
        Obtiene la forma canónica de la bipartición usando transformaciones geométricas
        del hipercubo (permutaciones y complementaciones) según metodología del PDF.
        """
        # Conversión a representación para canonicalización
        # Como trabajamos con nodos (tipo, idx) en lugar de estados binarios,
        # adaptamos la canonicalización al contexto de variables
        
        # Ordenar grupos por criterios geométricos consistentes
        grupo_a_ordenado = sorted(grupo_a, key=lambda x: (x[0], x[1]))
        grupo_b_ordenado = sorted(grupo_b, key=lambda x: (x[0], x[1]))
        
        # Asegurar forma canónica: el grupo "menor" lexicográficamente va first
        if grupo_a_ordenado < grupo_b_ordenado:
            return (grupo_a_ordenado, grupo_b_ordenado)
        else:
            return (grupo_b_ordenado, grupo_a_ordenado)

    def _valor_estado_variable(self, idx, v_idx):
        # ... (código existente) ...
        try:
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx]
        except (IndexError, AttributeError) as e:
            self.logger.error(f"Error al acceder al valor del estado para v_idx={v_idx}, idx={idx}: {e}")
            return 0.0

    def _binario_a_entero(self, binario):
        # ... (código existente) ...
        return int("".join(map(str, binario)), 2)

    def _distancia_hamming(self, v, u):
        # ... (código existente) ...
        return sum(b1 != b2 for b1, b2 in zip(v, u))

    def permutaciones_coordenadas(self, n):
        # ... (código existente) ...
        return list(permutations(range(n)))

    def complementaciones_coordenadas(self, n):
        # ... (código existente) ...
        return list(product([False, True], repeat=n))

    def aplicar_transformacion(self, estado, permutacion, complemento):
        # ... (código existente) ...
        estado_permutado = [estado[i] for i in permutacion]
        estado_transformado = [bit ^ int(comp) for bit, comp in zip(estado_permutado, complemento)]
        return estado_transformado

    def obtener_canonica(self, biparticion_frozenset_de_frozensets, n):
        # ... (código existente) ...
        min_bip_canonica = None
        
        grupo1_bin = list(biparticion_frozenset_de_frozensets)[0]
        grupo2_bin = list(biparticion_frozenset_de_frozensets)[1]
        
        for perm in self.permutaciones_coordenadas(n):
            for comp in self.complementaciones_coordenadas(n):
                grupo1_transformado = frozenset(
                    tuple(self.aplicar_transformacion(list(s), perm, comp)) for s in grupo1_bin
                )
                grupo2_transformado = frozenset(
                    tuple(self.aplicar_transformacion(list(s), perm, comp)) for s in grupo2_bin
                )
                
                bip_transformada = frozenset([grupo1_transformado, grupo2_transformado])
                
                if min_bip_canonica is None or bip_transformada < min_bip_canonica:
                    min_bip_canonica = bip_transformada
        
        self.logger.debug(f"Bipartición original: {biparticion_frozenset_de_frozensets}")
        self.logger.debug(f"Bipartición canónica: {min_bip_canonica}")
        return min_bip_canonica

    def reconstruir_tpm(self,):
        # ... (código existente) ...
        if not self.sia_subsistema or not self.sia_subsistema.ncubos:
            self.logger.error("Subsistema o n-cubos no inicializados para reconstruir TPM.")
            return None

        tensores = [ncubo.data for ncubo in self.sia_subsistema.ncubos]
        
        if not tensores:
            return np.array([[]])
            
        tpm = tensores[0]
        for t in tensores[1:]:
            try:
                tpm = np.tensordot(tpm, t, axes=0)
            except ValueError as e:
                self.logger.error(f"Error al realizar tensordot en reconstruir_tpm: {e}")
                return None

        tpm = tpm.reshape((2**self.sia_subsistema.n_variables, 2**self.sia_subsistema.n_variables))
        return tpm

    def mostrar_slice_tensor(self, v_idx, condicion_estado):
        # ... (código existente) ...
        prob_0, prob_1 = self._tensor_slice(v_idx, condicion_estado)
        self.logger.info(f"P(X_{v_idx}=0 | estado={condicion_estado}) = {prob_0:.4f}")
        self.logger.info(f"P(X_{v_idx}=1 | estado={condicion_estado}) = {prob_1:.4f}")
    
    def encontrar_ruta_minima(self, origen, destino, estados_bin):
        # ... (código existente) ...
        origen_tuple = tuple(origen)
        destino_tuple = tuple(destino)

        n = len(estados_bin[0])
        
        visitado = set()
        cola = deque([(origen_tuple, [origen_tuple])])
        
        while cola:
            actual_tuple, camino = cola.popleft()
            
            if actual_tuple == destino_tuple:
                return camino

            if actual_tuple in visitado:
                continue
            
            visitado.add(actual_tuple)

            actual_list = list(actual_tuple)
            for i in range(n):
                vecino_list = actual_list[:]
                vecino_list[i] ^= 1
                vecino_tuple = tuple(vecino_list)

                if vecino_tuple not in visitado:
                    cola.append((vecino_tuple, camino + [vecino_tuple]))
        
        return None