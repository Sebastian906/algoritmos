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
        
        # Obtener los estados binarios del subsistema. Estos son los estados del hipercubo.
        estados_bin = self.sia_subsistema.estados() if callable(self.sia_subsistema.estados) else self.sia_subsistema.estados
        
        # El número de dimensiones del hipercubo es la longitud de un estado binario
        num_dimensiones_hipercubo = len(estados_bin[0]) if len(estados_bin) > 0 else 0

        # --- Parte 1: Cálculo de la Tabla de Costos Secuencial ---
        # `variables_ordenadas` se refiere a los índices locales (v_idx) de las variables
        # dentro del subsistema (0 a len(self.sia_subsistema.ncubos)-1).
        variables_ordenadas_local = sorted(range(len(self.sia_subsistema.ncubos)))
        tabla_costos_por_variable = {}
        # Aquí se calcula T[i][j] para cada variable v_idx (local_idx)
        for v_idx_local in variables_ordenadas_local:
            tabla_costos_por_variable[v_idx_local] = self.calcular_tabla_costos_variable(estados_bin, v_idx_local)

        # --- Parte 2: Aplicación de Heurística ---
        seed = 42
        heuristica = Heuristicas(seed, tabla_costos_por_variable, self.sia_subsistema, mapa_global_a_local)
        
        # La función heurística ahora debe retornar la bipartición de los (tipo, idx)
        # y su costo asociado.
        # Pasa los índices globales, no los locales, para que la heurística construya los (tipo, idx).
        (grupo_a_nodos, grupo_b_nodos), mejor_costo_heur = heuristica.spectral_clustering_bipartition2(
            estados_bin, nodos_alcance, nodos_mecanismo, modo='aislado'
        )
        
        mejores = None
        mejor_costo = float("inf")
        
        # Ya que la heurística devuelve la bipartición de los nodos (tipo, idx),
        # no necesitamos la conversión compleja a estados_bin para la canónica aquí.
        # La forma canónica para la bipartición de nodos (variables) no está directamente
        # definida por `obtener_canonica` que opera sobre estados binarios.
        # Si se necesita una "forma canónica" para los grupos de (tipo, idx),
        # se necesitaría una función `obtener_canonica_nodos` diferente, que
        # opere sobre los identificadores (tipo, idx) y no sobre los estados binarios.
        # Por ahora, simplemente tomamos el resultado de la heurística.

        if grupo_a_nodos and grupo_b_nodos: # Asegurarse de que la heurística encontró una bipartición válida
            mejores = (grupo_a_nodos, grupo_b_nodos) # Ya están en el formato (tipo, idx)
            mejor_costo = mejor_costo_heur
            print(f"Mejor solución heurística: Grupo A: {mejores[0]} vs Grupo B: {mejores[1]}")
        else:
            print("No se encontró solución heurística o la bipartición fue trivial/inválida.")
            # mejores y mejor_costo ya están inicializados a None y inf

        # --- Parte 4: Formateo y Retorno de la Solución ---
        if mejores:
            # `fmt_biparte_q` espera listas de (tipo, idx) para formatear.
            fmt_mip = fmt_biparte_q(list(mejores[0]), list(mejores[1]))
        else:
            fmt_mip = "No se encontró partición válida"
            mejor_costo = float("inf") # Asegurarse de que el costo sea infinito si no hay solución.
            
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_costo,
            distribucion_subsistema=self.sia_dists_marginales, # Asumiendo que esto se calcula previamente
            distribucion_particion=None, # Esto podría calcularse si la partición fuera de estados
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    # --- El resto de las funciones de GeometricSIA (calcular_tabla_costos_variable,
    #     _valor_estado_variable, _binario_a_entero, _distancia_hamming,
    #     permutaciones_coordenadas, complementaciones_coordenadas,
    #     aplicar_transformacion, obtener_canonica, reconstruir_tpm,
    #     mostrar_slice_tensor, encontrar_ruta_minima)
    #     permanecen sin cambios en este archivo, ya que la lógica de la canónica
    #     para estados binarios y el cálculo de la tabla de costos es correcta para lo que hacen.
    #     Solo la llamada y el procesamiento del resultado de la heurística cambió.
    # ---

    def calcular_tabla_costos_variable(self, estados_bin, v_idx):
        # ... (código existente) ...
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
                            for v_inner in range(n): # Renombrado para evitar conflicto con la variable `v` del loop principal
                                if self._distancia_hamming(estados_bin[u], estados_bin[v_inner]) == 1 and \
                                self._distancia_hamming(estados_bin[v_inner], estados_bin[j]) < self._distancia_hamming(estados_bin[u], estados_bin[j]):
                                    if v_inner not in visited:
                                        # Línea 18: Acumulación del costo.
                                        # Nota: T[i][v_inner] puede no estar calculado aún.
                                        # El algoritmo 1 sugiere acumular, pero un BFS "clásico" calcula niveles.
                                        # Si el documento requiere una propagación de costos a través de T[i][v_inner]
                                        # en un orden específico, esto debe ser claro en el algoritmo 1.
                                        # Si T[i][v_inner] es el costo de (i a v_inner), entonces está bien.
                                        T[i][j] += gamma * T[i][v_inner]
                                        visited.add(v_inner)
                                        nextQ.append(v_inner)
                        Q = nextQ
                        level += 1

                # Aplicar factor gamma al total acumulado de la distancia entre i y j
                T[i][j] *= gamma

        return T

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