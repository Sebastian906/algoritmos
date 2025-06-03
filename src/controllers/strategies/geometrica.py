import time
import numpy as np
from itertools import combinations
import random
import math
from collections import deque, defaultdict
import multiprocessing
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.controllers.strategies.heurisiticas import Heuristicas

def calcular_tabla_costos(estados_bin, val_estado):
    """
    Cálculo de la Tabla de Costos T mediante BFS modificado (versión global, sin self)
    """
    n = len(estados_bin)
    T = np.zeros((n, n))
    def distancia_hamming(v, u):
        return sum(b1 != b2 for b1, b2 in zip(v, u))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = distancia_hamming(estados_bin[i], estados_bin[j])
            gamma = 2.0 ** -d
            T[i][j] = abs(val_estado[i] - val_estado[j])
            if d > 1:
                Q = deque([i])
                visited = set([i])
                level = 0
                while level < d and Q:
                    nextQ = deque()
                    for u in Q:
                        for v in range(n):
                            if distancia_hamming(estados_bin[u], estados_bin[v]) == 1 and \
                               distancia_hamming(estados_bin[v], estados_bin[j]) < distancia_hamming(estados_bin[u], estados_bin[j]):
                                if v not in visited:
                                    T[i][j] += gamma * T[i][v]
                                    visited.add(v)
                                    nextQ.append(v)
                    Q = nextQ
                    level += 1
            # Aplicar factor gamma al total acumulado
            T[i][j] *= gamma
    return T

def calcular_tabla_costos_worker(args):
    estados_bin, v, vals = args
    return (v, calcular_tabla_costos(estados_bin, vals))

class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        random.seed(42)
        np.random.seed(42)
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger("GEOMETRIC")

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """Estrategia original con búsqueda exhaustiva"""
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        nodos_mecanismo = sorted(list(self.sia_subsistema.dims_ncubos))
        nodos_alcance = sorted(list(self.sia_subsistema.indices_ncubos))
        indices_globales = sorted(list(self.sia_subsistema.indices_ncubos))
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}
        estados_bin = self.sia_subsistema.estados() if callable(self.sia_subsistema.estados) else self.sia_subsistema.estados
        # Paralelización de la creación de la tabla de costos
        variables_ordenadas = sorted(range(len(self.sia_subsistema.ncubos)))
        args_list = []
        for v in variables_ordenadas:
            val_estado = [self._valor_estado_variable(self._binario_a_entero(e), v) for e in estados_bin]
            args_list.append((estados_bin, v, val_estado))
        tabla_costos = {}
        with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
            resultados = pool.map(calcular_tabla_costos_worker, args_list)
        tabla_costos = dict(resultados)
        mejores = None
        mejor_costo = float("inf")
        heuristica = Heuristicas(seed=42)
        heuristica.set_sia_context(self.sia_subsistema, mapa_global_a_local)
        mejor_solucion_heur, mejor_costo_heur = heuristica.simulated_annealing_bipartition(
            estados_bin, tabla_costos, nodos_alcance, use_corrected_evaluation=True
        )
        if mejor_solucion_heur:
            # Convertir la solución heurística al formato esperado (tiempo, nodo)
            solucion_formateada = (
                [(1, n) for n in mejor_solucion_heur[0]],  # Grupo A en tiempo futuro
                [(1, n) for n in mejor_solucion_heur[1]] + [(0, n) for n in nodos_mecanismo]  # Grupo B + mecanismo
            )
            print(solucion_formateada[0], solucion_formateada[1])
            # Usar la mejor solución entre exhaustiva y heurística
            if mejor_costo_heur < mejor_costo:
                mejores = solucion_formateada
                print(f"Heurística encontró mejor solución: costo {mejor_costo_heur} vs {mejor_costo}")
                mejor_costo = mejor_costo_heur
        else:
            print("No se encontró solución heurística")

        # Formatear la mejor solución encontrada
        if mejores:
            fmt_mip = fmt_biparte_q(mejores[0], mejores[1])
        else:
            fmt_mip = "No se encontró partición válida"
        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_costo,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
    def _valor_estado_variable(self, idx, v_idx):
        try:
            return self.sia_subsistema.ncubos[v_idx].data.flat[idx]
        except (IndexError, AttributeError):
            return 0.0
    def _binario_a_entero(self, binario):
        return int("".join(str(b) for b in binario), 2)