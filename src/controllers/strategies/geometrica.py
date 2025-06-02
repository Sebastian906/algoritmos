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
from src.controllers.strategies.heurisiticas import Heuristicas

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
        tabla_costos = {v: self.calcular_tabla_costos_variable(estados_bin, v) for v in range(len(self.sia_subsistema.ncubos))}
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
            for v, tabla in tabla_costos.items():
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
        heuristica = Heuristicas()
        mejor_solucion, mejor_costo = heuristica.simulated_annealing_bipartition(estados_bin, tabla_costos, nodos_alcance)
        print("resultado normal")
        print(mejores[0], mejores[1])
        print("resultado heuristica")
        print(mejor_solucion[0], mejor_solucion[1])
        fmt_mip = fmt_biparte_q(mejor_solucion[0], mejor_solucion[1]) if mejor_solucion else "No se encontró partición válida"

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mejor_costo,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=None,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

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