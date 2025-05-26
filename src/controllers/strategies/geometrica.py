import time
import numpy as np
from itertools import combinations

from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from itertools import chain, combinations
# ...existing code...

class GeometricSIA(SIA):
    """
    Estrategia geométrica para encontrar la bipartición óptima (MIP).
    Utiliza los n-cubos ya generados por el sistema y busca la partición causal
    que minimiza la pérdida (EMD) entre la distribución marginal original y la particionada.
    """

    def __init__(self, gestor):
        super().__init__(gestor)
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger("GEOMETRIC")

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """
        Busca la bipartición óptima usando la tabla de costos geométrica.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        nodos_mecanismo = list(self.sia_subsistema.dims_ncubos)
        nodos_alcance = list(self.sia_subsistema.indices_ncubos)

        # Mapeo de índices globales a locales
        indices_globales = list(self.sia_subsistema.indices_ncubos)
        mapa_global_a_local = {global_idx: local_idx for local_idx, global_idx in enumerate(indices_globales)}

        # Usa la función estados() para obtener todos los estados binarios posibles
        estados_bin = self.sia_subsistema.estados() if callable(self.sia_subsistema.estados) else self.sia_subsistema.estados
        num_estados = len(estados_bin)

        # Calcula la tabla de costos una sola vez
        tabla_costos = self.calcular_tabla_costos(estados_bin)

        mejores = None
        mejor_costo = float("inf")

        # Genera todas las biparticiones no triviales de nodos_alcance
        def todas_biparticiones(nodos):
            nodos = list(nodos)
            for r in range(1, len(nodos)):
                for grupoA in combinations(nodos, r):
                    grupoB = [n for n in nodos if n not in grupoA]
                    yield grupoA, grupoB

        for grupoA, grupoB in todas_biparticiones(nodos_alcance):
            # Puedes ajustar aquí cómo defines los grupos para la función de costo
            # Por ejemplo, podrías definir que grupoA es el conjunto aislado y grupoB el resto
            indicesA = [mapa_global_a_local[n] for n in grupoA]
            indicesB = [mapa_global_a_local[n] for n in grupoB]

            costo_total = 0.0
            for i in range(num_estados):
                for j in range(num_estados):
                    # Si los estados difieren solo en los nodos de grupoA, cuenta el costo
                    if any(estados_bin[i][idx] != estados_bin[j][idx] for idx in indicesA):
                        int_v = self._binario_a_entero(estados_bin[i])
                        int_u = self._binario_a_entero(estados_bin[j])
                        costo_total += tabla_costos[(int_v, int_u)]

            if costo_total < mejor_costo:
                mejor_costo = costo_total
                mejores = (
                    [(1, n) for n in grupoA],
                    [(1, n) for n in grupoB] + [(0, n) for n in nodos_mecanismo]
                )

        if mejores is not None:
            fmt_mip = fmt_biparte_q(
                mejores[0], mejores[1]
            )
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

    def calcular_tabla_costos(self, estados_bin):
        """
        Construye la tabla de costos T entre todos los pares de estados binarios v y u del sistema.
        Aplica: t(i,j) = γ * |X[i] - X[j]|, con γ = 2^(-dH(i,j))
        """
        T = {}
        num_estados = len(estados_bin)

        for i in range(num_estados):
            for j in range(num_estados):
                estado_v = estados_bin[i]
                estado_u = estados_bin[j]
                int_v = self._binario_a_entero(estado_v)
                int_u = self._binario_a_entero(estado_u)

                d = self._distancia_hamming(estado_v, estado_u)
                gamma = 2.0 ** -d if d > 0 else 1.0  # γ = 2^-dH

                # Aquí usas los n-cubos para obtener el valor del estado
                val_v = self._valor_estado(int_v)
                val_u = self._valor_estado(int_u)

                T[(int_v, int_u)] = gamma * abs(val_v - val_u)
        return T

    def _valor_estado(self, idx):
        """
        Obtiene el valor asociado a un estado dado su índice entero.
        Usa los cubos creados por ncubos.
        """
        valor = 0.0
        for ncubo in self.sia_subsistema.ncubos:
            try:
                valor += ncubo.data.flat[idx]
            except IndexError:
                continue
        return valor

    def _binario_a_entero(self, binario):
        """
        Convierte una lista o array binaria a entero.
        """
        return int("".join(str(b) for b in binario), 2)

    def _distancia_hamming(self, v, u):
        """
        Calcula la distancia de Hamming entre dos listas/arrays binarios.
        """
        return sum(b1 != b2 for b1, b2 in zip(v, u))