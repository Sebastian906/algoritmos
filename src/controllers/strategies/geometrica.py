import time
import numpy as np
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparte_geometrico
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL, INFTY_POS, EFECTO, ACTUAL
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile

class GeometricSIA(SIA):
    def __init__(self, gestor):
        super().__init__(gestor)
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger("GEOMETRIC")

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        self.tiempos = (
            np.zeros(self.sia_subsistema.dims_ncubos.size, dtype=np.int8),
            np.zeros(self.sia_subsistema.indices_ncubos.size, dtype=np.int8),
        )

        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.indices_alcance = self.sia_subsistema.indices_ncubos

        tabla_costos = self._calcular_tabla_costos()
        candidatos = self._identificar_candidatos(tabla_costos)
        mip = self._evaluar_candidatos(candidatos, tabla_costos)

        fmt_mip = fmt_biparte_geometrico(mip, list(set(range(len(self.indices_mecanismo) + len(self.indices_alcance))) - set(mip)))

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mip[1],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=None,  # Se puede agregar si es relevante
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _calcular_tabla_costos(self):
        T = {}
        for v in range(2**(len(self.indices_mecanismo))):
            for u in range(2**(len(self.indices_alcance))):
                d = bin(v ^ u).count("1")
                gamma = 2 ** -d
                Xv = self._valor_estado(v)
                Xu = self._valor_estado(u)
                T[(v, u)] = gamma * abs(Xv - Xu)  # simplificación
        return T

    def _valor_estado(self, estado):
        return self.sia_subsistema.valor_estado(estado)

    def _identificar_candidatos(self, tabla_costos):
        return [k for k, v in tabla_costos.items() if v == 0]

    def _evaluar_candidatos(self, candidatos, tabla_costos):
        if not candidatos:
            return ((), INFTY_POS)
        mejor = min(candidatos, key=lambda c: tabla_costos[c])
        return (mejor, tabla_costos[mejor])
