import time
import numpy as np
import pandas as pd
from pyphi import Network, Subsystem
from pyphi.labels import NodeLabels
from pyphi.models.cuts import Bipartition, Part

from src.controllers.manager import Manager
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.models.enums.distance import MetricDistance
from src.models.base.application import aplicacion

from src.constants.base import NET_LABEL, TYPE_TAG, STR_ONE
from src.constants.models import DUMMY_ARR, DUMMY_PARTITION, PYPHI_LABEL, PYPHI_ANALYSIS_TAG

from src.controllers.methods.memory_reduction import MemoryReductionStrategy
from src.controllers.methods.block_processing import BlockProcessingStrategy
from src.controllers.methods.parallelization import ParallelizationStrategy
from src.controllers.methods.compact_representation import CompactRepresentationStrategy

class Phi(SIA):
    def __init__(self, config: Manager) -> None:
        super().__init__(config)
        # Inicia la sesión de perfilado con una etiqueta específica
        profiler_manager.start_session(f"{NET_LABEL}{len(config.estado_inicial)}{config.pagina}")
        self.logger = SafeLogger(PYPHI_ANALYSIS_TAG)
        
        self.estrategias = config.estrategias  # Lista de estrategias disponibles

    @profile(context={TYPE_TAG: PYPHI_ANALYSIS_TAG})
    def aplicar_estrategia(self, condiciones: str, alcance: str, mecanismo: str, estrategia_muestreo=None, estrategia=None):
        self.sia_tiempo_inicio = time.time()
        
        # Prepara el subsistema con base en las condiciones, alcance y mecanismo
        alcance, mecanismo, subsistema = self.preparar_subsistema(condiciones, alcance, mecanismo)
        estados_totales = np.array([alcance, mecanismo], dtype=object)

        resultado_final = None
        
        # Aplica la estrategia seleccionada
        if isinstance(estrategia, MemoryReductionStrategy) and estrategia_muestreo:
            resultado_final = estrategia.muestrear_estados(estados_totales, porcentaje=estrategia_muestreo)
        elif isinstance(estrategia, BlockProcessingStrategy):
            resultado_final = estrategia.aplicar_estrategia(self.sia_cargar_tpm(),  self.calcular_mip)
        elif isinstance(estrategia, ParallelizationStrategy):
            resultado_final = estrategia.aplicar_estrategia(estados_totales, self.calcular_mip)
        elif isinstance(estrategia, CompactRepresentationStrategy):
            resultado_final = estrategia.comprimir_estados(estados_totales)
        else:
            # Si no hay estrategia específica, se calcula el MIP de cada estado individualmente
            resultado_final = [self.calcular_mip(estado) for estado in estados_totales]
        
        # Selecciona el estado con el valor máximo de phi
        mip = max(resultado_final, key=lambda x: x.phi, default=None)
        
        if not mip:
            return None

        # Retorna la solución con los resultados obtenidos
        return Solution(
            estrategia=PYPHI_LABEL,
            perdida=mip.phi,
            distribucion_subsistema=DUMMY_ARR,
            distribucion_particion=DUMMY_PARTITION,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=DUMMY_PARTITION,
        )

    def calcular_mip(self, estado):
        # Crea un subsistema con el estado actual y los nodos candidatos
        subsistema = Subsystem(self.sia_red, state=estado, nodes=self.sia_candidato)

        
        # Calcula el efecto o causa MIP dependiendo de la métrica de distancia seleccionada
        return (
            subsistema.effect_mip(self.sia_mecanismo, self.sia_alcance)
            if aplicacion.distancia_metrica == MetricDistance.EMD_EFECTO.value
            else subsistema.cause_mip(self.sia_mecanismo, self.sia_alcance)
        )

    def preparar_subsistema(self, condiciones: str, futuros: str, presentes: str):
        # Convierte el estado inicial en una tupla de enteros
        estado_inicial = tuple(int(s) for s in self.sia_gestor.estado_inicial)
        print(estado_inicial)
        longitud = len(estado_inicial)
        
        # Genera etiquetas y asignaciones de nodos
        indices = tuple(range(longitud))
        etiquetas = tuple("ABCDEFGHIJKLMNOPQRST"[:longitud])
        completo = NodeLabels(etiquetas, indices)
        
        # Crea la red de PyPhi con el TPM cargado
        self.sia_red = Network(tpm=self.sia_cargar_tpm(), node_labels=completo)
        
        
        # Determina los nodos candidatos basados en las condiciones
        self.sia_candidato = tuple(completo[i] for i, bit in enumerate(condiciones) if bit == STR_ONE)
        
        # Crea un subsistema con la configuración generada
        subsistema = Subsystem(network=self.sia_red, state=estado_inicial, nodes=self.sia_candidato)

        # Calcula los alcances y mecanismos con base en futuros y presentes
        self.sia_alcance = tuple(ind for ind, (bit, cond) in enumerate(zip(futuros, condiciones)) if (bit == STR_ONE) and (cond == STR_ONE))
        self.sia_mecanismo = tuple(ind for ind, (bit, cond) in enumerate(zip(presentes, condiciones)) if (bit == STR_ONE) and (cond == STR_ONE))
        
        return self.sia_alcance, self.sia_mecanismo, subsistema