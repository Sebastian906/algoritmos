import time 
import numpy as np
from itertools import combinations

# Imports del framework del proyecto
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_geometrico
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL, INFTY_POS
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile


class GeometricSIA(SIA):
    """
    Estrategia geométrica para encontrar la bipartición óptima (MIP).
    Utiliza una representación del sistema como hipercubo y evalúa transiciones 
    entre estados con una función de costo basada en distancia de Hamming.
    """
    
    def __init__(self, gestor):
        super().__init__(gestor)
        # Inicia perfilador con nombre de red y número de nodos
        profiler_manager.start_session(f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}")
        self.logger = SafeLogger("GEOMETRIC")

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(self, condicion, alcance, mecanismo):
        """
        Método principal que ejecuta la estrategia geométrica:
        1. Prepara el subsistema
        2. Crea los hipercubos
        3. Calcula la tabla de costos
        4. Identifica candidatos a MIP
        5. Evalúa y retorna la solución
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # Preparación de índices de cubos (variables del mecanismo y del alcance)
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.indices_alcance = self.sia_subsistema.indices_ncubos

        # Inicializa estructuras temporales para tiempos (opcional/diagnóstico)
        self.tiempos = (
            np.zeros(self.sia_subsistema.dims_ncubos.size, dtype=np.int8),
            np.zeros(self.sia_subsistema.indices_ncubos.size, dtype=np.int8),
        )

        # Construye cubos con combinaciones de tamaño 10
        self._crear_ncubos(tamaño_cubo=2)
        

        # Calcula la tabla de costos entre pares de estados
        tabla_costos = self._calcular_tabla_costos()
        #muestra de la tabla de costos (opcional/diagnóstico)

        # Identifica candidatos con costo mínimo (idealmente 0)
        candidatos = self._identificar_candidatos(tabla_costos)

        # Evalúa y selecciona la mejor bipartición candidata
        mip = self._evaluar_candidatos(candidatos, tabla_costos)
        

        # Formatea la solución para salida
        nodos_totales = len(self.indices_mecanismo) + len(self.indices_alcance)
        nodos_complementarios = list(set(range(nodos_totales)) - set(mip[0]))
        fmt_mip = fmt_biparte_geometrico(mip[0], nodos_complementarios)

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=mip[0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=None,  # Opcional: puede incluirse si es útil
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _crear_ncubos(self, tamaño_cubo):
        """
        Genera todos los hipercubos posibles de tamaño `tamaño_cubo`
        a partir de las combinaciones de los índices del mecanismo.
        """
        self.ncubos = [list(c) for c in combinations(self.indices_mecanismo, tamaño_cubo)]
        print(f"Generando {len(self.ncubos)} cubos de tamaño {tamaño_cubo}...")

    def _calcular_tabla_costos(self):
        """
        Calcula la tabla de costos T para cada par de estados posibles (v, u)
        en cada cubo generado. Se utiliza:
        - La distancia de Hamming entre los estados
        - Un factor de decaimiento exponencial: gamma = 2^-d
        - Diferencias de probabilidad condicional obtenidas de los tensores
        """
        T = {}  # Diccionario con claves: (i_cubo, estado_v, estado_u)

        for i_cubo, indices_cubo in enumerate(self.ncubos):
            n = len(indices_cubo)

            for v in range(2**n):      # Estado v del mecanismo
                for u in range(2**n):  # Estado u del alcance
                    d = bin(v ^ u).count("1")         # Distancia de Hamming entre v y u
                    gamma = 2 ** -d                    # Factor de decrecimiento

                    # Valor del estado v y u en los tensores correspondientes
                    Xv = self._valor_estado(v)
                    Xu = self._valor_estado(u)

                    # Costo de transición almacenado
                    T[(i_cubo, v, u)] = gamma * abs(Xv - Xu)

        return T

    def _valor_estado(self, estado):
        """
        Obtiene el valor asociado a un estado binario completo (según la TPM).
        """
        return self.sia_subsistema.valor_estado(estado)

    def _identificar_candidatos(self, tabla_costos):
        """
        Filtra los pares de estados con costo cero como candidatos óptimos.
        """
        return [k for k, v in tabla_costos.items() if v == 0]

    def _evaluar_candidatos(self, candidatos, tabla_costos):
        """
        Evalúa los candidatos y selecciona el que tenga menor costo (idealmente 0).
        """
        if not candidatos:
            return ((), INFTY_POS)  # No hay candidatos, retorna infinito
        mejor = min(candidatos, key=lambda c: tabla_costos[c])
        return (mejor, tabla_costos[mejor])
    def _valor_estado_cubo(self, estado, indices_cubo, tipo='mecanismo'):
        """
        Extrae el valor asociado a un estado binario dentro de un cubo.
        Internamente usa self.sia_subsistema.valor_estado(estado), que ya maneja la conversión.
        """
        return self._valor_estado(estado)
