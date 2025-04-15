import time
from typing import Any, Union, Tuple, List, Set
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto, ABECEDARY
from src.middlewares.profile import profiler_manager, profile
from src.funcs.format import fmt_biparte_q
from src.controllers.manager import Manager
from src.models.base.sia import SIA

from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_NEG,
    INFTY_POS,
    LAST_IDX,
    EFECTO,
    ACTUAL,
)


class NodesQ(SIA):
    """
    Clase QNodes optimizada para el análisis de redes mediante el algoritmo Q.

    Esta implementación incluye optimizaciones para reducir tiempos de ejecución:
    - Memoización mejorada para cálculos de EMD
    - Estructuras de datos optimizadas para accesos rápidos
    - Pre-cálculo de componentes comunes
    - Eliminación de recálculos en la función submodular
    - Vectorización de operaciones donde es posible
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        
        # Memoización mejorada usando diccionarios específicos
        self.memoria_individual = {}  # Para EMDs y distribuciones individuales
        self.memoria_combinaciones = {}  # Para EMDs de combinaciones
        self.memoria_particiones = {}  # Para resultados de particiones

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        self.logger = SafeLogger(QNODES_STRAREGY_TAG)

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        """
        Aplica la estrategia de análisis Q-Nodes al subsistema especificado.
        
        Args:
            condicion (str): Condición inicial del sistema
            alcance (str): Alcance del análisis
            mecanismo (str): Mecanismo a analizar
            
        Returns:
            Solution: Solución que contiene la partición óptima y métricas asociadas
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size

        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        
        # Precálculo de EMDs individuales para cada vértice
        self._precalcular_emds_individuales(vertices)
        
        # Ejecutar algoritmo
        mip = self.algorithm(vertices)
        
        # Asegurar que mip sea una lista de tuplas para el formateo correcto
        mip_list = self._asegurar_formato_lista(mip)
        
        # Obtener el complemento y formatear la solución
        complement = self.nodes_complement(mip_list)
        fmt_mip = fmt_biparte_q(mip_list, complement)
        print(f"{fmt_mip}")
        return Solution(
            estrategia=QNODES_LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
    
    def _precalcular_emds_individuales(self, vertices: List[Tuple[int, int]]) -> None:
        """
        Precalcula los EMDs individuales para cada vértice para evitar recálculos.
        
        Args:
            vertices: Lista de vértices del sistema.
        """
        for vertex in vertices:
            if vertex not in self.memoria_individual:
                temporal = [[], []]
                tiempo, indice = vertex
                temporal[tiempo].append(indice)
                
                particion_delta = self.sia_subsistema.bipartir(
                    np.array(temporal[EFECTO], dtype=np.int8),
                    np.array(temporal[ACTUAL], dtype=np.int8),
                )
                vector_marginal = particion_delta.distribucion_marginal()
                emd = emd_efecto(vector_marginal, self.sia_dists_marginales)
                
                self.memoria_individual[vertex] = (emd, vector_marginal)
    
    def _asegurar_formato_lista(self, item) -> List[Tuple[int, int]]:
        """
        Convierte recursivamente cualquier estructura de nodos a una lista plana de tuplas.
        
        Args:
            item: Objeto a convertir (tupla, lista de tuplas, etc.)
            
        Returns:
            Lista plana de tuplas (tiempo, índice)
        """
        result = []
        
        if isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
            # Es una tupla simple (tiempo, índice)
            return [item]
        elif isinstance(item, (list, tuple)):
            # Es una estructura anidada
            for subitem in item:
                result.extend(self._asegurar_formato_lista(subitem))
        
        return result

    def algorithm(self, vertices: List[Tuple[int, int]]):
        """
        Implementa el algoritmo Q optimizado para encontrar la partición óptima.
        
        Args:
            vertices: Lista de vértices donde cada uno es una tupla (tiempo, índice)
            
        Returns:
            La clave de la partición óptima en memoria_particiones
        """
        omegas_origen = [vertices[0]]
        deltas_origen = vertices[1:]
        vertices_fase = vertices.copy()

        total = len(vertices_fase) - 2
        for i in range(total):
            self.logger.debug(f"Fase: {i+1}/{total}")
            
            # Reiniciar para cada fase
            omegas_ciclo = [vertices_fase[0]]
            deltas_ciclo = vertices_fase[1:]

            for j in range(len(deltas_ciclo) - 1):
                emd_local = INFTY_POS
                indice_mip = -1

                # Evaluar cada delta candidato
                for k in range(len(deltas_ciclo)):
                    delta_k = deltas_ciclo[k]
                    
                    # Calcular EMD de la combinación
                    emd_union, emd_delta, dist_marginal = self.funcion_submodular(
                        delta_k, omegas_ciclo
                    )
                    emd_iteracion = emd_union - emd_delta

                    # Actualizar si encontramos mejor opción
                    if emd_iteracion < emd_local:
                        emd_local = emd_iteracion
                        indice_mip = k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal

                # Actualizar omegas y deltas
                if indice_mip >= 0:
                    omegas_ciclo.append(deltas_ciclo[indice_mip])
                    deltas_ciclo.pop(indice_mip)

            # Almacenar la partición candidata como tupla inmutable
            if deltas_ciclo:
                delta_candidato = deltas_ciclo[LAST_IDX]
                particion_key = self._convertir_a_clave_inmutable(delta_candidato)
                
                self.memoria_particiones[particion_key] = (
                    emd_particion_candidata, 
                    dist_particion_candidata
                )
                
                # Formar el par candidato para la siguiente fase
                ultimo_omega = omegas_ciclo[LAST_IDX]
                
                # Crear un nuevo par combinando omega y delta
                if isinstance(ultimo_omega, tuple) and isinstance(delta_candidato, tuple):
                    par_candidato = [ultimo_omega, delta_candidato]
                elif isinstance(ultimo_omega, list):
                    if isinstance(delta_candidato, tuple):
                        par_candidato = ultimo_omega + [delta_candidato]
                    else:
                        par_candidato = ultimo_omega + delta_candidato
                else:
                    if isinstance(delta_candidato, list):
                        par_candidato = [ultimo_omega] + delta_candidato
                    else:
                        par_candidato = [ultimo_omega, delta_candidato]
                
                # Actualizar para la siguiente fase
                omegas_ciclo.pop()
                omegas_ciclo.append(par_candidato)
                vertices_fase = omegas_ciclo

        # Encontrar la mejor partición
        return min(
            self.memoria_particiones, 
            key=lambda k: self.memoria_particiones[k][0]
        )
    
    def _convertir_a_clave_inmutable(self, obj):
        """
        Convierte un objeto a una clave inmutable para usar como clave de diccionario.
        
        Args:
            obj: Objeto a convertir
            
        Returns:
            Versión inmutable del objeto (usualmente una tupla)
        """
        if isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, int) for x in obj):
            # Ya es una tupla simple (tiempo, índice)
            return obj
        elif isinstance(obj, (list, tuple)):
            # Es una estructura compleja, convertir recursivamente
            elementos = []
            for item in obj:
                elementos.append(self._convertir_a_clave_inmutable(item))
            return tuple(elementos)
        else:
            # Fallback para otros tipos
            return obj

    def funcion_submodular(
        self, deltas: Union[Tuple[int, int], List], omegas: List
    ) -> Tuple[float, float, np.ndarray]:
        """
        Evalúa el impacto de combinar nodos delta con el conjunto omega.
        
        Args:
            deltas: Un nodo individual (tupla) o grupo de nodos
            omegas: Lista de nodos ya agrupados
            
        Returns:
            (EMD combinación, EMD delta individual, Distribución marginal delta)
        """
        # Optimización para nodos individuales
        if isinstance(deltas, tuple) and len(deltas) == 2 and all(isinstance(x, int) for x in deltas):
            if deltas in self.memoria_individual:
                emd_delta, vector_delta_marginal = self.memoria_individual[deltas]
            else:
                # Calcular para un delta individual
                temporal = [[], []]
                d_tiempo, d_indice = deltas
                temporal[d_tiempo].append(d_indice)
                
                # Bipartir y calcular EMD
                particion_delta = self.sia_subsistema.bipartir(
                    np.array(temporal[EFECTO], dtype=np.int8),
                    np.array(temporal[ACTUAL], dtype=np.int8),
                )
                vector_delta_marginal = particion_delta.distribucion_marginal()
                emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
                
                # Guardar en memoria
                self.memoria_individual[deltas] = (emd_delta, vector_delta_marginal)
        else:
            # Caso para múltiples deltas
            temporal = [[], []]
            self._procesar_nodos(deltas, temporal)
            
            # Crear clave única para esta configuración
            key_delta = self._convertir_a_clave_inmutable(deltas)
            
            if key_delta in self.memoria_individual:
                emd_delta, vector_delta_marginal = self.memoria_individual[key_delta]
            else:
                # Bipartir y calcular EMD
                particion_delta = self.sia_subsistema.bipartir(
                    np.array(temporal[EFECTO], dtype=np.int8),
                    np.array(temporal[ACTUAL], dtype=np.int8),
                )
                vector_delta_marginal = particion_delta.distribucion_marginal()
                emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
                
                # Guardar en memoria
                self.memoria_individual[key_delta] = (emd_delta, vector_delta_marginal)
        
        # Si no hay omegas, solo devolver información de delta
        if not omegas:
            return emd_delta, emd_delta, vector_delta_marginal
        
        # Preparar estructura para la combinación
        temporal_union = [[], []]
        
        # Procesar deltas
        self._procesar_nodos(deltas, temporal_union)
        
        # Procesar omegas
        self._procesar_nodos(omegas, temporal_union)
        
        # Crear clave única para esta combinación
        omega_key = self._convertir_a_clave_inmutable(omegas)
        delta_key = self._convertir_a_clave_inmutable(deltas)
        combo_key = (delta_key, omega_key)
        
        # Verificar si ya calculamos esta combinación
        if combo_key in self.memoria_combinaciones:
            emd_union = self.memoria_combinaciones[combo_key]
        else:
            # Bipartir y calcular EMD para la unión
            particion_union = self.sia_subsistema.bipartir(
                np.array(temporal_union[EFECTO], dtype=np.int8),
                np.array(temporal_union[ACTUAL], dtype=np.int8),
            )
            vector_union = particion_union.distribucion_marginal()
            emd_union = emd_efecto(vector_union, self.sia_dists_marginales)
            
            # Guardar en memoria
            self.memoria_combinaciones[combo_key] = emd_union
        
        return emd_union, emd_delta, vector_delta_marginal
    
    def _procesar_nodos(self, nodos, temporal):
        """
        Procesa nodos (simples o complejos) y los añade a la estructura temporal.
        
        Args:
            nodos: Nodo individual o estructura de nodos
            temporal: Lista de dos listas para almacenar índices [actual, efecto]
        """
        if isinstance(nodos, tuple) and len(nodos) == 2 and all(isinstance(x, int) for x in nodos):
            # Es una tupla (tiempo, índice)
            tiempo, indice = nodos
            temporal[tiempo].append(indice)
        elif isinstance(nodos, (list, tuple)):
            # Es una estructura anidada
            for item in nodos:
                if isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
                    # Es una tupla (tiempo, índice)
                    tiempo, indice = item
                    temporal[tiempo].append(indice)
                elif isinstance(item, (list, tuple)):
                    # Procesar recursivamente
                    self._procesar_nodos(item, temporal)

    def nodes_complement(self, nodes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Obtiene el complemento de un conjunto de nodos respecto a todos los vértices.
        
        Args:
            nodes: Lista de nodos
            
        Returns:
            Lista de nodos complementarios
        """
        return list(self.vertices - set(nodes))