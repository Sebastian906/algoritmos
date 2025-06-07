import time
import numpy as np
from collections import deque
import multiprocessing
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.funcs.format import fmt_biparte_q
from src.constants.models import GEOMETRIC_LABEL, GEOMETRIC_ANALYSIS_TAG
from src.constants.base import TYPE_TAG, NET_LABEL
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profiler_manager, profile
from src.controllers.strategies.heurisiticas import Heuristicas
from itertools import permutations, product
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
            resultados = pool.map(calcular_tabla_costos_worker, args_list, chunksize=1)
        tabla_costos = dict(resultados)
        
        mejores = None
        mejor_costo = float("inf")
        seed=42
        heuristica = Heuristicas(seed, tabla_costos,self.sia_subsistema, mapa_global_a_local)
        
        mejor_solucion_heur, mejor_costo_heur = heuristica.spectral_clustering_bipartition(
            estados_bin, nodos_alcance, nodos_mecanismo
        )
        
        if mejor_solucion_heur:
            # Convertir la solución heurística al formato esperado (tiempo, nodo)
            solucion_formateada = (
                [(n) for n in mejor_solucion_heur[0]],  # Grupo A en tiempo futuro
                [(n) for n in mejor_solucion_heur[1]]  # Grupo B + mecanismo
            )
            # Usar la mejor solución entre exhaustiva y heurística
            mejores = solucion_formateada
            if mejor_costo_heur < mejor_costo:
                mejor_costo = mejor_costo_heur
        else:
            print("No se encontró solución heurística")
        print(f"Mejor solución heurística:{mejores[0]} vs {mejores[1]}")
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
    def _tensor_slice(self, v_idx, estado):
        """
        Aplica un slice al tensor de probabilidad condicional de la variable v_idx
        para obtener P(X=0 | estado) y P(X=1 | estado).
        """
        tensor = self.sia_subsistema.ncubos[v_idx].data  # forma: (2^n, 2)
        idx = self._binario_a_entero(estado)
        prob_0 = tensor[idx, 0]
        prob_1 = tensor[idx, 1]
        return prob_0, prob_1

    def _binario_a_entero(self, binario):
        return int("".join(map(str, binario)), 2)

    def reconstruir_tpm(self):
        """
        Reconstruye la TPM completa como producto tensorial de los tensores elementales
        P(V^{t+1} | V^t) = ⊗_i P(X_i^{t+1} | V^t)
        Resultado: matriz de tamaño (2^n, 2^n)
        """
        tensores = [ncubo.data for ncubo in self.sia_subsistema.ncubos]  # cada uno de forma (2^n, 2)
        tpm = tensores[0]
        for t in tensores[1:]:
            tpm = np.tensordot(tpm, t, axes=0)  # tensorial product

        # reordenar a forma (2^n, 2^n)
        tpm = tpm.reshape((2**self.sia_subsistema.n_variables, 2**self.sia_subsistema.n_variables))
        return tpm

    def mostrar_slice_tensor(self, v_idx, condicion_estado):
        """
        Muestra un slice del tensor correspondiente a la variable v_idx
        condicionando en un estado específico del sistema.
        """
        prob_0, prob_1 = self._tensor_slice(v_idx, condicion_estado)
        print(f"P(X_{v_idx}=0 | estado={condicion_estado}) = {prob_0:.4f}")
        print(f"P(X_{v_idx}=1 | estado={condicion_estado}) = {prob_1:.4f}")
    
    def permutaciones_coordenadas(self, n):
        return list(permutations(range(n)))

    def complementaciones_coordenadas(self, n):
        return list(product([False, True], repeat=n))

    def aplicar_transformacion(self, estado, permutacion, complemento):
        estado_permutado = [estado[i] for i in permutacion]
        estado_transformado = [bit ^ int(comp) for bit, comp in zip(estado_permutado, complemento)]
        return estado_transformado
    def obtener_canonica(self, biparticion, n):
        # Genera todas las transformaciones posibles
        min_bip = None
        for perm in self.permutaciones_coordenadas(n):
            for comp in self.complementaciones_coordenadas(n):
                bip_transformada = frozenset([
                    self.aplicar_transformacion(list(b), perm, comp)
                    for b in biparticion
                ])
                if min_bip is None or bip_transformada < min_bip:
                    min_bip = bip_transformada
        return min_bip
    def encontrar_ruta_minima(self, origen, destino, estados_bin):
        
        n = len(estados_bin[0])
        visitado = set()
        cola = deque([(origen, [origen])])
        
        while cola:
            actual, camino = cola.popleft()
            if actual == destino:
                return camino
            visitado.add(tuple(actual))
            for i in range(n):
                vecino = actual[:]
                vecino[i] ^= 1  # cambiar un bit
                if tuple(vecino) not in visitado:
                    cola.append((vecino, camino + [vecino]))
