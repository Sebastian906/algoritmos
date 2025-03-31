from dataclasses import dataclass, field
from pathlib import Path
import time
import os
import numpy as np

from src.models.base.application import aplicacion
from src.constants.base import (
    ABC_START,
    COLON_DELIM,
    CSV_EXTENSION,
    SAMPLES_PATH,
    RESOLVER_PATH,
)

# Importar estrategias
from src.controllers.methods.compact_representation import CompactRepresentationStrategy
from src.controllers.methods.block_processing import BlockProcessingStrategy
from src.controllers.methods.parallelization import ParallelizationStrategy
from src.controllers.methods.memory_reduction import MemoryReductionStrategy

@dataclass
class Manager:
    """
    Clase para gestionar redes y configuraciones de procesamiento.
    """

    estado_inicial: str
    ruta_base: Path = Path(SAMPLES_PATH)

    # Diccionario que almacena las estrategias de optimización
    estrategias: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Inicializa el diccionario de estrategias con valores por defecto.
        """
        self.estrategias = {
            'representacion': CompactRepresentationStrategy(),
            'bloques': BlockProcessingStrategy(),
            'paralelizacion': ParallelizationStrategy(),
            'reduccion': MemoryReductionStrategy()
        }

    def preparar_estrategias(self, estrategias: dict = None):
        """
        Configura estrategias de gestión de memoria.
        
        Args:
            estrategias (dict, optional): Diccionario con estrategias personalizadas.
        """
        if estrategias:
            self.estrategias.update(estrategias)

    @property
    def pagina(self) -> str:
        return aplicacion.pagina_sample_network

    @property
    def tpm_filename(self) -> Path:
        return self.ruta_base / f"N{len(self.estado_inicial)}{self.pagina}.{CSV_EXTENSION}"

    @property
    def output_dir(self) -> Path:
        return Path(f"{RESOLVER_PATH}/N{len(self.estado_inicial)}{self.pagina}/{self.estado_inicial}")

    def generar_red(self, dimensiones: int, datos_discretos: bool = True) -> str:
        """
        Genera una red de estados (TPM) y la guarda en un archivo.
        """
        np.random.seed(aplicacion.semilla_numpy)

        if dimensiones < 1:
            raise ValueError("Las dimensiones deben ser positivas")

        # Cálculo del tamaño y tiempo estimado
        num_estados = 1 << dimensiones
        total_size_gb = (num_estados * dimensiones) / (1024**3)
        estimated_time = total_size_gb * 2

        print(f"Tamaño estimado: {total_size_gb:.6f} GB")
        print(f"Tiempo estimado: {estimated_time:.1f} segundos")

        if total_size_gb > 1:
            if input("El sistema ocupará más de 1GB. ¿Continuar? (s/n): ").lower() != "s":
                return None

        # Verificar existencia de archivos y generar nuevo nombre
        base_path = Path(SAMPLES_PATH)
        base_path.mkdir(parents=True, exist_ok=True)

        suffix = ABC_START
        while (base_path / f"N{dimensiones}{suffix}.{CSV_EXTENSION}").exists():
            if input(f"Ya existe N{dimensiones}{suffix}.{CSV_EXTENSION}. ¿Generar nueva red? (s/n): ").lower() != "s":
                return f"N{dimensiones}{suffix}.{CSV_EXTENSION}"
            suffix = chr(ord(suffix) + 1)

        filename = f"N{dimensiones}{suffix}.{CSV_EXTENSION}"
        filepath = base_path / filename

        # Generar estados
        print("Generando estados...")
        start_time = time.time()

        if datos_discretos:
            states = np.random.randint(2, size=(num_estados, dimensiones), dtype=np.int8)
        else:
            states = np.random.random(size=(num_estados, dimensiones))

        print(f"Generación completada en {time.time() - start_time:.2f} segundos")

        # Guardar archivo
        print(f"Guardando en {filepath}...")
        start_time = time.time()
        np.savetxt(filepath, states, delimiter=COLON_DELIM, fmt="%d" if datos_discretos else "%.6f")

        file_size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"Archivo guardado: {file_size_gb:.6f} GB")
        print(f"Tiempo de guardado: {time.time() - start_time:.2f} segundos")

        return filename