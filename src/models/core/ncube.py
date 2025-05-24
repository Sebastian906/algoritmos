from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np


@dataclass(frozen=True)
class NCube:
    """
    N-cubo hace referencia a un cubo n-dimensional, donde estarán indexados según la posición de precedencia de los datos, permitiendo el rápido acceso y operación en memoria.
    - `indice`: índice original del n-cubo asociado con un literal (0:A, 1:B, 2:C, ...) que permita representabilidad en su alcance o tiempo futuro.
    - `dims`: dimensiones activas actuales del n-cubo, es aquí donde se conoce la dimensionalidad según su cantidad de elementos, de forma tal que si este en el tiempo es condicionado o marginalizado tendrá una dimensionalidad menor o igual a la original a pesar que haya una alta dimensión específica.
    - `data`: arreglo numpy con los datos indexados según la notación de origen, de ser necesario se aplica una transformación sobre estos que los reindexe si se desea otra notación particular.
    """

    indice: int
    dims: NDArray[np.int8]
    data: np.ndarray

    def __post_init__(self):
        """Validación de tamaño y dimensionalidad tras inicialización.

        Raises:
            ValueError: Se valida que hayan dimensiones y cumpla con las dimensiones de un cubo n-dimensional.
        """
        if self.dims.size and self.data.shape != (2,) * self.dims.size:
            raise ValueError(
                f"Forma inválida {self.data.shape} para dimensiones {self.dims}"
            )

    def condicionar(
        self,
        indices_condicionados: NDArray[np.int8],
        estado_inicial: NDArray[np.int8],
    ) -> "NCube":
        numero_dims = self.dims.size
        seleccion = [slice(None)] * numero_dims

        # Solo condiciona si la dimensión está en self.dims
        condicionados_locales = []
        for condicion in indices_condicionados:
            if condicion in self.dims:
                level_arr = numero_dims - (np.where(self.dims == condicion)[0][0] + 1)
                seleccion[level_arr] = estado_inicial[condicion]
                condicionados_locales.append(condicion)

        nuevas_dims = np.array(
            [dim for dim in self.dims if dim not in condicionados_locales],
            dtype=np.int8,
        )
        nuevo_data = self.data[tuple(seleccion)]
        # Asegura que la forma de data y dims coincidan
        if nuevo_data.ndim == 0:
            nuevo_data = np.array([nuevo_data])
        return NCube(
            data=nuevo_data,
            dims=nuevas_dims,
            indice=self.indice,
        )

    def marginalizar(self, ejes: NDArray[np.int8]) -> "NCube":
        """
        Marginalizar a nivel del n-cubo permite acoplar o colapsar una o más dimensiones manteniendo la probabilidad condicional.
        El n-cubo puede esquematizarse de forma tal que se aprecie el solapamiento y promedio ente caras, donde la dimensión más baja es el primer desplazamiento dimensional sobre el arreglo.
        Es importante validar la intersección de ejes puesto es una rutina llamada en sistema desde marginalizar como particionar.

        Args:
        ----
            ejes (NDArray[np.int8]): Arreglo con las dimensiones a marginalizar o eliminar. Se valida que los ejes o dimensiones dadas estén y finalmente alineamos nuevamente con las dimensiones locales, donde con numpy debemos hacer uso de la dimensión complementaria para alinear la dimensión externa a la más interna.

        Returns:
        -------
            NCube: El n-cubo marginalizado en las dimensiones dadas. Donde es equivalente el marginalizar sobre (a, b,) que primero en (a,) y luego en (b,) o viceversa.

        Example:
        -------
            >>> dimensiones = np.array([2, 3])
            >>> mi_ncubo
            NCube(index=0):
            dims=[0 1 2]
            shape=(2, 2, 2)
            data=
                [[[0. 0.]
                [1. 1.]],
                [[1. 1.]
                [1. 1.]]]

            >>> mi_ncubo.marginalizar(dimensiones)
            NCube(index=0):
                dims=[0]
                shape=(2,)
                data=
                    [0.75 0.75]

            Se han agrupado los valores del n-cubo por promedio, dejando los remanentes en la dimension 0.
        """

        marginable_axis = np.intersect1d(ejes, self.dims)
        if not marginable_axis.size:
            return self
        numero_dims = self.dims.size - 1
        ejes_locales = tuple(
            numero_dims - dim_idx
            for dim_idx, axis in enumerate(self.dims)
            if axis in marginable_axis
        )
        new_dims = np.array(
            [d for d in self.dims if d not in marginable_axis],
            dtype=np.int8,
        )
        return NCube(
            data=np.mean(self.data, axis=ejes_locales, keepdims=False),
            dims=new_dims,
            indice=self.indice,
        )

    def __str__(self) -> str:
        dims_str = f"dims={self.dims}"
        forma_str = f"shape={self.data.shape}"
        datos_str = str(self.data).replace("\n", "\n" + " " * 8)
        return (
            f"NCube(index={self.indice}):\n"
            f"    {dims_str}\n"
            f"    {forma_str}\n"
            f"    data=\n        {datos_str}"
        )
