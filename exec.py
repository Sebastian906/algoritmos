from src.middlewares.profile import profiler_manager
from src.main import leer_columna_excel  # Importar directamente desde main.py

def main():
    """Inicializar el aplicativo."""
    profiler_manager.enabled = True

    # Llamar directamente a leer_columna_excel desde main.py
    leer_columna_excel("c:\\Users\\mauri\\Downloads\\Prueba aotomatizador.xlsx", "Prueba aotomatizador")

if __name__ == "__main__":
    main()