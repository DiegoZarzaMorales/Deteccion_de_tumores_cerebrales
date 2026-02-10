"""Wrapper para lanzar la aplicación Flask definida en web.app.

Este archivo solo importa la instancia `app` desde web/app.py y la ejecuta.
Toda la lógica de rutas, HTML y manejo de archivos está en web/app.py.
"""

from web.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
