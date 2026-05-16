"""Wrapper para lanzar la aplicación Flask definida en web.app.

Este archivo solo importa la instancia `app` desde web/app.py y la ejecuta.
Toda la lógica de rutas, HTML y manejo de archivos está en web/app.py.
"""

import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Numpy built with MINGW-W64 on Windows 64 bits is experimental.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in (exp2|nextafter|log10)",
    category=RuntimeWarning,
    module=r"numpy\.core\.getlimits",
)

from web.app import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
