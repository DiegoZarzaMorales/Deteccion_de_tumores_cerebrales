# ============================================================
# FRONTEND WEB SENCILLO (Flask + HTML)
# Subir imágenes T1wCE y marcar tumor con la IA
# ============================================================

import os
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template_string

import pydicom

from ml.predecir import cargar_modelo, predecir_imagen

# ------------------------------------------------------------
# Configuración básica (usar raíz del proyecto)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "static"))

modelo = None
DEVICE = None


# ------------------------------------------------------------
# Carga perezosa del modelo
# ------------------------------------------------------------

def get_modelo():
    global modelo, DEVICE
    if modelo is not None:
        return modelo, DEVICE

    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado en {MODEL_PATH}. Ejecuta entrenar.py primero."
        )

    modelo = cargar_modelo(MODEL_PATH, device=DEVICE)
    return modelo, DEVICE


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def guardar_figura_prediccion(img, mascara, mascara_prob=None):
    """Genera una imagen PNG con la predicción y devuelve la ruta relativa."""
    fig_cols = 3 if mascara_prob is not None else 2
    fig, axes = plt.subplots(1, fig_cols, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(mascara, cmap="Reds", alpha=0.5)
    axes[1].set_title("Tumor detectado")
    axes[1].axis("off")

    if mascara_prob is not None and fig_cols == 3:
        im = axes[2].imshow(mascara_prob, cmap="hot")
        axes[2].set_title("Mapa de probabilidad")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    filename = f"pred_{uuid.uuid4().hex}.png"
    out_path = os.path.join(RESULTS_FOLDER, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return f"/static/results/{filename}"


def extraer_estadisticas_dicom(dcm_path, mascara):
    """Extrae métricas básicas útiles para el clínico."""
    try:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return {
            "area_px": int(mascara.sum()),
            "area_mm2": None,
            "side": None,
            "quality_warning": "No se pudo leer metadatos DICOM para métricas detalladas.",
        }

    h, w = arr.shape
    area_px = int(mascara.sum())

    area_mm2 = None
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is not None and len(spacing) >= 2 and area_px > 0:
        try:
            sy = float(spacing[0])
            sx = float(spacing[1])
            area_mm2 = area_px * sy * sx
        except Exception:
            area_mm2 = None

    side = None
    if area_px > 0:
        ys, xs = np.nonzero(mascara)
        if len(xs) > 0:
            cx = float(xs.mean())
            side = (
                "mitad izquierda de la imagen" if cx < w / 2 else "mitad derecha de la imagen"
            )

    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    dyn = float(arr_norm.max() - arr_norm.min())
    quality_warning = None
    if dyn < 0.2:
        quality_warning = "Bajo contraste: el realce tumoral podría no ser claramente visible."

    return {
        "area_px": area_px,
        "area_mm2": area_mm2,
        "side": side,
        "quality_warning": quality_warning,
    }


# ------------------------------------------------------------
# Plantilla HTML sencilla
# ------------------------------------------------------------

INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Detección de Tumores Cerebrales</title>
  <style>
    body { font-family: Arial, sans-serif; background: #1e272e; color: #ecf0f1; margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 40px auto; background: #2f3640; padding: 20px 30px; border-radius: 8px; }
    h1 { text-align: center; margin-bottom: 10px; }
    p.desc { text-align: center; color: #dcdde1; }
    form { margin-top: 20px; text-align: center; }
    .upload-box { border: 2px dashed #718093; padding: 20px; border-radius: 8px; background: #353b48; }
    input[type=file] { color: #ecf0f1; margin-top: 10px; }
    button { margin-top: 15px; background: #00a8ff; border: none; color: white; padding: 10px 20px; font-size: 15px; border-radius: 4px; cursor: pointer; }
    button:hover { background: #0097e6; }
    .note { font-size: 13px; color: #dcdde1; margin-top: 8px; }
    .results { margin-top: 30px; }
    .summary { margin-bottom: 20px; padding: 12px 16px; background: #273c75; border-radius: 6px; font-size: 14px; }
    .result-card { margin-bottom: 24px; padding: 10px; background: #353b48; border-radius: 6px; }
    .result-img { max-width: 100%; border-radius: 4px; box-shadow: 0 0 10px rgba(0,0,0,0.6); }
    .tag { margin-top: 8px; font-weight: bold; padding: 4px 8px; display: inline-block; border-radius: 4px; font-size: 13px; }
    .tag-ok { background: #2ecc71; color: #ffffff; }
    .tag-alert { background: #e74c3c; color: #ffffff; }
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Detección de Tumores Cerebrales</h1>
    <p class=\"desc\">Sube una imagen DICOM T1wCE o un ZIP con varias imágenes, y la IA marcará automáticamente la región tumoral.</p>

    <form action=\"/predict\" method=\"post\" enctype=\"multipart/form-data\">
      <div class=\"upload-box\">
        <p>Selecciona archivo (.dcm o .zip):</p>
        <input type=\"file\" name=\"file\" accept=\".dcm,.zip\" required />
        <p class=\"note\">• .dcm = una sola imagen T1wCE<br>• .zip = varias imágenes T1wCE</p>
      </div>
      <button type=\"submit\">Analizar con IA</button>
    </form>

    {% if error %}
      <div class=\"results\">
        <p style=\"color:#e84118;\">Error: {{ error }}</p>
      </div>
    {% endif %}

    {% if images %}
      <div class=\"results\">
        <h2>Resultados</h2>
        {% if summary %}
          <div class=\"summary\">
            <strong>Resumen del estudio:</strong><br>
            • Imágenes analizadas: {{ summary.num_images }}<br>
            • Imágenes con tumor: {{ summary.num_with_tumor }}<br>
            {% if summary.total_area_mm2 and summary.total_area_mm2 > 0 %}
              • Volumen relativo estimado: {{ \"%.1f\" % summary.total_area_mm2 }} mm²·slice<br>
            {% else %}
              • Volumen relativo estimado: basado en área en píxeles (sin escala física completa)<br>
            {% endif %}
          </div>
        {% endif %}
        {% for img in images %}
          <div class=\"result-card\">
            <img src=\"{{ img.src }}\" class=\"result-img\" />
            {% if img.area > 0 %}
              <div class=\"tag tag-alert\">TUMOR DETECTADO (área aprox. {{ img.area }} px)</div>
              <div style=\"margin-top:4px; font-size:13px;\">
                {% if img.area_mm2 %}
                  Tamaño estimado: {{ \"%.1f\" % img.area_mm2 }} mm².
                {% endif %}
                {% if img.side %}
                  Localización aproximada: {{ img.side }}.
                {% endif %}
                {% if img.quality_warning %}
                  <br><span style=\"color:#f5cd79;\">Aviso de calidad: {{ img.quality_warning }}</span>
                {% endif %}
              </div>
            {% else %}
              <div class=\"tag tag-ok\">SIN TUMOR CLARO</div>
            {% endif %}
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""


# ------------------------------------------------------------
# Rutas
# ------------------------------------------------------------


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, images=None, error=None, summary=None)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template_string(
            INDEX_HTML, images=None, error="No se envió ningún archivo", summary=None
        )

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    try:
        model, device = get_modelo()
    except Exception as e:
        return render_template_string(INDEX_HTML, images=None, error=str(e), summary=None)

    images_out = []

    class Summary:
        def __init__(self):
            self.num_images = 0
            self.num_with_tumor = 0
            self.total_area_mm2 = 0.0

    summary = Summary()

    if ext == ".dcm":
        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.dcm")
        file.save(save_path)

        try:
            img, mascara, mascara_prob = predecir_imagen(
                model, save_path, device=device, threshold=0.5
            )
            img_src = guardar_figura_prediccion(img, mascara, mascara_prob)
            area = int(mascara.sum())
            stats = extraer_estadisticas_dicom(save_path, mascara)

            summary.num_images += 1
            if area > 0:
                summary.num_with_tumor += 1
                if stats["area_mm2"]:
                    summary.total_area_mm2 += stats["area_mm2"]

            images_out.append(
                {
                    "src": img_src,
                    "area": area,
                    "area_mm2": stats["area_mm2"],
                    "side": stats["side"],
                    "quality_warning": stats["quality_warning"],
                }
            )
        except Exception as e:
            return render_template_string(
                INDEX_HTML,
                images=None,
                error=f"Error al procesar DICOM: {e}",
                summary=None,
            )

    elif ext == ".zip":
        import zipfile

        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.zip")
        file.save(save_path)

        extract_dir = os.path.join(UPLOAD_FOLDER, f"zip_{uuid.uuid4().hex}")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(save_path, "r") as zf:
                zf.extractall(extract_dir)

            dcm_paths = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".dcm"):
                        dcm_paths.append(os.path.join(root, f))

            if not dcm_paths:
                return render_template_string(
                    INDEX_HTML,
                    images=None,
                    error="El ZIP no contiene archivos .dcm",
                    summary=None,
                )

            for dcm_path in sorted(dcm_paths)[:20]:
                try:
                    img, mascara, mascara_prob = predecir_imagen(
                        model, dcm_path, device=device, threshold=0.5
                    )
                    img_src = guardar_figura_prediccion(img, mascara, mascara_prob)
                    area = int(mascara.sum())
                    stats = extraer_estadisticas_dicom(dcm_path, mascara)

                    summary.num_images += 1
                    if area > 0:
                        summary.num_with_tumor += 1
                        if stats["area_mm2"]:
                            summary.total_area_mm2 += stats["area_mm2"]

                    images_out.append(
                        {
                            "src": img_src,
                            "area": area,
                            "area_mm2": stats["area_mm2"],
                            "side": stats["side"],
                            "quality_warning": stats["quality_warning"],
                        }
                    )
                except Exception:
                    continue

        except Exception as e:
            return render_template_string(
                INDEX_HTML,
                images=None,
                error=f"Error al procesar ZIP: {e}",
                summary=None,
            )

    else:
        return render_template_string(
            INDEX_HTML,
            images=None,
            error="Tipo de archivo no soportado. Usa .dcm o .zip",
            summary=None,
        )

    return render_template_string(
        INDEX_HTML,
        images=images_out,
        error=None,
        summary=summary if summary.num_images > 0 else None,
    )
