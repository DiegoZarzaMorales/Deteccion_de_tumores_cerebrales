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
from flask import Flask, request, render_template

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

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)

modelos = {}
DEVICE = None



# ------------------------------------------------------------
# Carga perezosa del modelo
# ------------------------------------------------------------

def get_modelo(model_type="unet"):
    global modelos, DEVICE
    if model_type in modelos:
        return modelos[model_type], DEVICE

    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "unet":
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en {MODEL_PATH}. Ejecuta entrenar.py primero."
            )
        modelos[model_type] = cargar_modelo(MODEL_PATH, device=DEVICE)
    else:
        raise ValueError("Tipo de modelo no soportado")

    return modelos[model_type], DEVICE


def predecir_con_modelo(model, model_type, dcm_path, device):
    return predecir_imagen(model, dcm_path, device=device, threshold=0.5)


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
# Rutas
# ------------------------------------------------------------


@app.route("/", methods=["GET"])
def main_menu():
    return render_template(
        "main_menu.html", images=None, error=None, summary=None, selected_model="unet"
    )


@app.route("/predict", methods=["POST"])
def predict():
    model_type = (request.form.get("model_type") or "unet").strip().lower()
    if model_type not in {"unet", "yolo26"}:
        model_type = "unet"

    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template(
            "main_menu.html",
            images=None,
            error="No se envió ningún archivo",
            summary=None,
            selected_model=model_type,
        )

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    try:
        model, device = get_modelo(model_type)
    except Exception as e:
        return render_template(
            "main_menu.html",
            images=None,
            error=str(e),
            summary=None,
            selected_model=model_type,
        )

    images_out = []

    class Summary:
        def __init__(self):
            self.num_images = 0
            self.num_with_tumor = 0
            self.total_area_mm2 = 0.0
            self.model_type = model_type

    summary = Summary()

    if ext == ".dcm":
        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.dcm")
        file.save(save_path)

        try:
            img, mascara, mascara_prob = predecir_con_modelo(model, model_type, save_path, device)
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
            return render_template(
                "main_menu.html",
                images=None,
                error=f"Error al procesar DICOM: {e}",
                summary=None,
                selected_model=model_type,
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
                return render_template(
                    "main_menu.html",
                    images=None,
                    error="El ZIP no contiene archivos .dcm",
                    summary=None,
                    selected_model=model_type,
                )

            for dcm_path in sorted(dcm_paths)[:20]:
                try:
                    img, mascara, mascara_prob = predecir_con_modelo(
                        model, model_type, dcm_path, device
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
            return render_template(
                "main_menu.html",
                images=None,
                error=f"Error al procesar ZIP: {e}",
                summary=None,
                selected_model=model_type,
            )

    else:
        return render_template(
            "main_menu.html",
            images=None,
            error="Tipo de archivo no soportado. Usa .dcm o .zip",
            summary=None,
            selected_model=model_type,
        )

    return render_template(
        "main_menu.html",
        images=images_out,
        error=None,
        summary=summary if summary.num_images > 0 else None,
        selected_model=model_type,
    )
