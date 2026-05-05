# ============================================================
# FRONTEND WEB SENCILLO (Flask + HTML)
# Subir imágenes T1wCE y marcar tumor con la IA
# ============================================================

import os
import uuid
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_required, login_user, logout_user, current_user

import pydicom

from ml.predecir import cargar_modelo, predecir_imagen
from .models import db, User, LoginHistory, Note

# ------------------------------------------------------------
# Configuración básica (usar raíz del proyecto)
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "models", "epoch_checkpoints", "checkpoint_epoch_10.pth")
DATABASE_PATH = os.path.join(BASE_DIR, "tumores_cerebrales.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates"),
)

# Configuración de seguridad y BD
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuración de sesiones para máxima compatibilidad
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Inicializar BD
db.init_app(app)

# Inicializar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Por favor inicia sesión para acceder a esta página.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Crear tablas si no existen
with app.app_context():
    db.create_all()

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
# Rutas de Autenticación
# ------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    """Ruta de login."""
    if current_user.is_authenticated:
        return redirect(url_for("main_menu"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            flash("Por favor completa todos los campos.", "danger")
            return redirect(url_for("login"))
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            # Registrar el login en el historial
            login_history = LoginHistory(
                user_id=user.id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', '')[:255]
            )
            db.session.add(login_history)
            db.session.commit()
            
            login_user(user, remember=request.form.get("remember", False))
            flash(f"¡Bienvenido, {user.username}!", "success")
            return redirect(url_for("main_menu"))
        else:
            flash("Usuario o contraseña incorrectos.", "danger")
    
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Ruta de registro."""
    if current_user.is_authenticated:
        return redirect(url_for("main_menu"))
    
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        password_confirm = request.form.get("password_confirm", "").strip()
        
        # Validaciones
        if not username or not email or not password:
            flash("Por favor completa todos los campos.", "danger")
            return redirect(url_for("register"))
        
        if len(username) < 3:
            flash("El nombre de usuario debe tener al menos 3 caracteres.", "danger")
            return redirect(url_for("register"))
        
        if password != password_confirm:
            flash("Las contraseñas no coinciden.", "danger")
            return redirect(url_for("register"))
        
        if len(password) < 6:
            flash("La contraseña debe tener al menos 6 caracteres.", "danger")
            return redirect(url_for("register"))
        
        if User.query.filter_by(username=username).first():
            flash("El nombre de usuario ya existe.", "danger")
            return redirect(url_for("register"))
        
        if User.query.filter_by(email=email).first():
            flash("El email ya está registrado.", "danger")
            return redirect(url_for("register"))
        
        # Crear usuario
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash("¡Cuenta creada exitosamente! Por favor inicia sesión.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    """Ruta de logout."""
    logout_user()
    flash("Has cerrado sesión.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------
# Rutas de Notas (Bloc de Notas)
# ------------------------------------------------------------

@app.route("/notes", methods=["GET"])
@login_required
def notes():
    """Vista principal de notas."""
    user_notes = Note.query.filter_by(user_id=current_user.id).order_by(Note.updated_at.desc()).all()
    return render_template("notes.html", notes=user_notes)


@app.route("/notes/create", methods=["GET", "POST"])
@login_required
def create_note():
    """Crear una nueva nota."""
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        
        if not title or not content:
            flash("El título y contenido son requeridos.", "danger")
            return redirect(url_for("create_note"))
        
        note = Note(title=title, content=content, user_id=current_user.id)
        db.session.add(note)
        db.session.commit()
        
        flash("Nota creada exitosamente.", "success")
        return redirect(url_for("notes"))
    
    return render_template("create_note.html")


@app.route("/notes/<int:note_id>/edit", methods=["GET", "POST"])
@login_required
def edit_note(note_id):
    """Editar una nota existente."""
    note = Note.query.get_or_404(note_id)
    
    # Verificar que el usuario sea el propietario
    if note.user_id != current_user.id:
        flash("No tienes permiso para editar esta nota.", "danger")
        return redirect(url_for("notes"))
    
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        
        if not title or not content:
            flash("El título y contenido son requeridos.", "danger")
            return redirect(url_for("edit_note", note_id=note_id))
        
        note.title = title
        note.content = content
        note.updated_at = datetime.utcnow()
        db.session.commit()
        
        flash("Nota actualizada exitosamente.", "success")
        return redirect(url_for("notes"))
    
    return render_template("edit_note.html", note=note)


@app.route("/notes/<int:note_id>/delete", methods=["POST"])
@login_required
def delete_note(note_id):
    """Eliminar una nota."""
    note = Note.query.get_or_404(note_id)
    
    # Verificar que el usuario sea el propietario
    if note.user_id != current_user.id:
        flash("No tienes permiso para eliminar esta nota.", "danger")
        return redirect(url_for("notes"))
    
    db.session.delete(note)
    db.session.commit()
    
    flash("Nota eliminada exitosamente.", "success")
    return redirect(url_for("notes"))


@app.route("/profile", methods=["GET"])
@login_required
def profile():
    """Ver perfil del usuario e historial de logins."""
    login_history = LoginHistory.query.filter_by(user_id=current_user.id).order_by(LoginHistory.login_date.desc()).all()
    return render_template("profile.html", login_history=login_history)


# ------------------------------------------------------------
# Rutas principales (protegidas con autenticación)
# ------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Redirige a login si no está autenticado, o a main_menu si lo está."""
    if current_user.is_authenticated:
        return redirect(url_for("main_menu"))
    return redirect(url_for("login"))


@app.route("/main", methods=["GET"])
@login_required
def main_menu():
    return render_template(
        "main_menu.html", images=None, error=None, summary=None, selected_model="unet"
    )


@app.route("/predict", methods=["POST"])
@login_required
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
