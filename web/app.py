# ============================================================
# FRONTEND WEB SENCILLO (Flask + HTML)
# Subir imágenes T1wCE y marcar tumor con la IA
# ============================================================

import os
import shutil
import warnings
import uuid
from datetime import datetime, timedelta

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
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    flash,
    jsonify,
    session,
    send_from_directory,
    Response,
)
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from werkzeug.utils import secure_filename

import pydicom
from .models import db, User, LoginHistory, Note, PatientStudy

# ------------------------------------------------------------
# Configuración de rutas, modelos
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
DATABASE_PATH = os.path.join(BASE_DIR, "tumores_cerebrales.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def _ensure_runtime_directories():
    """Asegura que existan los directorios usados por la app web."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)


def _remove_uploaded_path(path):
    """Elimina un archivo temporal de uploads si existe."""
    try:
        if path and os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
    except Exception:
        pass


def _clear_uploads_folder():
    """Borra los temporales de uploads para mantener solo resultados."""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for entry in os.listdir(UPLOAD_FOLDER):
                full_path = os.path.join(UPLOAD_FOLDER, entry)
                _remove_uploaded_path(full_path)
    except Exception:
        pass


_clear_uploads_folder()


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

# Duración por defecto para sesiones "permanentes" (cuando el usuario marca "Recuérdame")
app.permanent_session_lifetime = timedelta(days=30)

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
    try:
        notes_result = db.session.execute(db.text("PRAGMA table_info(notes)"))
        note_columns = {row[1] for row in notes_result.fetchall()}
        if 'patient_id' not in note_columns:
            db.session.execute(db.text("ALTER TABLE notes ADD COLUMN patient_id VARCHAR(120)"))

        patient_result = db.session.execute(db.text("PRAGMA table_info(patient_studies)"))
        patient_columns = {row[1] for row in patient_result.fetchall()}
        if 'patient_name' not in patient_columns:
            db.session.execute(db.text("ALTER TABLE patient_studies ADD COLUMN patient_name VARCHAR(160)"))
        if 'result_images' not in patient_columns:
            db.session.execute(db.text("ALTER TABLE patient_studies ADD COLUMN result_images TEXT"))

        users_result = db.session.execute(db.text("PRAGMA table_info(users)"))
        user_columns = {row[1] for row in users_result.fetchall()}
        if 'profile_image' not in user_columns:
            db.session.execute(db.text("ALTER TABLE users ADD COLUMN profile_image BLOB"))
        if 'profile_image_mime' not in user_columns:
            db.session.execute(db.text("ALTER TABLE users ADD COLUMN profile_image_mime VARCHAR(64)"))
        if 'profile_image_updated_at' not in user_columns:
            db.session.execute(
                db.text("ALTER TABLE users ADD COLUMN profile_image_updated_at DATETIME")
            )

        db.session.commit()
    except Exception:
        db.session.rollback()


ALLOWED_AVATAR_EXTENSIONS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".gif"}
ALLOWED_AVATAR_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jfif": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
MAX_AVATAR_BYTES = 2 * 1024 * 1024  # 2 MB


@app.context_processor
def inject_avatar_helpers():
    """Expone helpers de avatar a todas las plantillas."""

    def user_avatar_url(user=None):
        user = user or current_user
        if (
            getattr(user, "is_authenticated", False)
            and getattr(user, "has_profile_image", False)
        ):
            version = ""
            if getattr(user, "profile_image_updated_at", None):
                version = user.profile_image_updated_at.strftime("%Y%m%d%H%M%S")
            return url_for("serve_profile_avatar", user_id=user.id, v=version)
        return url_for("static", filename="images/pfp-default/Default-pfp-tumoria.png")

    return {"user_avatar_url": user_avatar_url}

modelos = {}
DEVICE = None


def _import_ml_helpers():
    """Carga las utilidades de ML solo cuando se necesitan."""
    try:
        from ml.predecir import cargar_modelo, predecir_imagen
    except ImportError as exc:
        raise RuntimeError(
            "La funcionalidad de IA no está disponible en esta instalación. "
            "Si solo quieres usar la plataforma web, instala requirements-web.txt."
        ) from exc

    return cargar_modelo, predecir_imagen


def listar_modelos_disponibles():
    """Lista archivos .pth bajo models/ para el selector de la web."""
    if not os.path.isdir(MODELS_DIR):
        return []

    entries = []
    for root, _, files in os.walk(MODELS_DIR):
        for filename in sorted(files):
            if not filename.lower().endswith(".pth"):
                continue
            full_path = os.path.join(root, filename)
            rel_id = os.path.relpath(full_path, MODELS_DIR).replace("\\", "/")
            entries.append(
                {
                    "id": rel_id,
                    "label": rel_id,
                    "path": full_path,
                }
            )

    # Preferir best_model.pth al inicio de la lista si existe.
    entries.sort(key=lambda item: (0 if item["id"] == "best_model.pth" else 1, item["id"]))
    return entries


def resolver_ruta_modelo(model_id=None):
    """Resuelve un id relativo seguro a ruta absoluta dentro de models/."""
    available = {item["id"]: item for item in listar_modelos_disponibles()}
    if model_id:
        model_id = model_id.replace("\\", "/").lstrip("/")
        if model_id in available:
            return available[model_id]["path"], model_id

    if "best_model.pth" in available:
        return available["best_model.pth"]["path"], "best_model.pth"
    if available:
        first = next(iter(available.values()))
        return first["path"], first["id"]
    return MODEL_PATH, "best_model.pth"


def render_main_menu(
    images=None,
    error=None,
    summary=None,
    selected_model="unet",
    selected_weights=None,
):
    """Renderiza el menú principal incluyendo el catálogo de pesos .pth."""
    available_models = listar_modelos_disponibles()
    if selected_weights is None:
        _, selected_weights = resolver_ruta_modelo(None)
    return render_template(
        "main_menu.html",
        images=images,
        error=error,
        summary=summary,
        selected_model=selected_model,
        available_models=available_models,
        selected_weights=selected_weights,
    )


# ------------------------------------------------------------
# Carga perezosa del modelo
# ------------------------------------------------------------

def get_modelo(model_type="unet", model_id=None):
    global modelos, DEVICE

    try:  # No se necesita torch si no se entrenará IA en la máquina
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch no está instalado. La web puede abrir sin IA, pero la ruta /predict requiere ML."
        ) from exc

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type != "unet":
        raise ValueError("Tipo de modelo no soportado")

    model_path, resolved_id = resolver_ruta_modelo(model_id)
    cache_key = f"unet::{resolved_id}"
    if cache_key in modelos:
        return modelos[cache_key], DEVICE, resolved_id

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró un modelo entrenado en {MODELS_DIR}. Ejecuta entrenar.py primero."
        )

    cargar_modelo, _ = _import_ml_helpers()
    modelos[cache_key] = cargar_modelo(model_path, device=DEVICE)
    return modelos[cache_key], DEVICE, resolved_id


def predecir_con_modelo(model, model_type, dcm_path, device):
    _, predecir_imagen = _import_ml_helpers()
    return predecir_imagen(model, dcm_path, device=device, threshold=0.5)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def guardar_figura_prediccion(img, mascara, mascara_prob=None):
    """Genera una imagen PNG con la predicción y devuelve la ruta relativa."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo generar la imagen de resultado porque matplotlib no está instalado. "
            "La interfaz web puede abrir sin él, pero la ruta /predict requiere esta dependencia."
        ) from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(mascara, cmap="Reds", alpha=0.5)
    axes[1].set_title("Tumor detectado")
    axes[1].axis("off")

    plt.tight_layout()

    filename = f"pred_{uuid.uuid4().hex}.png"
    out_path = os.path.join(RESULTS_FOLDER, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return f"/static/results/{filename}"


def extraer_estadisticas_dicom(dcm_path, mascara):
    """Extrae métricas básicas útiles para el clínico."""
    try:
        # por alguna razon npQuier funciona pero no quiero eliminarle el Quier pq me da mello que ya no jale
        import numpy as npQuier

        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(npQuier.float32)
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
        ys, xs = npQuier.nonzero(mascara)
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
            
            remember = bool(request.form.get("remember"))
            login_user(user, remember=remember)
            # Establecer session.permanent solo si el usuario eligió "Recuérdame".
            session.permanent = remember
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
    """Vista principal de notas, opcionalmente filtrada por paciente."""
    selected_patient_id = (request.args.get("patient_id") or "").strip()
    query = Note.query.filter_by(user_id=current_user.id)
    if selected_patient_id:
        query = query.filter(
            (Note.patient_id == selected_patient_id)
            | (
                Note.patient_id.is_(None)
                & (
                    Note.title.contains(selected_patient_id)
                    | Note.content.contains(selected_patient_id)
                )
            )
        )
    user_notes = query.order_by(Note.updated_at.desc()).all()
    return render_template(
        "notes.html",
        notes=user_notes,
        selected_patient_id=selected_patient_id,
    )


@app.route("/notes/create", methods=["GET", "POST"])
@login_required
def create_note():
    """Crear una nueva nota; opcionalmente vinculada a un paciente."""
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        patient_id = request.form.get("patient_id", "").strip()

        if not content:
            flash("El contenido es requerido.", "danger")
            return redirect(request.referrer or url_for("notes"))

        if not title:
            title = f"Paciente: {patient_id}" if patient_id else "Nota nueva"

        note = Note(title=title, content=content, user_id=current_user.id, patient_id=patient_id or None)
        db.session.add(note)
        db.session.commit()

        flash("Nota creada exitosamente.", "success")
        if patient_id:
            return redirect(url_for("notes", patient_id=patient_id))
        return redirect(url_for("notes"))

    patient_id = request.args.get("patient_id", "").strip()
    return render_template("create_note.html", patient_id=patient_id)


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


@app.route('/pfp-default/<path:filename>')
def pfp_default(filename):
    """Sirve la imagen por defecto del perfil desde la carpeta static/images/pfp-default."""
    return send_from_directory(os.path.join(BASE_DIR, 'static', 'images', 'pfp-default'), filename)


@app.route("/profile/avatar/<int:user_id>")
@login_required
def serve_profile_avatar(user_id):
    """Sirve la foto de perfil almacenada en la base de datos."""
    user = User.query.get_or_404(user_id)
    if not user.profile_image:
        return redirect(
            url_for("static", filename="images/pfp-default/Default-pfp-tumoria.png")
        )
    return Response(
        user.profile_image,
        mimetype=user.profile_image_mime or "image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.route("/profile/avatar", methods=["POST"])
@login_required
def upload_profile_avatar():
    """Actualiza la foto de perfil del usuario autenticado."""
    file = request.files.get("avatar")
    if file is None or not file.filename:
        flash("No se seleccionó ninguna imagen.", "danger")
        return redirect(url_for("profile"))

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_AVATAR_EXTENSIONS:
        flash("Formato no permitido. Usa JPG, JFIF, PNG, WEBP o GIF.", "danger")
        return redirect(url_for("profile"))

    data = file.read()
    if not data:
        flash("El archivo de imagen está vacío.", "danger")
        return redirect(url_for("profile"))
    if len(data) > MAX_AVATAR_BYTES:
        flash("La imagen supera el límite de 2 MB.", "danger")
        return redirect(url_for("profile"))

    current_user.profile_image = data
    current_user.profile_image_mime = ALLOWED_AVATAR_MIMES.get(ext, "image/jpeg")
    current_user.profile_image_updated_at = datetime.utcnow()
    db.session.commit()

    flash("Foto de perfil actualizada.", "success")
    return redirect(url_for("profile"))


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


@app.route("/history", methods=["GET"])
@login_required
def patient_history():
    """Muestra el historial agrupado por paciente para el usuario autenticado."""
    search_query = (request.args.get("q") or "").strip()
    studies_query = PatientStudy.query.filter_by(user_id=current_user.id)
    if search_query:
        studies_query = studies_query.filter(
            (PatientStudy.patient_id.contains(search_query))
            | (PatientStudy.patient_name.contains(search_query))
        )

    studies = studies_query.order_by(PatientStudy.created_at.desc()).all()
    grouped = {}
    for study in studies:
        group = grouped.setdefault(
            study.patient_id,
            {
                "patient_id": study.patient_id,
                "patient_name": study.patient_name or "",
                "studies": [],
            },
        )
        if study.patient_name and not group["patient_name"]:
            group["patient_name"] = study.patient_name
        group["studies"].append(study)

    return render_template("history.html", grouped=list(grouped.values()), search_query=search_query)


@app.route("/history/<string:patient_id>/name", methods=["POST"])
@login_required
def save_patient_name(patient_id):
    """Guarda o actualiza el nombre opcional de un paciente en el historial."""
    patient_name = (request.form.get("patient_name") or "").strip()
    studies = PatientStudy.query.filter_by(user_id=current_user.id, patient_id=patient_id).all()
    if not studies:
        flash("No se encontraron registros para ese paciente.", "warning")
        return redirect(url_for("patient_history", q=patient_id))

    current_name = (studies[0].patient_name or "").strip()
    if not patient_name:
        flash("Escribe un nombre antes de guardarlo.", "warning")
        return redirect(url_for("patient_history", q=patient_id))

    for study in studies:
        study.patient_name = patient_name

    db.session.commit()
    if current_name:
        flash("Nombre del paciente actualizado correctamente.", "success")
    else:
        flash("Nombre del paciente guardado correctamente.", "success")
    return redirect(url_for("patient_history", q=patient_id))


@app.route("/history/<int:study_id>/delete", methods=["POST"])
@login_required
def delete_patient_study(study_id):
    """Elimina un estudio del historial del usuario autenticado."""
    study = PatientStudy.query.filter_by(id=study_id, user_id=current_user.id).first_or_404()
    db.session.delete(study)
    db.session.commit()
    flash("Registro del historial eliminado correctamente.", "success")
    return redirect(url_for("patient_history"))


@app.route("/main", methods=["GET"])
@login_required
def main_menu():
    return render_main_menu()


@app.route("/help", methods=["GET"])
@login_required
def help_page():
    """Muestra la guía de uso y las preguntas frecuentes."""
    return render_template("help.html")


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    model_type = (request.form.get("model_type") or "unet").strip().lower()
    if model_type not in {"unet", "yolo26"}:
        model_type = "unet"
    selected_weights = (request.form.get("model_weights") or "").strip() or None

    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_main_menu(
            error="No se envió ningún archivo",
            selected_model=model_type,
            selected_weights=selected_weights,
        )

    patient_id = (request.form.get("patient_id") or "").strip()
    if not patient_id:
        return render_main_menu(
            error="El ID del paciente es obligatorio para guardar el historial.",
            selected_model=model_type,
            selected_weights=selected_weights,
        )

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    try:
        model, device, selected_weights = get_modelo(
            model_type, model_id=selected_weights
        )
    except Exception as e:
        return render_main_menu(
            error=str(e),
            selected_model=model_type,
            selected_weights=selected_weights,
        )

    images_out = []
    _ensure_runtime_directories()
    _clear_uploads_folder()

    class Summary:
        def __init__(self):
            self.num_images = 0
            self.num_with_tumor = 0
            self.total_area_mm2 = 0.0
            self.model_type = model_type
            self.weights_id = selected_weights

    summary = Summary()

    if ext == ".dcm":
        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.dcm")
        _ensure_runtime_directories()
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
            _remove_uploaded_path(save_path)
            return render_main_menu(
                error=f"Error al procesar DICOM: {e}",
                selected_model=model_type,
                selected_weights=selected_weights,
            )
        finally:
            _remove_uploaded_path(save_path)

    elif ext == ".zip":
        import zipfile

        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.zip")
        extract_dir = os.path.join(UPLOAD_FOLDER, f"zip_{uuid.uuid4().hex}")
        _ensure_runtime_directories()
        file.save(save_path)

        try:
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(save_path, "r") as zf:
                zf.extractall(extract_dir)

            dcm_paths = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".dcm"):
                        dcm_paths.append(os.path.join(root, f))

            if not dcm_paths:
                return render_main_menu(
                    error="El ZIP no contiene archivos .dcm",
                    selected_model=model_type,
                    selected_weights=selected_weights,
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
            _remove_uploaded_path(save_path)
            _remove_uploaded_path(extract_dir)
            return render_main_menu(
                error=f"Error al procesar ZIP: {e}",
                selected_model=model_type,
                selected_weights=selected_weights,
            )
        finally:
            _remove_uploaded_path(save_path)
            _remove_uploaded_path(extract_dir)

    else:
        return render_main_menu(
            error="Tipo de archivo no soportado. Usa .dcm o .zip",
            selected_model=model_type,
            selected_weights=selected_weights,
        )

    if summary.num_images > 0:
        result_image_paths = [img["src"] for img in images_out if img.get("src")]
        existing_study = PatientStudy.query.filter_by(user_id=current_user.id, patient_id=patient_id).order_by(PatientStudy.created_at.desc()).first()
        study = PatientStudy(
            user_id=current_user.id,
            patient_id=patient_id,
            patient_name=existing_study.patient_name if existing_study else None,
            model_type=model_type,
            source_file=filename,
            num_images=summary.num_images,
            num_with_tumor=summary.num_with_tumor,
            total_area_mm2=float(summary.total_area_mm2),
            result_summary=(
                f"Imágenes analizadas: {summary.num_images}; "
                f"Con tumor: {summary.num_with_tumor}; "
                f"Área total estimada: {float(summary.total_area_mm2):.1f} mm²"
            ),
            result_images='|'.join(result_image_paths) if result_image_paths else None,
        )
        db.session.add(study)
        db.session.commit()

    return render_main_menu(
        images=images_out,
        summary=summary if summary.num_images > 0 else None,
        selected_model=model_type,
        selected_weights=selected_weights,
    )
