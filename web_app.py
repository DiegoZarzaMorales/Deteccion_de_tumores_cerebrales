"""Wrapper para lanzar la aplicación Flask reorganizada en web.app."""

from web.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
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
  <div class="container">
    <h1>Detección de Tumores Cerebrales</h1>
    <p class="desc">Sube una imagen DICOM T1wCE o un ZIP con varias imágenes, y la IA marcará automáticamente la región tumoral.</p>

    <form action="/predict" method="post" enctype="multipart/form-data">
      <div class="upload-box">
        <p>Selecciona archivo (.dcm o .zip):</p>
        <input type="file" name="file" accept=".dcm,.zip" required />
        <p class="note">• .dcm = una sola imagen T1wCE<br>• .zip = varias imágenes T1wCE</p>
      </div>
      <button type="submit">Analizar con IA</button>
    </form>

    {% if error %}
      <div class="results">
        <p style="color:#e84118;">Error: {{ error }}</p>
      </div>
    {% endif %}

    {% if images %}
      <div class="results">
        <h2>Resultados</h2>
        {% if summary %}
          <div class="summary">
            <strong>Resumen del estudio:</strong><br>
            • Imágenes analizadas: {{ summary.num_images }}<br>
            • Imágenes con tumor: {{ summary.num_with_tumor }}<br>
            {% if summary.total_area_mm2 and summary.total_area_mm2 > 0 %}
              • Volumen relativo estimado: {{ "%.1f" % summary.total_area_mm2 }} mm²·slice<br>
            {% else %}
              • Volumen relativo estimado: basado en área en píxeles (sin escala física completa)<br>
            {% endif %}
          </div>
        {% endif %}
        {% for img in images %}
          <div class="result-card">
            <img src="{{ img.src }}" class="result-img" />
            {% if img.area > 0 %}
              <div class="tag tag-alert">TUMOR DETECTADO (área aprox. {{ img.area }} px)</div>
              <div style="margin-top:4px; font-size:13px;">
                {% if img.area_mm2 %}
                  Tamaño estimado: {{ "%.1f" % img.area_mm2 }} mm².
                {% endif %}
                {% if img.side %}
                  Localización aproximada: {{ img.side }}.
                {% endif %}
                {% if img.quality_warning %}
                  <br><span style="color:#f5cd79;">Aviso de calidad: {{ img.quality_warning }}</span>
                {% endif %}
              </div>
            {% else %}
              <div class="tag tag-ok">SIN TUMOR CLARO</div>
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
        return render_template_string(INDEX_HTML, images=None, error="No se envió ningún archivo")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    try:
        model, device = get_modelo()
    except Exception as e:
        return render_template_string(INDEX_HTML, images=None, error=str(e))

    images_out = []

    # Resumen global para ZIPs o estudios con varias imágenes
    class Summary:
      def __init__(self):
        self.num_images = 0
        self.num_with_tumor = 0
        self.total_area_mm2 = 0.0

    summary = Summary()

    if ext == ".dcm":
        # Guardar archivo subido
        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.dcm")
        file.save(save_path)

        try:
            img, mascara, mascara_prob = predecir_imagen(model, save_path, device=device, threshold=0.5)
            img_src = guardar_figura_prediccion(img, mascara, mascara_prob)
            area = int(mascara.sum())
            stats = extraer_estadisticas_dicom(save_path, mascara)

            summary.num_images += 1
            if area > 0:
              summary.num_with_tumor += 1
              if stats["area_mm2"]:
                summary.total_area_mm2 += stats["area_mm2"]

            images_out.append({
              "src": img_src,
              "area": area,
              "area_mm2": stats["area_mm2"],
              "side": stats["side"],
              "quality_warning": stats["quality_warning"],
            })
        except Exception as e:
            return render_template_string(INDEX_HTML, images=None, error=f"Error al procesar DICOM: {e}")

    elif ext == ".zip":
        import zipfile

        save_path = os.path.join(UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.zip")
        file.save(save_path)

        extract_dir = os.path.join(UPLOAD_FOLDER, f"zip_{uuid.uuid4().hex}")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(save_path, "r") as zf:
                zf.extractall(extract_dir)

            # Buscar todos los .dcm dentro del zip
            dcm_paths = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith(".dcm"):
                        dcm_paths.append(os.path.join(root, f))

            if not dcm_paths:
                return render_template_string(INDEX_HTML, images=None, error="El ZIP no contiene archivos .dcm")

            # Procesar algunos (o todos)
            for dcm_path in sorted(dcm_paths)[:20]:  # límite de 20 para no saturar
                try:
                img, mascara, mascara_prob = predecir_imagen(model, dcm_path, device=device, threshold=0.5)
                img_src = guardar_figura_prediccion(img, mascara, mascara_prob)
                area = int(mascara.sum())
                stats = extraer_estadisticas_dicom(dcm_path, mascara)

                summary.num_images += 1
                if area > 0:
                  summary.num_with_tumor += 1
                  if stats["area_mm2"]:
                    summary.total_area_mm2 += stats["area_mm2"]

                images_out.append({
                  "src": img_src,
                  "area": area,
                  "area_mm2": stats["area_mm2"],
                  "side": stats["side"],
                  "quality_warning": stats["quality_warning"],
                })
                except Exception:
                    continue

        except Exception as e:
            return render_template_string(INDEX_HTML, images=None, error=f"Error al procesar ZIP: {e}")

    else:
      return render_template_string(INDEX_HTML, images=None, error="Tipo de archivo no soportado. Usa .dcm o .zip")

    return render_template_string(INDEX_HTML, images=images_out, error=None, summary=summary if summary.num_images > 0 else None)


if __name__ == "__main__":
    # Ejecutar en desarrollo
    app.run(host="0.0.0.0", port=5000, debug=True)
