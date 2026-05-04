import numpy as np
import cv2
import pydicom
from pathlib import Path


def cargar_modelo_yolo(model_path="yolo26n-seg.pt"):
    """Carga un modelo YOLO (detección o segmentación)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "No se encontró 'ultralytics'. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    modelo = YOLO(model_path)
    print(f"✓ Modelo YOLO cargado desde: {model_path}")
    return modelo


def _dicom_to_uint8_rgb(img_path):
    ds = pydicom.dcmread(img_path)
    img = ds.pixel_array.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_u8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    return img, img_rgb


def predecir_imagen_yolo(modelo, img_path, conf=0.25, iou=0.45, device="cpu"):
    """Predicción YOLO sobre una imagen DICOM.

    Devuelve:
      - img_original (float32)
      - mascara_binaria (uint8)
      - mascara_prob (float32)
    """
    img_original, img_rgb = _dicom_to_uint8_rgb(img_path)
    h, w = img_original.shape

    results = modelo.predict(source=img_rgb, conf=conf, iou=iou, device=device, verbose=False)
    if not results:
        return img_original, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.float32)

    result = results[0]
    mascara_prob = np.zeros((h, w), dtype=np.float32)

    boxes = getattr(result, "boxes", None)
    confs = None
    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.detach().cpu().numpy()

    masks = getattr(result, "masks", None)
    if masks is not None and getattr(masks, "data", None) is not None and len(masks.data) > 0:
        masks_np = masks.data.detach().cpu().numpy().astype(np.float32)

        for idx, m in enumerate(masks_np):
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            det_conf = float(confs[idx]) if confs is not None and idx < len(confs) else 0.5
            mascara_prob = np.maximum(mascara_prob, m * det_conf)
    elif boxes is not None and len(boxes) > 0:
        boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        for idx, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            det_conf = float(confs[idx]) if confs is not None and idx < len(confs) else 0.5
            if x2 > x1 and y2 > y1:
                mascara_prob[y1:y2, x1:x2] = np.maximum(mascara_prob[y1:y2, x1:x2], det_conf)

    mascara_binaria = (mascara_prob >= max(0.1, conf * 0.8)).astype(np.uint8)
    return img_original, mascara_binaria, mascara_prob


def predecir_carpeta_yolo(modelo, carpeta_path, output_dir="predicciones_yolo", conf=0.25, iou=0.45, device="cpu"):
    """Predice todas las imágenes DICOM de una carpeta y guarda visualizaciones PNG."""
    from .predecir import visualizar_prediccion

    carpeta_path = Path(carpeta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    archivos = list(carpeta_path.glob("*.dcm"))
    print(f"\nEncontradas {len(archivos)} imágenes")

    for idx, archivo in enumerate(archivos):
        print(f"\nProcesando [{idx + 1}/{len(archivos)}]: {archivo.name}")
        img, mascara, mascara_prob = predecir_imagen_yolo(
            modelo,
            archivo,
            conf=conf,
            iou=iou,
            device=device,
        )
        save_path = output_dir / f"pred_yolo_{archivo.stem}.png"
        visualizar_prediccion(img, mascara, mascara_prob, save_path=save_path)

    print(f"\n✓ Todas las predicciones guardadas en: {output_dir}")
