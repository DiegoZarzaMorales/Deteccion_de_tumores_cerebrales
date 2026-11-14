# Confianza de la predicción

En este proyecto hay **dos confianzas** (y, si aplica, una por subclase). Se calculan en `ml/predecir.py` y se muestran en la web como porcentaje (`valor × 100`).

Rango de cada valor: `[0, 1]` → **0 %–100 %**.

| Métrica | Qué responde | Dónde se ve |
|---------|--------------|-------------|
| **Confianza de detección** | ¿Hay tumor o no? | Panel «Detección» y texto `Confianza detección` |
| **Confianza de clasificación** | ¿Qué tan segura es la etiqueta NCR/ED/ET? | Panel «Clasificación (N%)» y texto `Confianza clasificación` |
| **Confianza por clase** | Certeza media de cada subregión presente | Leyenda del 3.er panel y línea `Clasificación: NCR (…%) · …` |

---

## Idea general

No se usa un único píxel ni el máximo. Se promedian probabilidades del **softmax** en la región relevante.

- **Detección** mira la masa tumoral frente al fondo (`1 − p_0`).
- **Clasificación** mira la probabilidad de la **clase elegida** en cada píxel tumoral (`p_ŷ`).

Así pueden diferir: el modelo puede estar muy seguro de que hay tumor, pero menos seguro de si un píxel es ED o ET.

---

## Caso multiclasse (U-Net con 4 salidas)

Salidas tras **softmax**:

| Clase `c` | Significado |
|-----------|-------------|
| `0` | Fondo (sin tumor) |
| `1` | NCR (núcleo necrótico) |
| `2` | ED (edema) |
| `3` | ET (tumor realzante) |

### Variables

| Variable | Qué es |
|----------|--------|
| `probs` / \(p_c(x)\) | Softmax por píxel. Forma `(C, H, W)`. Suma 1 en cada \(x\). |
| `clases` / \(\hat{y}(x)\) | \(\arg\max_c\, p_c(x)\) |
| `T` | Píxeles con \(\hat{y}(x) > 0\) (NCR ∪ ED ∪ ET) |
| \(\|T\|\) | Número de píxeles en `T` |
| \(T_c\) | Píxeles con \(\hat{y}(x) = c\) |

### 1) Confianza de detección

**Positiva** (\(\|T\| > 0\)):

\[
\mathrm{confianza\_deteccion} = \frac{1}{\|T\|} \sum_{x \in T} \bigl(1 - p_0(x)\bigr)
\]

Media de la probabilidad de “ser tumor” (cualquier subclase) en la máscara.

**Negativa** (\(T\) vacío):

\[
\mathrm{confianza\_deteccion} = \frac{1}{HW} \sum_{x} p_0(x)
\]

Media de la probabilidad de fondo en toda la imagen.

### 2) Confianza de clasificación

Solo si hay tumor:

\[
\mathrm{confianza\_clasificacion} = \frac{1}{\|T\|} \sum_{x \in T} p_{\hat{y}(x)}(x)
\]

Media de la probabilidad de **la clase predicha** en cada píxel tumoral.

Si no hay tumor → no se reporta (no hay subclase que puntuar).

### 3) Confianza por clase

Para cada clase tumoral presente \(c \in \{1,2,3\}\):

\[
\mathrm{confianza}_c = \frac{1}{\|T_c\|} \sum_{x \in T_c} p_c(x)
\]

### Ejemplo rápido

- En `T`, \(1-p_0 \approx 0.96\) y \(p_{\hat{y}} \approx 0.78\)  
  → detección **96 %**, clasificación **78 %**.
- Solo ED y ET presentes, con medias 0.82 y 0.71  
  → leyenda `ED (82%) · ET (71%)`.

---

## Caso binario (modelos con 1 salida)

Solo hay **confianza de detección** (sigmoid WT). No hay clasificación NCR/ED/ET:

**Positiva:** media de \(p(x)\) en la máscara.  
**Negativa:** media de \(1-p(x)\) en toda la imagen.  
`confianza_clasificacion` = `None`.

---

## Dónde se usa

1. `predecir_imagen()` devuelve detección, clasificación y dict por clase.
2. **Figura 1×3** (`web/app.py`): `%` en el título de Detección y de Clasificación; leyenda con `%` por clase.
3. **Tarjeta web**: `Confianza detección`, `Confianza clasificación` y `Clasificación: NCR (…%) · …`.

---

## Qué no representa

- No es exactitud clínica ni probabilidad calibrada de diagnóstico.
- No mide el tamaño del tumor (eso es el área / mm²).
- Detección alta + clasificación más baja es posible y coherente (tumor sí, subtipo más ambiguo).
