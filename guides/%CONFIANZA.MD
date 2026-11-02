# Confianza de la predicción

La **confianza** resume qué tan seguro está el modelo de su decisión en la región relevante de la imagen. Se calcula en `ml/predecir.py` y se muestra en la web como porcentaje (`confianza × 100`).

Rango: valor en `[0, 1]` (0 %–100 %).

---

## Idea general

No se usa un único píxel ni el máximo de probabilidad. Se toma la **media** de la probabilidad asignada a la clase predicha:

- Si hay tumor → media solo sobre píxeles de tumor.
- Si no hay tumor → media de la probabilidad de “fondo / sin tumor” en toda la imagen.

Así, una detección positiva refleja la certeza dentro de la máscara, y una negativa refleja qué tan seguro está el modelo de que no hay tumor.

---

## Caso multiclasse (U-Net con 4 salidas)

Salidas del modelo tras **softmax**:

| Clase `c` | Significado |
|-----------|-------------|
| `0` | Fondo (sin tumor) |
| `1` | NCR (núcleo necrótico) |
| `2` | ED (edema) |
| `3` | ET (tumor realzante) |

### Variables

| Variable | Qué es |
|----------|--------|
| `logits` | Salida cruda del U-Net, forma `(1, C, H, W)` |
| `probs` / \(p_c(x)\) | Softmax por píxel: \(p_c(x) = \mathrm{softmax}(\mathrm{logits})_c(x)\). Forma `(C, H, W)`. Suma 1 en cada píxel \(x\). |
| `C` | Número de clases (aquí `4`) |
| `H`, `W` | Alto y ancho del mapa de predicción (p. ej. 256×256) |
| `clases` / \(\hat{y}(x)\) | Clase predicha: \(\hat{y}(x) = \arg\max_c\, p_c(x)\). Forma `(H, W)`. |
| `T` | Región de tumor: todos los píxeles con \(\hat{y}(x) > 0\) (NCR ∪ ED ∪ ET) |
| \(\|T\|\) | Número de píxeles en `T` |
| `confianza` | Escalar en `[0, 1]` que se reporta |

### Fórmulas

**Detección positiva** (\(\|T\| > 0\)):

\[
\mathrm{confianza} = \frac{1}{\|T\|} \sum_{x \in T} p_{\hat{y}(x)}(x)
\]

Es decir: en cada píxel de tumor se toma la probabilidad de **la clase que el modelo eligió ahí**, y se promedia.

**Detección negativa** (\(T\) vacío):

\[
\mathrm{confianza} = \frac{1}{HW} \sum_{x} p_0(x)
\]

Media de la probabilidad de fondo en toda la imagen.

### Ejemplo rápido

- Píxeles tumorales con \(p_{\hat{y}} \approx 0.85\) → confianza ≈ **85 %** (positiva).
- Sin tumor y \(p_0 \approx 0.92\) en promedio → confianza ≈ **92 %** (negativa).

---

## Caso binario (modelos con 1 salida)

Salida tras **sigmoid**: mapa `prob_map` / \(p(x) \in [0,1]\) = probabilidad de tumor (WT).

La máscara binaria `mascara_binaria` sale del postproceso (componente principal). Sea \(T = \{x : \mathrm{máscara}(x) = 1\}\).

**Positiva** (\(\|T\| > 0\)):

\[
\mathrm{confianza} = \frac{1}{\|T\|} \sum_{x \in T} p(x)
\]

**Negativa** (\(T\) vacío):

\[
\mathrm{confianza} = \frac{1}{HW} \sum_{x} \bigl(1 - p(x)\bigr)
\]

Media de la probabilidad de “no tumor”.

---

## Dónde se usa

1. `predecir_imagen()` calcula `confianza` y la devuelve como 4.º valor.
2. **Web** (`web/app.py`): título de la figura (`Detección: Positiva/Negativa (N%)`) y texto `Confianza: N%` junto al tamaño estimado.

---

## Qué no representa

- No es exactitud clínica ni probabilidad calibrada de diagnóstico.
- No mide el tamaño del tumor (eso es el área / mm²).
- Depende del umbral/postproceso solo en el sentido de **qué píxeles entran en `T`**; el valor promediado es la probabilidad del modelo en esos píxeles (o el fondo si `T` está vacío).
