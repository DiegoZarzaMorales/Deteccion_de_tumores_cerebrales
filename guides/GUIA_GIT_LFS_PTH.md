# Guia rapida para subir un `.pth` con Git LFS

## 1) Instalar Git LFS
En tu PC de entrenamiento instala Git LFS una sola vez:

```bash
git lfs install
```

## 2) Clonar el repo
```bash
git clone <URL_DEL_REPOSITORIO>
cd Deteccion_de_tumores_cerebrales
```

## 3) Activar LFS para archivos `.pth`
Si el proyecto no lo tiene configurado, registra el patron:

```bash
git lfs track "*.pth"
```

Eso crea o actualiza `.gitattributes`. Luego guardalo en Git:

```bash
git add .gitattributes
git commit -m "Configurar Git LFS para modelos pth"
```

## 4) Copiar tu modelo entrenado
Coloca el archivo generado en la carpeta del proyecto, por ejemplo:

```bash
cp /ruta/del/modelo_entrenado.pth models/best_model.pth
```

## 5) Subir el archivo
```bash
git add models/best_model.pth
git commit -m "Agregar modelo entrenado"
git push
```

## 6) Verificar que LFS lo esta manejando
```bash
git lfs ls-files
```

Si aparece el `.pth`, esta bien configurado.


## Comandos rapidos
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add models/best_model.pth
git commit -m "Agregar modelo entrenado con Git LFS"
git push
git lfs ls-files
```
