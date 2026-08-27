#!/usr/bin/env bash
# setup_code_dirs.sh
# Crea la estructura de subcarpetas dentro de code/ para modularizar
# el proyecto MBH-Seg25 (pipelines 3D/2D + utilidades compartidas + tests).
# No crea archivos, solo directorios (con .gitkeep para que git los rastree).
#
# Uso:
#   chmod +x setup_code_dirs.sh
#   ./setup_code_dirs.sh
#
# Ejecutar desde la raíz del repo (donde vive la carpeta code/), o pasar
# la ruta como primer argumento:
#   ./setup_code_dirs.sh /ruta/a/mi/repo

set -euo pipefail

ROOT_DIR="${1:-.}"
CODE_DIR="${ROOT_DIR}/code"

if [ ! -d "${CODE_DIR}" ]; then
  echo "Error: no se encontró la carpeta '${CODE_DIR}'. Verifica la ruta." >&2
  exit 1
fi

# Carpetas a crear dentro de code/
DIRS=(
  "config"
  "src/mbh_seg"
  "src/mbh_seg/common"
  "src/mbh_seg/pipeline_3d"
  "src/mbh_seg/pipeline_2d"
  "scripts"
  "tests"
  "data/raw"
  "data/interim"
  "data/processed"
  "outputs/checkpoints"
  "outputs/logs"
  "reports/figures"
)

echo "Creando estructura de carpetas dentro de: ${CODE_DIR}"

for d in "${DIRS[@]}"; do
  target="${CODE_DIR}/${d}"
  mkdir -p "${target}"
  # .gitkeep para que git rastree carpetas vacías
  touch "${target}/.gitkeep"
  echo "  ✓ ${target}"
done

echo "Listo. Estructura creada bajo ${CODE_DIR}/"
