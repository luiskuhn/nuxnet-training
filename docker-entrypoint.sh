#!/bin/sh
set -eu

# Preserve the convenient image CLI while allowing MLflow (and users) to
# replace the command with an executable such as ``python`` or ``bash``.
if [ "$#" -eq 0 ] || [ "${1#-}" != "$1" ]; then
    set -- python -m numorph_nuclei_segmentation.numorph_nuclei_segmentation "$@"
fi

exec "$@"
