#!/usr/bin/env bash
# Rebuild everything from source: CAD -> STEP -> STL -> validation -> renders -> reports
set -euo pipefail
cd "$(dirname "$0")/src"
python3 generate_caliper.py      # body (SDF -> mesh), logo booleans, STEP files
python3 export_stl.py            # print STLs, orientation study, split + pins
python3 validate_model.py || echo "validation reported failures (see build/validation.json)"
python3 render_views.py          # renders/*.png
python3 make_reports.py          # reports/Dimensions.md, Print_Settings.md, Geometry_Validation.md
