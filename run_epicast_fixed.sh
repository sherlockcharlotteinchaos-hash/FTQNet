#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/FTQNet/chickenpox_hungary.sh
bash scripts/FTQNet/dengue_colombia.sh
bash scripts/FTQNet/dengue_argentina.sh
bash scripts/FTQNet/influenza_us_2.sh
bash scripts/FTQNet/influenza_us.sh
bash scripts/FTQNet/ili.sh
bash scripts/FTQNet/covid-19.sh
bash scripts/FTQNet/wearable.sh
bash scripts/FTQNet/a_stool.sh
bash scripts/FTQNet/m3_stool.sh
bash scripts/FTQNet/f4_stool.sh
bash scripts/FTQNet/a_saliva.sh