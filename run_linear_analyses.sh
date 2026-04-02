#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run dihedral-features -> PCA -> TICA -> RMSF on an MD trajectory.

USAGE:
  run_linear_analyses.sh -t TOPOLOGY -x TRAJ_XTC -o PREFIX --lag LAG [options]

REQUIRED:
  -t, --topology   Topology file (.pdb/.gro; must match XTC atom count/order)
  -x, --xtc        Trajectory file (.xtc)
  -o, --out        Output prefix (e.g., hp35)
      --lag        TICA lag in frames (integer)

OPTIONS:
      --dt         Time per frame (for TICA implied timescales), default: 1.0
      --stride     Frame stride for all analyses, default: 1
      --k          Number of components for PCA/TICA, default: 5
      --selection  Atom selection for dihedrals (default: "protein")
      --plot       Generate PCA/TICA/RMSF plots
      --no-pbc-fix Disable PBC nojump handling in RMSF

EXAMPLE:
  ./run_linear_analyses.sh \
    -t /home/hitesh/Downloads/gromacs/native.pdb \
    -x /home/hitesh/Downloads/RUN195_combined.xtc \
    -o hp35 --lag 10 --dt 10 --plot
EOF
}

top=""
xtc=""
out=""
lag=""
dt="1.0"
stride="1"
k="5"
selection="protein"
plot="0"
no_pbc_fix="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--topology) top="${2:-}"; shift 2 ;;
    -x|--xtc) xtc="${2:-}"; shift 2 ;;
    -o|--out) out="${2:-}"; shift 2 ;;
    --lag) lag="${2:-}"; shift 2 ;;
    --dt) dt="${2:-}"; shift 2 ;;
    --stride) stride="${2:-}"; shift 2 ;;
    --k) k="${2:-}"; shift 2 ;;
    --selection) selection="${2:-}"; shift 2 ;;
    --plot) plot="1"; shift ;;
    --no-pbc-fix) no_pbc_fix="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$top" || -z "$xtc" || -z "$out" || -z "$lag" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

dihed_py="$script_dir/dihedrals_from_gro_xtc.py"
pca_py="$script_dir/pca_from_features.py"
tica_py="$script_dir/tica_from_features.py"
rmsf_py="$script_dir/rmsf_per_residue.py"

for f in "$dihed_py" "$pca_py" "$tica_py" "$rmsf_py"; do
  [[ -f "$f" ]] || { echo "Missing required script: $f" >&2; exit 1; }
done

echo "Topology: $top"
echo "Trajectory: $xtc"
echo "Out prefix: $out"
echo "Stride: $stride  PCA/TICA k: $k  TICA lag: $lag  dt: $dt"

echo
echo "1) Dihedral features (sin/cos) ..."
python3 "$dihed_py" -t "$top" -x "$xtc" -s "$selection" --stride "$stride" --out-prefix "$out"

echo
echo "2) PCA ..."
pca_args=( -i "${out}_features.npy" -k "$k" --out-prefix "$out" )
if [[ "$plot" == "1" ]]; then
  pca_args+=( --plot )
fi
python3 "$pca_py" "${pca_args[@]}"

echo
echo "3) TICA ..."
tica_args=( -i "${out}_features.npy" --lag "$lag" -k "$k" --dt "$dt" --out-prefix "${out}_tica_${lag}" )
if [[ "$plot" == "1" ]]; then
  tica_args+=( --plot --plot-timeseries )
fi
python3 "$tica_py" "${tica_args[@]}"

echo
echo "4) RMSF per residue (Cα, aligned) ..."
rmsf_args=( -t "$top" -x "$xtc" --stride "$stride" --out-prefix "${out}_rmsf" )
if [[ "$plot" == "1" ]]; then
  rmsf_args+=( --plot )
fi
if [[ "$no_pbc_fix" == "1" ]]; then
  rmsf_args+=( --no-pbc-fix )
fi
python3 "$rmsf_py" "${rmsf_args[@]}"

echo
echo "Done."
echo "Key outputs:"
echo "  - ${out}_features.npy"
echo "  - ${out}_pca.npy, ${out}_pca.png (if --plot)"
echo "  - ${out}_tica_${lag}_tica.npy, ${out}_tica_${lag}_tica.png (if --plot)"
echo "  - ${out}_rmsf.csv, ${out}_rmsf.png (if --plot)"

