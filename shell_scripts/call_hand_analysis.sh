#!/usr/bin/env bash
set -e

fsurdat="$1"; shift
demdata="$1"; shift
outdir="$1"; shift
cndx="$1"; shift
nchunks="$1"; shift
done_file="$1"; shift

scriptdir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
script="${scriptdir}/../hand_analysis_global.py"
cd "$(dirname "${script}")"

module load ncarenv

conda run -n Representative_Hillslopes "${script}" --sfcfile "${fsurdat}" --dem-data-path "${demdata}" -o "${outdir}" ${cndx}

printf -v cndx_padded "%02d" ${cndx}
touch "${done_file}"


exit 0
