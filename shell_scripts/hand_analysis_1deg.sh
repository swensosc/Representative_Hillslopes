#!/usr/bin/env bash
set -e

scriptdir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
script="${scriptdir}/call_hand_analysis.sh"

fsurdat="/glade/campaign/cesm/cesmdata/cseg/inputdata/lnd/clm2/surfdata_map/surfdata_0.9x1.25_78pfts_CMIP6_simyr2000_c170824.nc"
demdata="/glade/campaign/cgd/tss/projects/hillslopes/MERIT/data"
outdir="/glade/derecho/scratch/samrabin/hillslope_script_testing/1deg.96afd57"

nchunks=6
mem="32GB"
walltime="12:00:00"

outdir_logs="${outdir}/logs"
mkdir -p "${outdir_logs}"

nchunks_sq=$((nchunks*nchunks))
for cndx in $(seq 1 ${nchunks_sq}); do

    printf -v cndx_padded "%02d" ${cndx}
    printf "Chunk %d: " ${cndx}

    done_file="${outdir}/done_chunk${cndx_padded}"
    if [[ -e "${done_file}" ]]; then
        printf "Skipping (already done)\n"
        continue
    fi

    log_base="${outdir_logs}/chunk${cndx_padded}"

    pattern="${log_base}_*.err"
    if compgen -G "${pattern}" > /dev/null; then
        latest_log_err="$(ls -tr ${pattern} | tail -n 1)"
        if [[ $(grep -l "No DEM files found matching template" "${latest_log_err}" | wc -l) -gt 0 ]]; then
            printf "Skipping (previously tried but no files matching template)\n"
            continue
        fi
    fi

    printf "Submitting... "

    now="$(date "+%Y%m%d_%H%M%S")"
    log_base="${log_base}_${now}"
    log_out="${log_base}.out"
    log_err="${log_base}.err"

    qsub -V -q casper -A $PROJECT -l walltime="${walltime}" -l select=1:ncpus=1:mem="${mem}" -o "${log_out}" -e "${log_err}" -- "${script}" "${fsurdat}" "${demdata}" "${outdir}" ${cndx} ${nchunks} "${done_file}"
done


exit 0
