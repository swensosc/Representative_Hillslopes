#!/usr/bin/env bash
set -e

scriptdir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
call_hand_analysis_script="${scriptdir}/call_hand_analysis.sh"
batch_submit_preamble_script="${scriptdir}/batch_submit_preamble.sh"

##########################
# Handle input arguments #
##########################

# Set default values of optional inputs
nchunks=6
dry_run=0

script="$0"
function usage {
    echo " "
    echo -e "usage: $script FSURDAT_FILE DEMDATA_DIR OUTPUT_DIR [-n/--nchunks]\n"
}
function help {
    usage
    echo "MANDATORY:"
    echo "   fsurdat_file   netCDF file from which latitude and longitude will be taken"
    echo "   demdata_dir    Directory containing digital elevation map data"
    echo "   output_dir     Directory to which outputs should be saved"
    echo -e "\nOPTIONAL:" 
    echo "   --dry-run        Print the job submission commands that will be run, without actually running them."
    echo "   -n/--nchunks N   Number of chunks to split processing into. Default: ${nchunks}"
}

# Handle mandatory arguments
n_mandatory_args=3
if [[ $# -lt ${n_mandatory_args} ]]; then
    echo "${script} requires 3 positional arguments: fsurdat_file, demdata_dir, and output_dir" >&2
    help
    exit 1
fi
fsurdat="$1"; shift
if [[ ! -e "${fsurdat}" ]]; then
    echo "fsurdat not found: ${fsurdat}" >&2
    exit 1
fi
demdata="$1"; shift
if [[ ! -e "${demdata}" ]]; then
    echo "demdata not found: ${demdata}" >&2
    exit 1
fi
outdir="$1"; shift
if [[ "${outdir}" == "-"* ]]; then
   # User might have accidentally left off outdir but added an optional argument
   while true; do
      echo "You requested output directory ${outdir}."
      read -p "Are you sure? yes or no: " yn
      case $yn in
         [Yy]* ) break;;
         [Nn]* ) exit;;
         * ) echo "Please answer yes or no.";;
      esac
   done
fi
echo -e "Saving to ${outdir}\n"

# Handle optional arguments
while [ "$1" != "" ]; do
   case $1 in
       --dry-run )
           dry_run=1
           ;;
       -n | --nchunks | --n-chunks ) shift
           nchunks=$1
           ;;
       -h | --help )
           help
           exit 0
           ;;
       * )
           echo "Invalid option: $1" >&2
           help
           exit 1
           ;;
   esac
   shift
done

##########################


outdir_logs="${outdir}/logs"
if [[ ${dry_run} -eq 0 ]]; then
   mkdir -p -- "${outdir_logs}"
fi

nchunks_sq=$((nchunks*nchunks))
now="$(date "+%Y%m%d_%H%M%S")"
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

    # Log file names
    log_base="${log_base}_${now}"
    log_out="${log_base}.out"
    log_err="${log_base}.err"


    if [[ ${dry_run} -eq 0 ]]; then
        printf "Submitting... "
        cmd=
    else
        cmd=echo
    fi
    if [[ -f ${batch_submit_preamble_script} ]]; then
        cmd="${cmd} $(. ${scriptdir}/batch_submit_preamble.sh)"
    fi
    $cmd "${call_hand_analysis_script}" "${fsurdat}" "${demdata}" "${outdir}" ${cndx} ${nchunks} "${done_file}"
    if [[ ${dry_run} -eq 1 ]]; then
        echo " "
    fi
done


exit 0
