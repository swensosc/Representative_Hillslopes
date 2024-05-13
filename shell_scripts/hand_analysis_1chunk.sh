#!/usr/bin/env bash
set -e

scriptdir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
onechunk_preamble_script="${scriptdir}/1chunk_preamble.sh"
python_script="${scriptdir}/../hand_analysis_global.py"
cd "$(dirname "${python_script}")"

##########################
# Handle input arguments #
##########################

# Set default values of optional inputs
done_file=
dry_run=0

script="$0"
function usage {
    echo " "
    echo -e "usage: $script FSURDAT_FILE DEMDATA_DIR OUTPUT_DIR CNDX NCHUNKS [--done-file] [--dry-run]\n"
}
function help {
    usage
    echo "MANDATORY:"
    echo "   fsurdat_file   netCDF file from which latitude and longitude will be taken"
    echo "   demdata_dir    Directory containing digital elevation map data"
    echo "   output_dir     Directory to which outputs should be saved"
    echo "   cndx           Index (1-based) of the chunk to process"
    echo "   nchunks        How many chunks the grid is being split into"
    echo -e "\nOPTIONAL:"
    echo "   --done-file PATH   An empty file that will be created when the analysis completes successfully."
    echo "   --dry-run          Print the job submission commands that will be run, without actually running them."
}

# Handle mandatory arguments
n_mandatory_args=5
if [[ $# -lt ${n_mandatory_args} ]]; then
    echo "${script} requires ${n_mandatory_args} positional arguments: fsurdat_file, demdata_dir, output_dir, cndx, and nchunks" >&2
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
cndx="$1"; shift
nchunks="$1"; shift

# Handle optional arguments
while [ "$1" != "" ]; do
   case $1 in
       --done-file ) shift
           done_file="$1"
            ;;
       --dry-run )
           dry_run=1
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

if [[ ${dry_run} -eq 0 ]]; then
    cmd=
else
    echo " "
    cmd="echo -e"
fi
if [[ -f "${onechunk_preamble_script}" ]]; then
    cmd="$cmd $(. "${onechunk_preamble_script}")"
fi
$cmd "${python_script}" --sfcfile "${fsurdat}" --dem-data-path "${demdata}" -o "${outdir}" ${cndx}

if [[ ${dry_run} -eq 0 && "${done_file}" != "" ]]; then
   mkdir -p "$(dirname "${done_file}")"
   touch "${done_file}"
fi


exit 0
