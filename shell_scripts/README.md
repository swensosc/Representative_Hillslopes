# README for `Representative_Hillslopes` shell scripts
Sam S. Rabin (samrabin@ucar.edu)
2024-04-19

## `hand_analysis_loop.sh`

This script performs the Height Above Nearest Drainage (HAND) analysis for each gridcell. It breaks the analysis into a number of chunks to enable parallel processing. Do
```bash
./hand_analysis_loop.sh --help
```
for more information.

Note that if you want to submit each chunk's analysis to a separate batch job, you must provide `batch_submit_preamble.sh` (which will not be tracked by git). This script must `echo` everything except the actual call of `call_hand_analysis.sh`. It will be called in a subshell of `hand_analysis_loop.sh`, so you can refer to that script's variables `$log_out` and `$log_err` to save logs of STDOUT and STDERR, respectively.

### Example
This was tested on NSF NCAR's Casper machine like so (bash):

```bash
# Preamble for submitting a PBS job. Note "echo echo".
echo echo qsub -V -q casper -A $PROJECT -l walltime=12:00:00 -l select=1:ncpus=1:mem=32GB -o \${log_out} -e \${log_err} -- > batch_submit_preamble.sh

# Arguments
fsurdat="/glade/campaign/cesm/cesmdata/cseg/inputdata/lnd/clm2/surfdata_map/surfdata_0.9x1.25_78pfts_CMIP6_simyr2000_c170824.nc"
demdata="/glade/campaign/cgd/tss/projects/hillslopes/MERIT/data"
outdir="$SCRATCH/hillslope_script_testing/hand_analysis_loop"

# Call script
./hand_analysis_loop.sh $fsurdat $demdata $outdir
```
      

