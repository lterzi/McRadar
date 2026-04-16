#! /bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --partition=small
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=l.terzi@lmu.de
#SBATCH --job-name=mcradar
#SBATCH --exclude=usm-cl-seitz3,usm-cl-183r01,th-cl-hua18

radarPosX=$1
time=$2
number_of_beams=$3
path=$4
elv=$5
source /home/l/L.Terzi/standard_python/bin/activate && \	
    python3 calc_McRadar_output.py $radarPosX $number_of_beams $time $path $elv && \
deactivate
