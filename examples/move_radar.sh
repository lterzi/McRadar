#radarPos=$(python -c "import numpy as np; print(' '.join(map(str, np.arange(-20000, 30000, 500))))")
radarPos=$(python -c "import numpy as np; print(' '.join(map(str, np.arange(-20000, 55000, 500))))")
path=exp059_u15_q12_ccn7_xi3e7_rt1_habit3_agg4_rfrag0_cfrag0_ffrag0
echo $radarPos
time=4200
n_beams=20
elv=30
#'9.6_35.5_94.0GHz_elv90_output_DDA_kdtree_melted_water_core_oriavgTru_gridVolume_beta{}_beta_std{}_particles0000${time}.000_radarPosX${pos}.nc'.format(elv[0],beta,beta_std,time,int(radarPosX1))
for pos in $radarPos; do
    echo $pos
    #file=9.6_35.5_94.0GHz_elv30_output_DDA_kdtree_melted_water_core_oriavgTru_gridVolume_beta0_beta_std30_particles0000${time}.000_radarPosX${pos}test.nc
    #filename=$path/McRadar/particles00004200/$file
    #if [ ! -f "$filename" ]; then
        #echo "File $file exists."
    #else
    #echo $filename
    #echo "File $filename does not exist. Processing..."
    sbatch send2slurm.sh $pos $time $n_beams $path $elv
    #fi
    #sbatch send2slurm.sh $pos $time $n_beams $path
done
