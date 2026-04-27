#radarPos=$(python -c "import numpy as np; print(' '.join(map(str, np.arange(-20000, 30000, 500))))")
radarPos=$(python -c "import numpy as np; print(' '.join(map(str, np.arange(-10000, 70000, 500))))")
#path=exp059_u15_q12_ccn7_xi3e7_rt1_habit3_agg4_rfrag0_cfrag0_ffrag0
path=exp116_u10_q12_ccn7_xi5e9_rt1_kern3_agg5_rfrag0_cfrag0_ffrag0/
path=exp122_u10_q12_ccn7_xi5e9_rt1_kern3_agg5_rfrag0_cfrag6_ffrag0/
file=particles00010800_subset5_09.nc
echo $radarPos

time=10800 #4200
n_beams=20
elv=30
#'9.6_35.5_94.0GHz_elv90_output_DDA_kdtree_melted_water_core_oriavgTru_gridVolume_beta{}_beta_std{}_particles0000${time}.000_radarPosX${pos}.nc'.format(elv[0],beta,beta_std,time,int(radarPosX1))
for pos in $radarPos; do
    echo $pos
    fileName=9.6GHz_elv${elv}_output_DDA_kdtree_melted_water_core_oriavgTru_gridVolume_beta0_beta_std30_particles000010800.000_radarPosX${pos}_newsRange.nc
    #echo $path/McRadar/$fileName
    if [ -f "$path/McRadar/particles000010800/$fileName" ]; then
        echo "File $fileName already exists. Skipping..."
        continue
    fi
    sbatch send2slurm.sh $pos $time $n_beams $path$file $elv
    #python3 calc_McRadar_output.py $pos $n_beams $time $path $elv
    
done