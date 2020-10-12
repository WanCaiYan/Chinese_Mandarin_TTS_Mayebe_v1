#!/bin/bash
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/hujk17/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/hujk17/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/hujk17/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/hujk17/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

dire2="sample_wavs"
[ -d "$dire2" ] && rm -r "$dire2"
conda activate tensorflow_p36
cd Taco-2-Rayhane-WaveGan-Norm
dire1="tacotron_output"
[ -d "$dire1" ] && rm -r "$dire1"
CUDA_VISIBLE_DEVICES='3' python synthesize.py
cd ../WaveGAN
conda activate p_wavegan36_official_1
CUDA_VISIBLE_DEVICES='3' parallel-wavegan-decode \
--checkpoint selftrain_zhaodan_v1_long/checkpoint-1000000steps.pkl \
--dumpdir ../Taco-2-Rayhane-WaveGan-Norm/tacotron_output/eval_for_wavegan_v1 \
--outdir ../sample_wavs
