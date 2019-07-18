env_name=$1

for max_horizon in 100 200; do
    for rand_seed in 1234 2345 2314 1235; do
        # generate the config files
        exp_name=${env_name}_hor${max_horizon}_seed_${rand_seed}
        mkdir ./output/$exp_name

        # run the experiments
        python ./scripts/mbbl-run.py \
            --log-level debug --gpu \
            --history-buffer-size 20000 \
            --env $env_name --max-horizon $max_horizon --log ./output/$exp_name \
            2>&1 | tee ./output/${exp_name}/experiment.log
    done
done
