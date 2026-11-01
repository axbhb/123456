export PYTHONPATH=$PYTHONPATH:"./"

# Start timing
SCRIPT_START_TIME=$(date +%s)

### set configs ###
attack_type="patch"
attack_loss="rsa"
dataset="asymmetric"
model="maptr-bevpool"

tag=""
device_id=0
### set configs ###


### attack ###
echo "=== Step 1/2: Running Attack ==="
echo "Running with ${attack_loss^^} attack using camera ${attack_type} in ${dataset} dataset"

# set attack options
attack_options="attack.type=${attack_type} attack.loss=${attack_loss} attack.dataset=${dataset} attack.tag=${tag}"

# run
show_dir="dataset/${model}"
config_file="./projects/configs/maptr/maptr_tiny_r50_24e_bevpool_asymmetric.py"
checkpoint_file="./ckpts/maptr_tiny_r50_24e_bevpool.pth"
attack_config_file="./attack_toolkit/configs/attack_cfg.yaml"

python -W ignore tools/attack.py $config_file $checkpoint_file \
    --attack_config_file $attack_config_file \
    --attack-options $attack_options \
    --show-dir $show_dir \
    --device-id $device_id \
    --eval chamfer
### attack ###


### planning ###
echo "=== Step 2/2: Running Planning ==="

exp_dir="dataset/${model}/train_${attack_type}_${attack_loss}_${dataset}"
gt_traj_dir="dataset/maptr-bevpool/train_blind_rsa_asymmetric/results/planning/gt"
clean_traj_dir="dataset/maptr-bevpool/train_blind_rsa_asymmetric/results/planning/clean"
attack_traj_dir="${exp_dir}/results/planning/attack"

python attack_toolkit/src/planners/HybridAStar_planner.py \
    --dataset $dataset \
    --root_dir $exp_dir \
    --gt_traj_dir $gt_traj_dir \
    --clean_traj_dir $clean_traj_dir \
    --attack-options $attack_options \
    --collision_threshold 0.5 \
    --vis
### planning ###


# Calculate total execution time
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo "=== Experiment Complete ==="
echo "Results saved to: $exp_dir"
echo "Total execution time: ${TOTAL_DURATION} seconds ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"