export PYTHONPATH=$PYTHONPATH:"./"

# Start timing
SCRIPT_START_TIME=$(date +%s)

### set configs ###
attack_type="blind"
attack_loss="rsa"
dataset="asymmetric"
model="maptr-bevpool"

tag=""
device_id=0
### set configs ###


### attack ###
# set attack options
attack_options="attack.type=${attack_type} attack.loss=${attack_loss} attack.dataset=${dataset} attack.tag=${tag}"

# run
show_dir="dataset/${model}"
config_file="./projects/configs/maptr/maptr_tiny_r50_24e_bevpool_asymmetric.py"
checkpoint_file="./ckpts/maptr_tiny_r50_24e_bevpool.pth"
attack_config_file="./attack_toolkit/configs/attack_cfg_gt.yaml"

python -W ignore tools/attack.py $config_file $checkpoint_file \
    --attack_config_file $attack_config_file \
    --attack-options $attack_options \
    --show-dir $show_dir \
    --device-id $device_id \
    --eval chamfer
### attack ###