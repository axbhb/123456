
# Start timing
SCRIPT_START_TIME=$(date +%s)

# run attacks
bash ./run_rsa_blind.sh
bash ./run_rsa_patch.sh
bash ./run_eta_blind.sh
bash ./run_eta_patch.sh

# show results
python print_attack_results.py

# End timing
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
echo "Total execution time: ${TOTAL_DURATION} seconds ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"