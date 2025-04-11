DIR_MODEL="./models/run-2"
SAMPLE_DIR="./samples/run-2"

# Run two classification models in parallel (99 and 199)
echo "Starting first set of models..."
python generation_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_99.pth" -sp "$SAMPLE_DIR/m99" &
python generation_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_199.pth" -sp "$SAMPLE_DIR/m199" &

# Wait for all background jobs to complete
wait

# Run two classification models in parallel (299 and 399 and 499)
echo "Starting second set of models..."
python generation_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_299.pth" -sp "$SAMPLE_DIR/m299" &
python generation_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_399.pth" -sp "$SAMPLE_DIR/m399" &
python generation_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_499.pth" -sp "$SAMPLE_DIR/m499" &

# Wait for all background jobs to complete
wait

echo "All models have been evaluated."