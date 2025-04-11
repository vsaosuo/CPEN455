# Use a relative path or set an environment variable for better portability
DIR_MODEL="./models/run-3"
# Alternatively, uncomment the following line if you need an absolute path
# DIR_MODEL="/path/to/models/run-2"

# Check if directory exists
if [ ! -d "$DIR_MODEL" ]; then
    echo "Error: Directory $DIR_MODEL does not exist!"
    exit 1
fi

# Run two classification models in parallel (99 and 199)
echo "Starting first set of models..."
python classification_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_99.pth" &
python classification_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_199.pth" &

# Wait for all background jobs to complete
wait

# Run three classification models in parallel (299, 399, and 499)
echo "Starting second set of models..."
python classification_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_299.pth" &
python classification_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_399.pth" &
python classification_evaluation.py -md "$DIR_MODEL/pcnn_cpen455_from_scratch_499.pth" &

# Wait for all background jobs to complete
wait

echo "All models have been evaluated."