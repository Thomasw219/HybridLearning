docker run --name hybrid_learning_env_$1 \
    -it \
    --rm \
    --gpus all \
    --mount type=bind,source="$(pwd)",target=/root/src \
    --dns 8.8.8.8 \
    thomasw219/hybrid_learning:latest
