#!/usr/bin/env bash

export PYTHONPATH=$(pwd)

case "$1" in
    "train")
        CONFIG_NAME=$2
        NUM_EPOCHS=$3
        python3 configs/"${CONFIG_NAME}".py -d 0-7 -b 1 -e ${NUM_EPOCHS} --sync_bn 8 --no-clearml
        ;;
    "test")
        CONFIG_NAME=$2
        CKPT=$3
        python3 configs/"${CONFIG_NAME}".py -d 0-7 --eval --ckpt "${CKPT}"
        ;;
    "train-continue")
        CONFIG_NAME=$2
        CKPT=$3
        python3 configs/"${CONFIG_NAME}".py -d 0-7 -b 1 -e 30 --sync_bn 8 --no-clearml --ckpt "${CKPT}"
        ;;
    "pipeline")
        CONFIG_NAME=$2
        NUM_EPOCHS=$3
        CKPT_ID=$((NUM_EPOCHS-1))
        bash run.sh train ${CONFIG_NAME} ${NUM_EPOCHS}
        bash run.sh test ${CONFIG_NAME} outputs/${CONFIG_NAME}/latest/dump_model/checkpoint_epoch_${CKPT_ID}.pth
        ;;
    "reproduce")
        CONFIG_NAME=$2
        bash run.sh pipeline ${CONFIG_NAME} 30
        bash run.sh pipeline ${CONFIG_NAME} 110
        ;;
    *)
        echo "error"
esac
