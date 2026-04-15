#!/bin/bash

mkdir -p logs

source ~/ood/bin/activate

# -----------------------------
# MODELS
# -----------------------------
MODELS=(
  "google/vit-base-patch16-224"
  "microsoft/resnet-50"
)

# -----------------------------
# ID DATASETS
# -----------------------------
ID_DATASETS=(
  "cifar10"
  "cifar100"
)

# -----------------------------
# OOD DATASETS (COMMON BENCHMARKS)
# -----------------------------
OOD_DATASETS=(
  "svhn"
  "places365"
  "lsun"
  "sun397"
  "dtd"
  "food101"
  "oxford_flowers102"
  "stanford_cars"
)

# -----------------------------
# DATASET-SPECIFIC LOGIC
# -----------------------------

get_ood_config () {
  case $1 in
    svhn) echo "cropped_digits" ;;
    places365) echo "" ;;
    lsun) echo "" ;;
    sun397) echo "" ;;
    dtd) echo "" ;;
    food101) echo "" ;;
    oxford_flowers102) echo "" ;;
    stanford_cars) echo "" ;;
    *) echo "" ;;
  esac
}

get_ood_split () {
  case $1 in
    svhn) echo "test" ;;
    places365) echo "train" ;;
    lsun) echo "train" ;;
    sun397) echo "test" ;;
    dtd) echo "test" ;;
    food101) echo "validation" ;;
    oxford_flowers102) echo "validation" ;;
    stanford_cars) echo "test" ;;
    *) echo "test" ;;
  esac
}

# -----------------------------
# METHODS
# -----------------------------
METHODS="msp,energy,odin,feat_knn,feat_maha,gradnorm,gradorth,lowdim_grad_resid,gradvec_maha,twosided_resid,twosided_code_maha,feat_gmm,feat_pca,react,vim"

# -----------------------------
# PARALLEL CONTROL
# -----------------------------
MAX_JOBS=2

# -----------------------------
# RUN FUNCTION
# -----------------------------
run_experiment () {
  MODEL=$1
  ID_DATASET=$2
  OOD_DATASET=$3

  OOD_CONFIG=$(get_ood_config $OOD_DATASET)
  OOD_SPLIT=$(get_ood_split $OOD_DATASET)

  # Map friendly dataset names to their actual Hugging Face Hub IDs
  REAL_OOD_DATASET=$OOD_DATASET
  case $OOD_DATASET in
    places365) REAL_OOD_DATASET="ljnlonoljpiljm/places365-256px" ;;
    lsun) REAL_OOD_DATASET="pcuenq/lsun-bedrooms" ;;
    sun397) REAL_OOD_DATASET="tanganke/sun397" ;;
    dtd) REAL_OOD_DATASET="tanganke/dtd" ;;
    oxford_flowers102) REAL_OOD_DATASET="dpdl-benchmark/oxford_flowers102" ;;
    stanford_cars) REAL_OOD_DATASET="tanganke/stanford_cars" ;;
  esac

  MODEL_NAME=$(echo $MODEL | tr '/' '_')

  LOG_FILE="logs/${MODEL_NAME}_${ID_DATASET}_vs_${OOD_DATASET}.out"

  echo "========================================"
  echo "MODEL: $MODEL"
  echo "ID:    $ID_DATASET"
  echo "OOD:   $OOD_DATASET"
  echo "REAL_OOD: $REAL_OOD_DATASET"
  echo "CFG:   $OOD_CONFIG"
  echo "SPLIT: $OOD_SPLIT"
  echo "LOG:   $LOG_FILE"
  echo "========================================"

  python3 hf_ood_eval.py \
    --model_id "$MODEL" \
    --id_dataset "$ID_DATASET" \
    --ood_dataset "$REAL_OOD_DATASET" \
    --ood_config "$OOD_CONFIG" \
    --ood_test_split "$OOD_SPLIT" \
    --methods "$METHODS" \
    --test_batch_size 1 \
    --max_ood_test 2000 \
    --max_id_test 2000 \
    --max_id_train 10000 \
    --num_workers 2 \
    > "$LOG_FILE" 2>&1
}

# -----------------------------
# GRID EXECUTION
# -----------------------------
job_count=0

for MODEL in "${MODELS[@]}"; do
  for ID in "${ID_DATASETS[@]}"; do
    for OOD in "${OOD_DATASETS[@]}"; do

      run_experiment "$MODEL" "$ID" "$OOD" &

      ((job_count++))

      # Control parallelism
      if (( job_count % MAX_JOBS == 0 )); then
        wait
      fi

    done
  done
done

wait

echo "All experiments completed."
