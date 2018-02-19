export VOCAB_SOURCE=/data2/ec2bot/E2Cbot/cornell/Data/vocab50000.from
export VOCAB_TARGET=/data2/ec2bot/E2Cbot/cornell/Data/vocab50000.to
export TRAIN_SOURCES=/data2/ec2bot/E2Cbot/cornell/Data/dialog1.txt.ids50000
export TRAIN_TARGETS=/data2/ec2bot/E2Cbot/cornell/Data/dialog2.txt.ids50000
export DEV_SOURCES=/data2/ec2bot/E2Cbot/cornell/Data/testdialog1.txt.ids50000
export DEV_TARGETS=/data2/ec2bot/E2Cbot/cornell/Data/testdialog2.txt.ids50000
export TRAIN_STEPS=27800
export MODEL_DIR=/data2/ec2bot/E2Cbot/cornell/Model
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./config.yml,
      ./train.yml,
      ./text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
       - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 64 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
