# script to download model for BERT model (base)

set -e	# exit immediately if cmd returns nonzero status
set -u	# treat unset vars as error when subbing

CURR_DIR="$(realpath $(dirname "$0"))"
DEST_DIR="$CURR_DIR/pretrained_lms"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "downloading _lowercase_ BERT base model..."

if [[ ! -f bert/uncased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
  unzip uncased_L-12_H-768_A-12.zip
  rm uncased_L-12_H-768_A-12.zip
  cd uncased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
  tar -xzf bert-base-uncased.tar.gz
  rm bert-base-uncased.tar.gz
  rm bert_model* # necessary?
  cd ../../
fi
