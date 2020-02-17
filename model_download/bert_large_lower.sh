# script to download model for BERT model (large)

set -e	# exit immediately if cmd returns nonzero status
set -u	# treat unset vars as error when subbing

CURR_DIR="$(realpath $(dirname "$0"))"
DEST_DIR="$CURR_DIR/pretrained_lms"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "downloading _lowercase_ BERT large model..."

if [[ ! -f bert/uncased_L-24_H-1024_A-16/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
  unzip uncased_L-24_H-1024_A-16.zip
  rm uncased_L-24_H-1024_A-16.zip
  cd uncased_L-24_H-1024_A-16
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz"
  tar -xzf bert-large-uncased.tar.gz
  rm bert-large-uncased.tar.gz
  rm bert_model* # necessary?
  cd ../../
fi
