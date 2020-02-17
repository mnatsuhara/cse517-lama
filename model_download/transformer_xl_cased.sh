# script to download model for transformer XL model

set -e	# exit immediately if cmd returns nonzero status
set -u	# treat unset vars as error when subbing

CURR_DIR="$(realpath $(dirname "$0"))"
DEST_DIR="$CURR_DIR/pretrained_lms"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "downloading cased Transformer XL model..."

if [[ ! -f transformerxl/transfo-xl-wt103/config.json ]]; then
  mkdir -p 'transformerxl/transfo-xl-wt103'
  cd 'transformerxl/transfo-xl-wt103'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-vocab.bin' -O 'vocab.bin'
  # line from download_models.sh -- extracting plain text vocab for debugging purposes
  python -c 'import torch; print(*torch.load("vocab.bin")["sym2idx"].keys(), sep="\n")' | sort > vocab.txt
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-pytorch_model.bin' -O 'pytorch_model.bin'
  wget -c 'https://s3.amazonaws.com/models.huggingface.co/bert/transfo-xl-wt103-config.json' -O 'config.json'
  cd ../../
fi
