# script to download model for BERT model (base)

set -e	# exit immediately if cmd returns nonzero status
set -u	# treat unset vars as error when subbing

CURR_DIR="$(realpath $(dirname "$0"))"
DEST_DIR="$CURR_DIR/pretrained_lms"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "downloading _cased_ ELMo 5.5B model..."

if [[ ! -f elmo/original/vocab-2016-09-10.txt.txt ]]; then
  mkdir -p 'elmo'
  cd elmo
  mkdir -p 'original'
  cd original
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
   wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_softmax_weights.hdf5"
  wget -c "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt"
  cd ../../
fi
