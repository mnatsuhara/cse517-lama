# script to clean up any downloaded models

CURR_DIR="$(realpath $(dirname "$0"))"
DEST_DIR_NAME="pretrained_lms"
DEST_DIR="$CURR_DIR/$DEST_DIR_NAME"

cd "$DEST_DIR"

echo "found the following models..."
for d in */ ; do
  echo "$d"
done

echo "are you sure you want to continue? (yes/no)"
read ans
if [ $ans == 'yes' ]; then
  for d in */ ; do
    echo "deleting $d"
    rm -rf $d
  done
fi

echo "deleting $DEST_DIR_NAME directory..."
cd ../
rm -rf "$DEST_DIR"
