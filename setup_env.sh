export BASEPATH=/home/cc/sato
# RAW_DIR can be empty if using extracted feature files.
#export RAW_DIR=[path to the raw data]
export SHERLOCKPATH=$BASEPATH/sherlock
export EXTRACTPATH=$BASEPATH/extract
export PYTHONPATH=$PYTHONPATH:$SHERLOCKPATH
export PYTHONPATH=$PYTHONPATH:$BASEPATH
export TYPENAME='type78'
export CUDA_VISIBLE_DEVICES=0
