prefix=$(
  cd $(dirname $0)
  pwd
)
map=${1:-MoveToBeacon}
eps=${2:-0}
pushd "$prefix"
python3 -m pysc2.bin.agent --agent "ruled_agent.$map" --map "$map" --use_feature_units --max_episodes $eps
popd
