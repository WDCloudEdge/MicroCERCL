dir="$(dirname "$0")"
total_count=$1
is_scale=$2
let count=1

for ((i=1;i<=$total_count;i++))
do
	echo $3_mem_load_$count
    start_timestamp=$(date +%s) 
    start_timestamp=$((start_timestamp - 6 * 60))
    end_timestamp=$((start_timestamp + 12 * 60))
	kubectl apply -f $dir/mem_load.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") start create."
    sleep 180
	echo "$(date +"%Y-%m-%d %T") start delete."
	kubectl delete -f $dir/mem_load.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") finish delete."
	echo -e "\n"
	sleep 720
    python $dir/../log-collect/Log.py $3_mem_load_$count full $start_timestamp $end_timestamp &
	((count++))
done