dir="$(dirname "$0")"
total_count=$1
is_scale=$2
let count=1

for ((i=1;i<=$total_count;i++))
do
    # start time
	echo $3_mem_load_$count
    start_timestamp=$(date +%s) 
    start_timestamp=$((start_timestamp - 6 * 60))
    # apply chaos
	kubectl apply -f $dir/mem_load.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") start create."
    sleep 60
    # scale up
    echo "$(date +"%Y-%m-%d %T") start scale up 1."
    kubectl scale deployment adservice --replicas=$(( $(kubectl get deployment adservice -n hipster -o=jsonpath='{.spec.replicas}') + 1)) -n hipster
	sleep 120
    # delete chaos
	echo "$(date +"%Y-%m-%d %T") start delete."
	kubectl delete -f $dir/mem_load.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") finish delete."
    sleep 60
    # scale down
    echo "$(date +"%Y-%m-%d %T") start scale down 1."
    before_timestamp=$(date +%s) 
    python $dir/../log-collect/Log.py $3_mem_load_$count before $start_timestamp $before_timestamp
    kubectl scale deployment adservice --replicas=$(( $(kubectl get deployment adservice -n hipster -o=jsonpath='{.spec.replicas}') - 1)) -n hipster
    # end time
    after_timestamp=$(date +%s)
	echo -e "\n"
	sleep $((660 - $after_timestamp + $before_timestamp))
    end_timestamp=$((before_timestamp + 5 * 60)) 
    python $dir/../log-collect/Log.py $3_mem_load_$count after $before_timestamp $end_timestamp &
    
    ((count++))
done