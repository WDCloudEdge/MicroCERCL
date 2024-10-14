dir="$(dirname "$0")"
total_count=$1
is_scale=$2
let count=1

for ((i=1;i<=$total_count;i++))
do
	echo $3_net_latency_$count
	kubectl apply -f $dir/net_latency.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") start create."
    sleep 60
    if [ $is_scale ]; then
        echo "$(date +"%Y-%m-%d %T") start scale up 1."
        kubectl scale deployment reviews-v1 --replicas=$(( $(kubectl get deployment reviews-v1 -n bookinfo -o=jsonpath='{.spec.replicas}') + 1)) -n bookinfo
    fi
	sleep 120
	echo "$(date +"%Y-%m-%d %T") start delete."
	kubectl delete -f $dir/net_latency.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") finish delete."
    sleep 60
    if [ $is_scale ]; then
        echo "$(date +"%Y-%m-%d %T") start scale down 1."
        kubectl scale deployment reviews-v1 --replicas=$(( $(kubectl get deployment reviews-v1 -n bookinfo -o=jsonpath='{.spec.replicas}') - 1)) -n bookinfo
    fi
	echo -e "\n"
	sleep 660
	((count++))
done