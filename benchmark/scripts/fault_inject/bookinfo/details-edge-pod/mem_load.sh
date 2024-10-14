
exec >> 2024-02-06_label.txt

file_path="/home/cron/jupiter-hybrid/bookinfo/details-edge-pod/$1_mem_load.yaml"

echo ${file_path}

let count=1

for ((i=1;i<=2;i++))
do
	echo mem_load_$count
	kubectl apply -f ${file_path} -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") start create."
	sleep 180
	echo "$(date +"%Y-%m-%d %T") start delete."
	kubectl delete -f ${file_path} -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") finish delete."
	echo -e "\n"
	sleep 720
	((count++))
done