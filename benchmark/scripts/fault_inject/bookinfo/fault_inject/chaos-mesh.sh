file_path='/home/cron/jupiter-hybrid/bookinfo/fault_inject/cpu_load.yaml'

echo ${file_path}

for ((i=1;i<=2;i++))
do
	kubectl apply -f ${file_path}
	echo "$(date +"%Y-%m-%d %T") start create."
	sleep 300
	echo "$(date +"%Y-%m-%d %T") start delete."
	kubectl delete -f ${file_path}
	echo "$(date +"%Y-%m-%d %T") finish delete."
	sleep 60
done