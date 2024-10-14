
echo "exp1 bookinfo cloud"
kubectl apply -f ${bookinfo_file_path_cloud} -n bookinfo
sleep 300
curl ${web_url}/exp1?systemname=cloud
kubectl delete -f ${bookinfo_file_path_cloud} -n bookinfo
sleep 120


echo "exp1 bookinfo edge"
kubectl apply -f ${bookinfo_file_path_edge} -n bookinfo
sleep 300
curl ${web_url}/exp1?systemname=edge
kubectl delete -f ${bookinfo_file_path_edge} -n bookinfo
sleep 120

echo "exp1 bookinfo edge-cloud"
kubectl apply -f ${bookinfo_file_path_cloud_edge} -n bookinfo
sleep 300
curl ${web_url}/exp1?systemname=edge-cloud
kubectl delete -f ${bookinfo_file_path_cloud_edge} -n bookinfo
sleep 120



