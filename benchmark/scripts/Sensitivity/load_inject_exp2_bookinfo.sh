echo "exp2 bookinfo CPU"
kubectl apply -f ${bookinfo_cloud_edge_CPU1} -n bookinfo
sleep 600
curl ${web_url}/exp2?systemname=bookinfo&resource=CPU&size=600M
kubectl apply -f ${bookinfo_cloud_edge_CPU2} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=CPU&size=550M
kubectl apply -f ${bookinfo_cloud_edge_CPU3} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=CPU&size=500M
kubectl apply -f ${bookinfo_cloud_edge_CPU4} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=CPU&size=450M
kubectl apply -f ${bookinfo_cloud_edge_CPU5} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=CPU&size=400M
kubectl apply -f ${bookinfo_cloud_edge_CPU5} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=CPU&size=350M

echo "exp2 bookinfo MEM"
kubectl apply -f ${bookinfo_cloud_edge_MEM1} -n bookinfo
sleep 600
curl ${web_url}/exp2?systemname=bookinfo&resource=MEM&size=256Mi
kubectl apply -f ${bookinfo_cloud_edge_MEM2} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=MEM&size=192Mi
kubectl apply -f ${bookinfo_cloud_edge_MEM3} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=MEM&size=128Mi
kubectl apply -f ${bookinfo_cloud_edge_MEM4} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=MEM&size=64Mi
kubectl apply -f ${bookinfo_cloud_edge_MEM5} -n bookinfo


echo "exp2 bookinfo NET"
kubectl apply -f ${bookinfo_cloud_edge_NET1} -n bookinfo
sleep 600
curl ${web_url}/exp2?systemname=bookinfo&resource=NET&size=2M
kubectl apply -f ${bookinfo_cloud_edge_NET2} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=NET&size=1M
kubectl apply -f ${bookinfo_cloud_edge_NET3} -n bookinfo
sleep 240
curl ${web_url}/exp2?systemname=bookinfo&resource=NET&size=0.5M


