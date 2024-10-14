echo "exp2 hipster CPU"
kubectl apply -f ${hipster_cloud_edge_CPU1} -n hipster
sleep 600
curl ${web_url}/exp2?systemname=hipster&resource=CPU&size=250M
kubectl apply -f ${hipster_cloud_edge_CPU2} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=CPU&size=200M
kubectl apply -f ${hipster_cloud_edge_CPU3} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=CPU&size=150M
kubectl apply -f ${hipster_cloud_edge_CPU4} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=CPU&size=100M

echo "exp2 hipster MEM"
kubectl apply -f ${hipster_cloud_edge_MEM1} -n hipster
sleep 600
curl ${web_url}/exp2?systemname=hipster&resource=MEM&size=256Mi
kubectl apply -f ${hipster_cloud_edge_MEM2} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=MEM&size=192Mi
kubectl apply -f ${hipster_cloud_edge_MEM3} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=MEM&size=128Mi
kubectl apply -f ${hipster_cloud_edge_MEM4} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=MEM&size=64Mi
kubectl apply -f ${hipster_cloud_edge_MEM5} -n hipster
sleep 240

echo "exp2 hipster NET"
kubectl apply -f ${hipster_cloud_edge_NET1} -n hipster
sleep 600
curl ${web_url}/exp2?systemname=hipster&resource=NET&size=8M
kubectl apply -f ${hipster_cloud_edge_NET2} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=NET&size=4M
kubectl apply -f ${hipster_cloud_edge_NET3} -n hipster
sleep 240
curl ${web_url}/exp2?systemname=hipster&resource=NET&size=2M


