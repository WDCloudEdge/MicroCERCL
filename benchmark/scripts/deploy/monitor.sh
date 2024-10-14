##Prometheus
kubectl apply -f ./deploy/prometheus/setup
kubectl apply -f ./deploy/prometheus

##Chaos-mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org

kubectl create ns chaos-mesh

helm install chaos-mesh chaos-mesh/chaos-mesh --namespace=chaos-mesh

##Tcpdump
sudo yum install tcpdump-4.9.2
