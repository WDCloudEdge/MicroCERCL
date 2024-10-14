kubectl apply -f plus/istio-ingress.yaml

export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
export SECURE_INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="https")].port}')
export TCP_INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="tcp")].port}')

kubectl create -n istio-system secret generic horsecoder-credential \
--from-file=key=/home/data/cert/production.horsecoder.com_key.key \
--from-file=cert=/home/data/cert/production.horsecoder.com_chain.crt

kubectl apply -f plus/horsecoder-gateway.yaml
