kubectl apply -f samples/httpbin/httpbin.yaml -n horsecoder

kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: httpbin
  namespace: horsecoder
spec:
  hosts:
  - "production.horsecoder.com"
  gateways:
  - horsecoder-gateway
  http:
  - match:
    - uri:
        prefix: /status
    - uri:
        prefix: /delay
    route:
    - destination:
        port:
          number: 8000
        host: httpbin
EOF


