DOMAIN=michaeltay.mids255.com

# wait for the /health endpoint to return a 200 and then move on
finished=false
while ! $finished; do
	echo "[Printing pod status]"
	kubectl get pods -n michaeltay
	echo "[---------------------------------------------]"

    health_status=$(curl -o /dev/null -s -w "%{http_code}\n" -X GET "https://${DOMAIN}/health")
    if [ $health_status == "200" ]; then
        finished=true
        echo "API is ready"
    else
        echo "API not responding yet"
        sleep 3
    fi
done

# Run mlapi-predict endpoint
echo "testing '/mlapi-predict' endpoint"
curl -X 'POST' "https://${DOMAIN}/mlapi-predict" -L -H 'Content-Type: application/json' -d \
'
	{"text": ["I hate you.", "I love you."]}
' -w "\n"
