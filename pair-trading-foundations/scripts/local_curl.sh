# wait for the /health endpoint to return a 200 and then move on
finished=false
while ! $finished; do
    health_status=$(curl -o /dev/null -s -w "%{http_code}\n" -X GET "http://localhost:8000/health")
    if [ $health_status == "200" ]; then
        finished=true
        echo "API is ready"
    else
        echo "API not responding yet"
        sleep 3
    fi
done

# Run project-predict endpoint
echo "testing '/mlapi-predict' endpoint"
curl -X 'POST' "http://localhost:8000/mlapi-predict" -L -H 'Content-Type: application/json' -d \
'{ "requested_pairs": 3, "duration_in_days": 120, "dollar_amt": 100 }' -w "\n"
