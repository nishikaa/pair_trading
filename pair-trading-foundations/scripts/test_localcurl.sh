# Run mlapi-predict endpoint
echo "testing '/mlapi-predict' endpoint"
curl -X 'POST' "http://localhost:8000/mlapi-predict" -L -H 'Content-Type: application/json' -d \
'
    {"text": ["I hate you.", "I love you."]}
' -w "\n"
