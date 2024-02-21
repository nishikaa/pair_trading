# Run project-predict endpoint
echo "testing '/project-predict' endpoint"
curl -X 'POST' "http://localhost:8000/project-predict" -L -H 'Content-Type: application/json' -d \
'
    {"text": ["I hate you.", "I love you."]}
' -w "\n"
