from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .assistant import ai_assistant
import json

@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    data = json.loads(request.body)
    user_id = data.get('user_id')
    user_age = data.get('user_age')
    message = data.get('message')

    if not all([user_id, user_age, message]):
        return JsonResponse({"error": "Faltan datos requeridos"}, status=400)

    ai_response = ai_assistant.get_response(message, user_age)

    return JsonResponse({
        "response": ai_response
    })
