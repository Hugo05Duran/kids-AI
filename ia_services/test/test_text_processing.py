import unittest
from unittest.mock import patch, MagicMock
from ia_services.assistant import AIAssistant

class TestTextResponse(unittest.TestCase):
    def setUp(self):
        self.assistant = AIAssistant()
        self.user_id = "test_user"
        self.message = "I'm really angry right now!"
        self.age = 10

    @patch.object(AIAssistant, 'get_response_based_on_instructions')
    @patch.object(AIAssistant, 'get_instructions_for_emotion', return_value=[
        'Recognize the emotion',
        'Validate the feelings',
        'Offer help'
    ])
    def test_anger_response(self, mock_get_instructions, mock_get_response_based_on_instructions):
        # Simula la respuesta del modelo
        mock_get_response_based_on_instructions.return_value = "It seems like you're feeling really angry. It's okay to feel that way sometimes. How can I help?"

        response = self.assistant.get_response(self.user_id, self.message, self.age)
        
        # Verifica que se siga la secuencia correcta de instrucciones
        mock_get_instructions.assert_called_with('anger')
        mock_get_response_based_on_instructions.assert_called_once()
        self.assertEqual(response, "It seems like you're feeling really angry. It's okay to feel that way sometimes. How can I help?")

if __name__ == '__main__':
    unittest.main()
