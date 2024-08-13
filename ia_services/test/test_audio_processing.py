import unittest
from unittest.mock import patch, MagicMock
from ia_services.assistant import AIAssistant

class TestAudioResponse(unittest.TestCase):
    def setUp(self):
        self.assistant = AIAssistant()
        self.user_id = "test_user"
        self.audio_content = b"audio binary content"
        self.age = 10

    @patch.object(AIAssistant, 'transcribe_audio', return_value="I'm really angry right now!")
    @patch.object(AIAssistant, 'synthesize_speech')
    @patch.object(AIAssistant, 'get_response_based_on_instructions', return_value="It seems like you're feeling really angry. It's okay to feel that way sometimes. How can I help?")
    @patch.object(AIAssistant, 'get_instructions_for_emotion', return_value=[
        'Recognize the emotion',
        'Validate the feelings',
        'Offer help'
    ])
    def test_anger_audio_response(self, mock_get_instructions, mock_get_response_based_on_instructions, mock_synthesize_speech, mock_transcribe_audio):
        # Simula la respuesta del modelo
        mock_synthesize_speech.return_value = b"audio_response_binary_data"

        audio_response = self.assistant.process_audio_input(self.user_id, self.audio_content, self.age)
        
        # Verifica que se siga la secuencia correcta de instrucciones y que se genere la respuesta en audio
        mock_get_instructions.assert_called_with('anger')
        mock_get_response_based_on_instructions.assert_called_once()
        mock_synthesize_speech.assert_called_once()
        self.assertEqual(audio_response, b"audio_response_binary_data")

if __name__ == '__main__':
    unittest.main()
