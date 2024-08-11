import unittest
from unittest.mock import patch, MagicMock
from ia_services.assistant import AIAssistant

class AIAssistantTestCase(unittest.TestCase):
    def setUp(self):
        self.assistant = AIAssistant()
        self.user_id = "test_user"
        self.age = 10
        self.positive_message = "I got an A on my test!"
        self.negative_message = "I feel sad because I lost my toy."
        self.neutral_message = "I went to school today."

    @patch('openai.ChatCompletion.create')
    @patch('ia_services.content_filter.ContentFilter.classify_text')
    def test_get_response_negative(self, mock_classify, mock_openai):
        # Simular clasificación negativa
        mock_classify.return_value = {
            'general': {'negative': 0.6, 'neutral': 0.2, 'positive': 0.2},
            'specific_classification': {
                'anger': 0.1,
                'annoyance': 0.1,
                'disappointment': 0.1,
                'disapproval': 0.1,
                'disgust': 0.1,
                'fear': 0.1,
                'grief': 0.1,
                'sadness': 0.6,
                'remorse': 0.1
            }
        }
        mock_openai.return_value = MagicMock(choices=[MagicMock(message={'content': "Parece que estás pasando por algo difícil. ¿Quieres hablar más sobre ello? Estoy aquí para ayudarte."})])
        
        response = self.assistant.get_response(self.user_id, self.negative_message, self.age)
        self.assertIn("Parece que estás pasando por algo difícil", response)

    @patch('openai.ChatCompletion.create')
    @patch('ia_services.content_filter.ContentFilter.classify_text')
    def test_get_response_positive(self, mock_classify, mock_openai):
        # Simular clasificación positiva
        mock_classify.return_value = {
            'general': {'negative': 0.1, 'neutral': 0.2, 'positive': 0.7},
            'specific_classification': {
                'anger': 0.1,
                'annoyance': 0.1,
                'disappointment': 0.1,
                'disapproval': 0.1,
                'disgust': 0.1,
                'fear': 0.1,
                'grief': 0.1,
                'sadness': 0.1,
                'remorse': 0.1
            }
        }
        mock_openai.return_value = MagicMock(choices=[MagicMock(message={'content': "¡Eso suena genial!"})])
        
        response = self.assistant.get_response(self.user_id, self.positive_message, self.age)
        self.assertIn("¡Eso suena genial!", response)

if __name__ == '__main__':
    unittest.main()