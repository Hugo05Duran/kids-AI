import unittest
from unittest.mock import patch, MagicMock
from ia_services.assistant import AIAssistant

class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        self.instance = AIAssistant()
        self.user_id = "test_user"
        self.audio_content = "https://example.com/audio.wav"
        self.age = 10

    @patch('ia_services.assistant.boto3.client')
    @patch('ia_services.assistant.requests.get')
    def test_transcribe_audio(self, mock_requests_get, mock_boto_client):
        # Mock the Transcribe client
        mock_transcribe = MagicMock()
        mock_boto_client.return_value = mock_transcribe
        
        # Mock the transcription job status
        mock_transcribe.get_transcription_job.return_value = {
            'TranscriptionJob': {
                'TranscriptionJobStatus': 'COMPLETED',
                'Transcript': {
                    'TranscriptFileUri': 'http://example.com/transcript.json'
                }
            }
        }

        # Mock the request to get the transcript
        mock_requests_get.return_value.json.return_value = {
            'results': {
                'transcripts': [{'transcript': 'This is a test transcript'}]
            }
        }

        transcript = self.instance.transcribe_audio(self.audio_content)
        self.assertEqual(transcript, 'This is a test transcript')

    @patch('ia_services.assistant.boto3.client')
    def test_synthesize_speech(self, mock_boto_client):
        # Mock the Polly client
        mock_polly = MagicMock()
        mock_boto_client.return_value = mock_polly

        # Mock the Polly response
        mock_polly.synthesize_speech.return_value = {
            'AudioStream': MagicMock(read=lambda: b'test_audio')
        }

        audio = self.instance.synthesize_speech('Hello world')
        self.assertEqual(audio, b'test_audio')

    @patch('ia_services.assistant.boto3.client')
    @patch('ia_services.assistant.requests.get')
    @patch.object(AIAssistant, 'classify_and_track_sentiment', return_value={'specific_classification': {'sadness': 0.6}})
    @patch.object(AIAssistant, 'get_response', return_value="I'm here to help")
    def test_process_audio_input(self, mock_get_response, mock_classify, mock_requests_get, mock_boto_client):
        # Mock both Transcribe and Polly clients
        mock_transcribe = MagicMock()
        mock_boto_client.return_value = mock_transcribe

        # Mock the transcription job status
        mock_transcribe.get_transcription_job.return_value = {
            'TranscriptionJob': {
                'TranscriptionJobStatus': 'COMPLETED',
                'Transcript': {
                    'TranscriptFileUri': 'http://example.com/transcript.json'
                }
            }
        }

        # Mock the request to get the transcript
        mock_requests_get.return_value.json.return_value = {
            'results': {
                'transcripts': [{'transcript': 'This is a test transcript'}]
            }
        }

        audio_response = self.instance.process_audio_input(self.user_id, self.audio_content, self.age)
        self.assertEqual(audio_response, b'test_audio')

if __name__ == '__main__':
    unittest.main()

