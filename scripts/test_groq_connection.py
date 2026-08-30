from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from services.groq_client import GroqClient


def main():
    print('\nPHASE 12A: GROQ CONNECTION TEST')
    print('=' * 80)
    try:
        client = GroqClient(env_path=PROJECT_ROOT / '.env')
        result = client.test_connection()
        print('Status:', result['status'])
        print('Model:', result['model'])
        print('Base URL:', result['base_url'])
        print('Reply:', result['reply'])
        if result['status'] == 'success':
            print('\nGroq API key and model are working.')
        else:
            print('\nGroq responded, but the reply was unexpected. Check model/base URL.')
    except Exception as exc:
        print('\nGroq connection failed.')
        print('Error type:', type(exc).__name__)
        print('Error:', exc)
        print('\nCheck .env values:')
        print('GROQ_API_KEY=your_key_here')
        print('GROQ_BASE_URL=https://api.groq.com/openai/v1')
        print('GROQ_MODEL=openai/gpt-oss-20b')


if __name__ == '__main__':
    main()
