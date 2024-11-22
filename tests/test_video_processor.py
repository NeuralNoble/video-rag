from src import VideoProcessor
from dotenv import load_dotenv
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """
    Test the video processor functionality.
    """
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    try:
        # Initialize processor
        processor = VideoProcessor()

        # Test URL - short video for testing
        test_url = "https://youtu.be/BErxU9o_gOk?si=hW8C3hSXDiH_CZYd"  # First YouTube video

        print("Starting video processing...")
        print(f"URL: {test_url}")

        # Process video - note we now handle three return values
        transcription, video_info, transcript_path = processor.process_video(test_url)

        # Print results
        print("\nVideo Information:")
        print(f"Title: {video_info.get('title')}")
        print(f"Duration: {video_info.get('duration')} seconds")
        print(f"ID: {video_info.get('id')}")

        print("\nTranscription Sample:")
        print(transcription.text)

        print(f"\nTranscript saved to: {transcript_path}")

        print("\nTest successful!")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise


if __name__ == "__main__":
    main()