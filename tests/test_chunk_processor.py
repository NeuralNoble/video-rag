import logging
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.append(src_path)

from src import ChunkProcessor, ChunkConfig, EmbeddingType

# Setup logging
logging.basicConfig(level=logging.INFO)


def main():
    """Test the chunk processor with an actual transcript file."""
    try:
        # Initialize processor
        config = ChunkConfig(
            chunk_size=30,  # 30 seconds chunks
            overlap=5,  # 5 seconds overlap
            embedding_type=EmbeddingType.HUGGINGFACE
        )
        processor = ChunkProcessor(config)

        # Process transcript
        transcript_path = "../transcripts/How to Summarize a YouTube Video with ChatGPT_ _2024__20241119_003620.txt"  # Update with your transcript path
        video_id = "BErxU9o_gOk"  # This would be extracted from video info

        print(f"\nProcessing transcript: {transcript_path}")
        chunks = processor.process_transcript_file(transcript_path, video_id)

        # Print results
        print(f"\nProcessed {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"ID: {chunk['id']}")
            print(f"Time: [{chunk['metadata']['start_time']}s - {chunk['metadata']['end_time']}s]")
            print(f"Text: {chunk['metadata']['text'][:150]}...")
            print(f"URL: {chunk['metadata']['youtube_url']}")
            print(f"Embedding size: {len(chunk['values'])}")

        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()