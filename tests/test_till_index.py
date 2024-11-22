# tests/test_pipeline.py
import glob
import logging
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.append(src_path)

from src import VideoProcessor
from src import ChunkProcessor
from src import PineconeManager
from src import extract_video_id

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PipelineTest')


def main():
    """Test the complete pipeline."""
    try:
        # Load environment variables
        load_dotenv()

        # Test video URL
        url = "https://youtu.be/BErxU9o_gOk"
        logger.info(f"Testing pipeline with URL: {url}")

        # Initialize components
        video_processor = VideoProcessor()
        chunk_processor = ChunkProcessor()
        pinecone = PineconeManager("video-rag-test")

        # Extract video ID
        video_id = extract_video_id(url)
        logger.info(f"Video ID: {video_id}")

        # Check if already indexed
        is_indexed = pinecone.check_video_exists(video_id)
        logger.info(f"Video already indexed: {is_indexed}")

        if is_indexed:
            logger.info("Video already indexed in Pinecone, skipping processing")
            return

        # Process video to get transcript
        logger.info("Processing video...")
        _, _, transcript_path = video_processor.process_video(url)
        logger.info(f"Transcript saved to: {transcript_path}")

        # Create chunks and index
        logger.info("Creating chunks and generating embeddings...")
        chunks = chunk_processor.process_transcript_file(transcript_path, video_id)
        logger.info(f"Created {len(chunks)} chunks")

        # Print sample chunks
        logger.info("\nSample chunks:")
        for i, chunk in enumerate(chunks[:2]):
            logger.info(f"\nChunk {i + 1}:")
            logger.info(f"ID: {chunk['id']}")
            logger.info(f"Time: {chunk['metadata']['start_time']}s - {chunk['metadata']['end_time']}s")
            logger.info(f"Text: {chunk['metadata']['text'][:100]}...")
            logger.info(f"Embedding Size: {len(chunk['values'])}")

        # Index chunks
        logger.info("\nIndexing chunks...")
        pinecone.index_video_chunks(chunks, video_id)
        logger.info("Chunks indexed successfully")

        logger.info("\nPipeline test completed successfully!")

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()