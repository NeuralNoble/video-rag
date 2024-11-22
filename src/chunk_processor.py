# src/chunk_processor.py

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
import logging
from enum import Enum
import os
from dotenv import load_dotenv
import openai


# Load environment variables
load_dotenv()


class EmbeddingType(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


@dataclass
class ChunkConfig:
    chunk_size: int = 30  # seconds
    overlap: int = 5  # seconds
    embedding_type: EmbeddingType = EmbeddingType.HUGGINGFACE
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class ChunkProcessor:
    def __init__(self, config: Optional[ChunkConfig] = None):
        """Initialize the chunk processor with configuration."""
        self.config = config or ChunkConfig()
        self.logger = logging.getLogger('ChunkProcessor')

        # Initialize embedding model
        if self.config.embedding_type == EmbeddingType.HUGGINGFACE:
            self.logger.info(f"Initializing HuggingFace model: {self.config.hf_model_name}")
            self.model = SentenceTransformer(self.config.hf_model_name)
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not found in environment variables")

    def parse_timestamp(self, timestamp: str) -> int:
        """Convert [HH:MM:SS] format to seconds."""
        match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]', timestamp)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
        return 0

    def read_transcript(self, transcript_path: str) -> List[Dict]:
        """Read and parse transcript file with timestamps."""
        segments = []

        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                # Find timestamp and text
                timestamp_match = re.match(r'(\[\d{2}:\d{2}:\d{2}\])(.*)', line.strip())
                if timestamp_match:
                    timestamp, text = timestamp_match.groups()
                    start_time = self.parse_timestamp(timestamp)

                    # Calculate end time (use next segment's start time or add a fixed duration for the last segment)
                    if i < len(lines) - 1:
                        next_timestamp_match = re.match(r'(\[\d{2}:\d{2}:\d{2}\])', lines[i + 1].strip())
                        if next_timestamp_match:
                            end_time = self.parse_timestamp(next_timestamp_match.group(1))
                        else:
                            end_time = start_time + 5  # Default 5 seconds if can't determine
                    else:
                        end_time = start_time + 5  # Last segment

                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text.strip()
                    })

        return segments

    def create_chunks(self, segments: List[Dict], video_id: str) -> List[Dict]:
        """Create chunks from transcript segments."""
        chunks = []
        i = 0
        total_segments = len(segments)

        while i < total_segments:
            chunk_texts = []
            chunk_start = segments[i]['start']
            current_time = chunk_start

            # Collect segments for current chunk
            while i < total_segments and (current_time - chunk_start) < self.config.chunk_size:
                chunk_texts.append(segments[i]['text'])
                if i < total_segments - 1:
                    current_time = segments[i + 1]['start']
                else:
                    current_time = segments[i]['end']
                i += 1

            # Create chunk
            chunk_end = min(chunk_start + self.config.chunk_size, current_time)
            chunk = self._create_chunk_dict(
                " ".join(chunk_texts),
                chunk_start,
                chunk_end,
                video_id
            )
            chunks.append(chunk)

            # Move back for overlap
            while i > 0 and segments[i - 1]['start'] > (chunk_end - self.config.overlap):
                i -= 1

        return chunks

    def _create_chunk_dict(self, text: str, start_time: int, end_time: int, video_id: str) -> Dict:
        """Create a chunk dictionary with metadata."""
        return {
            "id": f"{video_id}_{start_time:06d}",
            "metadata": {
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "youtube_url": f"https://youtube.com/watch?v={video_id}&t={start_time}"
            }
        }

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks."""
        try:
            texts = [chunk['metadata']['text'] for chunk in chunks]

            if self.config.embedding_type == EmbeddingType.HUGGINGFACE:
                embeddings = self.model.encode(texts)
                for chunk, embedding in zip(chunks, embeddings):
                    chunk['values'] = embedding.tolist()
            else:
                # OpenAI embeddings
                for chunk in chunks:
                    response = openai.Embedding.create(
                        model="text-embedding-ada-002",
                        input=chunk['metadata']['text']
                    )
                    chunk['values'] = response['data'][0]['embedding']

            return chunks

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def process_transcript_file(self, transcript_path: str, video_id: str) -> List[Dict]:
        """Process a transcript file and return chunks with embeddings."""
        try:
            # Read and parse transcript
            segments = self.read_transcript(transcript_path)
            self.logger.info(f"Read {len(segments)} segments from transcript")

            # Create chunks
            chunks = self.create_chunks(segments, video_id)
            self.logger.info(f"Created {len(chunks)} chunks")

            # Generate embeddings
            chunks = self.generate_embeddings(chunks)
            self.logger.info("Generated embeddings for all chunks")

            return chunks

        except Exception as e:
            self.logger.error(f"Error processing transcript file: {str(e)}")
            raise