# src/pinecone_manager.py

from pinecone import Pinecone, Index
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class PineconeManager:
    # Model dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM = 384

    def __init__(self, index_name: str = "video-rag-test"):  # Changed default to our test index
        """
        Initialize Pinecone manager.

        Args:
            index_name: Name of the Pinecone index to use
        """
        self.logger = logging.getLogger('PineconeManager')

        # Initialize Pinecone
        api_key = os.getenv('PINECONE_API_KEY')

        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set in .env file")

        # Initialize Pinecone client and connect to existing index
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def check_video_exists(self, video_id: str) -> bool:
        """
        Check if video chunks already exist in the index.

        Args:
            video_id: YouTube video ID

        Returns:
            bool: True if video exists, False otherwise
        """
        try:
            # Query with filter for video_id
            results = self.index.query(
                vector=[0] * self.EMBEDDING_DIM,  # dummy vector matching dimension
                filter={"video_id": video_id},
                top_k=1
            )
            return len(results['matches']) > 0

        except Exception as e:
            self.logger.error(f"Error checking video existence: {str(e)}")
            raise

    def index_video_chunks(self, chunks: List[Dict], video_id: str):
        """
        Index video chunks in Pinecone.

        Args:
            chunks: List of chunks with embeddings and metadata
            video_id: YouTube video ID
        """
        try:
            # Validate embedding dimensions
            if len(chunks[0]['values']) != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.EMBEDDING_DIM}, "
                    f"got {len(chunks[0]['values'])}"
                )

            # Prepare vectors for upserting
            vectors = []
            for chunk in chunks:
                vector_data = {
                    "id": chunk["id"],
                    "values": chunk["values"],
                    "metadata": {
                        "video_id": video_id,
                        "start_time": chunk["metadata"]["start_time"],
                        "end_time": chunk["metadata"]["end_time"],
                        "text": chunk["metadata"]["text"],
                        "youtube_url": chunk["metadata"]["youtube_url"]
                    }
                }
                vectors.append(vector_data)

            # Upsert in batches of 100
            batch_size = 20
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            self.logger.info(f"Indexed {len(vectors)} chunks for video {video_id}")

        except Exception as e:
            self.logger.error(f"Error indexing video chunks: {str(e)}")
            raise

    def search_video(self, query_embedding: List[float], video_id: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant chunks within a specific video.

        Args:
            query_embedding: Embedding of the query text
            video_id: YouTube video ID to search within
            top_k: Number of results to return

        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Validate query embedding dimension
            if len(query_embedding) != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Query embedding dimension mismatch. Expected {self.EMBEDDING_DIM}, "
                    f"got {len(query_embedding)}"
                )

            # Query with video_id filter
            results = self.index.query(
                vector=query_embedding,
                filter={"video_id": video_id},  # Only search within this video
                top_k=top_k,
                include_metadata=True
            )

            # Format results
            formatted_results = []
            for match in results['matches']:
                formatted_results.append({
                    "score": match['score'],
                    "start_time": match['metadata']["start_time"],
                    "end_time": match['metadata']["end_time"],
                    "text": match['metadata']["text"],
                    "youtube_url": match['metadata']["youtube_url"]
                })

            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching video: {str(e)}")
            raise