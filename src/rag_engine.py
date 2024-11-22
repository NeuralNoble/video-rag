# src/rag_engine.py

import logging
from typing import List, Dict
import openai
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from src import PineconeManager
from src import extract_video_id

# Load environment variables
load_dotenv()


class RAGEngine:
    def __init__(self, video_url: str):
        """Initialize RAG Engine for a specific video."""
        self.logger = logging.getLogger('RAGEngine')

        # Initialize OpenAI
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key

        # Initialize embedding model for queries
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize vector store
        self.pinecone = PineconeManager("video-rag-test")

        # Set up video context
        self.video_id = extract_video_id(video_url)
        self.video_url = video_url

        # Keep track of conversation
        self.last_question = None
        self.last_chunks = None
        self.conversation_history = []

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text."""
        embedding = self.model.encode([query])[0]
        return embedding.tolist()

    def should_use_last_context(self, current_query: str) -> bool:
        """Determine if we should use the last context."""
        if not self.last_question or not self.last_chunks:
            return False

        # Use GPT to check if this is a follow-up
        prompt = f"""Given these two questions, is the second one a follow-up to the first one?
        Consider it a follow-up if it's:
        1. Asking for more details about the same topic
        2. Referring to something mentioned in the first question
        3. Using pronouns like "it", "that", "this" referring to the first question
        4. Asking for clarification about the first question

        Question 1: {self.last_question}
        Question 2: {current_query}

        Answer with just 'yes' or 'no'."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )

        return response.choices[0].message['content'].lower().strip() == 'yes'

    def get_relevant_chunks(self, query: str) -> List[Dict]:
        """Get relevant chunks from vector store using query."""
        query_embedding = self.generate_query_embedding(query)
        chunks = self.pinecone.search_video(query_embedding, self.video_id)
        return chunks

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate answer using OpenAI."""
        # Format conversation history
        history = "\n".join([
            f"Human: {h['question']}\nAssistant: {h['answer']}"
            for h in self.conversation_history[-2:]  # Last 2 exchanges
        ])

        # Format chunks
        context = "\n".join([
            f"[{chunk['start_time']}s - {chunk['end_time']}s]: {chunk['text']}"
            for chunk in chunks
        ])

        prompt = f"""Answer the question based on these video transcript excerpts and our conversation history.
        Use only information from the provided excerpts.

        Previous conversation:
        {history}

        Video transcript excerpts:
        {context}

        Question: {query}

        Answer: """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful AI that answers questions about videos based on their transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message['content'].strip()

    def chat(self, query: str) -> Dict:
        """Main chat method implementing RAG."""
        try:
            # Handle basic acknowledgments
            if query.lower().strip() in ['thanks', 'thank you', 'ok', 'okay']:
                return {
                    "answer": "You're welcome! Feel free to ask anything else about the video.",
                    "sources": []
                }

            # Determine if we should use previous context
            use_last_context = self.should_use_last_context(query)

            # Get relevant chunks (either new or reuse)
            chunks = self.last_chunks if use_last_context else self.get_relevant_chunks(query)

            if not chunks:
                return {
                    "answer": "I couldn't find relevant information in the video for that question. Could you rephrase it?",
                    "sources": []
                }

            # Generate answer
            answer = self.generate_answer(query, chunks)

            # Format response
            response = {
                "answer": answer,
                "sources": [
                    {
                        "timestamp": f"{chunk['start_time']}s - {chunk['end_time']}s",
                        "text": chunk['text'],
                        "url": f"{self.video_url}&t={chunk['start_time']}"
                    }
                    for chunk in chunks
                ]
            }

            # Update conversation tracking
            self.conversation_history.append({
                "question": query,
                "answer": answer
            })
            if not use_last_context:
                self.last_question = query
                self.last_chunks = chunks

            return response

        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            raise

    def start_new_chat(self):
        """Reset conversation tracking."""
        self.last_question = None
        self.last_chunks = None
        self.conversation_history = []