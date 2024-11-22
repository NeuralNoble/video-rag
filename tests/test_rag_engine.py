# tests/test_chat.py

import logging
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)

from src import RAGEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ChatTest')


def format_sources(sources: list):
    """Format source information nicely."""
    if not sources:
        return

    print("\nBased on these moments:")
    for source in sources:
        print(f"\n🕒 {source['timestamp']}")
        print(f"💬 {source['text']}")
        print(f"🔗 Watch at: {source['url']}")


def main():
    """Run interactive chat test."""
    try:
        # Load environment variables
        load_dotenv()

        # Test video
        video_url = "https://youtu.be/BErxU9o_gOk"
        print(f"\nInitializing RAG for video: {video_url}")

        try:
            # Initialize RAG engine
            rag = RAGEngine(video_url)
            print("\nReady! Ask questions about the video")
            print("\nExample questions:")
            print("- What is this video about?")
            print("- How does the process work?")
            print("- What limitations are mentioned?")
            print("\nCommands:")
            print("- 'quit' to exit")
            print("- 'new' for new chat\n")

            while True:
                # Get user input
                query = input("\nYou: ").strip()

                # Handle commands
                if query.lower() == 'quit':
                    print("\nGoodbye! 👋")
                    break
                elif query.lower() == 'new':
                    rag.start_new_chat()
                    print("\nStarting new chat...")
                    continue
                elif not query:
                    continue

                try:
                    # Show when it's using previous context
                    if rag.should_use_last_context(query):
                        print("\n(Using context from previous question)")

                    # Get response using RAG
                    response = rag.chat(query)

                    # Print answer and sources
                    print(f"\nAssistant: {response['answer']}")
                    if response['sources']:
                        format_sources(response['sources'])

                    print("\n" + "=" * 50)

                except Exception as e:
                    logger.error(f"Error in chat: {str(e)}")
                    print("\nSorry, there was an error. Please try asking your question differently.")

        except Exception as e:
            logger.error(f"Error initializing RAG engine: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error setting up chat environment: {str(e)}")
        raise


if __name__ == "__main__":
    main()