import datetime
import sys
import openai
import yt_dlp
from pathlib import Path
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


@dataclass
class VideoProcessorConfig:
    temp_audio_file: str = "temp_audio.mp3"
    model: str = "whisper-1"
    output_dir: str = "transcripts"


class VideoProcessor:
    def __init__(self):
        """Initialize VideoProcessor with OpenAI API key from environment."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Set up OpenAI API key
        openai.api_key = self.api_key

        # Initialize config
        self.config = VideoProcessorConfig()

        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger('VideoProcessor')

    def download_youtube_audio(self, url: str) -> Dict:
        """
        Download audio from YouTube video with updated options to handle restrictions.
        """
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': self.config.temp_audio_file.replace('.mp3', ''),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'logtostderr': False,
                'geo_bypass': True,
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'hls'],
                        'player_client': ['android', 'web'],
                    }
                }
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.logger.info("Downloading video audio...")
                info = ydl.extract_info(url, download=True)
                return info

        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            # First try updating yt-dlp
            self.logger.info("Attempting to update yt-dlp...")
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
                # Retry download after update
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return info
            except Exception as update_error:
                self.logger.error(f"Error after update attempt: {str(update_error)}")
                raise

    def transcribe_audio(self, audio_path: str, video_title: str) -> Tuple[Dict, str]:
        """
        Transcribe audio file using OpenAI Whisper API and save with timestamps.
        """
        try:
            # Create safe filename
            safe_title = "".join([c if c.isalnum() or c in (' ', '-', '_') else '_' for c in video_title])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{safe_title}_{timestamp}"

            # Transcribe using OpenAI API
            self.logger.info("Transcribing audio...")
            with open(audio_path, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    model=self.config.model,
                    file=audio_file,
                    response_format="verbose_json"
                )

            # Save transcript with timestamps
            txt_path = os.path.join(self.config.output_dir, f"{output_filename}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                for segment in transcript['segments']:
                    # Convert time to HH:MM:SS format
                    start_time = int(segment['start'])
                    hours = start_time // 3600
                    minutes = (start_time % 3600) // 60
                    seconds = start_time % 60
                    timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

                    # Write to file
                    f.write(f"{timestamp} {segment['text'].strip()}\n")

            self.logger.info(f"Transcript saved to: {txt_path}")
            return transcript, txt_path

        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            raise

    def process_video(self, url: str) -> Tuple[Dict, Dict, str]:
        """
        Process video: download and transcribe.

        Args:
            url: YouTube video URL

        Returns:
            Tuple[Dict, Dict, str]: (transcription, video_info, transcript_path)
        """
        try:
            # Download audio
            video_info = self.download_youtube_audio(url)

            # Get video title
            video_title = video_info.get('title', 'Untitled')

            # Transcribe
            transcription, transcript_path = self.transcribe_audio(
                self.config.temp_audio_file,
                video_title
            )

            # Clean up
            self._cleanup()

            return transcription, video_info, transcript_path

        except Exception as e:
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up temporary files."""
        temp_file = Path(self.config.temp_audio_file)
        if temp_file.exists():
            temp_file.unlink()