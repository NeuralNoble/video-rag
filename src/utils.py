# src/utils.py

import re
from urllib.parse import urlparse, parse_qs
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> str:
    """
    Extract video ID from different YouTube URL formats.

    Args:
        url: YouTube URL in any format (e.g., youtu.be/xxx, youtube.com/watch?v=xxx)

    Returns:
        str: Video ID

    Raises:
        ValueError: If video ID cannot be extracted

    Examples:
        >>> extract_video_id("https://youtu.be/BErxU9o_gOk")
        'BErxU9o_gOk'
        >>> extract_video_id("https://www.youtube.com/watch?v=BErxU9o_gOk")
        'BErxU9o_gOk'
    """
    try:
        # Try parsing as standard URL first
        parsed_url = urlparse(url)

        # Case 1: youtu.be format
        if 'youtu.be' in parsed_url.netloc:
            video_id = parsed_url.path.lstrip('/')
            return video_id.split('?')[0]  # Remove any query parameters

        # Case 2: youtube.com format
        elif 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                return query_params['v'][0]

        # Case 3: Try regex as fallback
        pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)

        raise ValueError(f"Could not extract video ID from URL: {url}")

    except Exception as e:
        logger.error(f"Error extracting video ID from URL {url}: {str(e)}")
        raise ValueError(f"Could not extract video ID from URL: {url}")


def format_time(seconds: int) -> str:
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string

    Example:
        >>> format_time(3661)
        '01:01:01'
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_timestamp(timestamp: str) -> int:
    """
    Convert HH:MM:SS format to seconds.

    Args:
        timestamp: Time in format [HH:MM:SS]

    Returns:
        int: Time in seconds

    Example:
        >>> parse_timestamp("[01:01:01]")
        3661
    """
    match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})\]', timestamp)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return 0

# Can add more utility functions here as needed