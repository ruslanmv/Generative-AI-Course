"""
Client for LLM Inference Server.

This module provides a command-line client for querying the FastAPI-based
LLM inference server with streaming response support.

Author: Ruslan Magana
Website: https://ruslanmv.com
License: Apache-2.0
"""

import argparse
import sys
import time
from typing import Optional

import requests


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the client.

    Returns:
        Parsed command-line arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Client for querying LLM Inference Server with streaming responses"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://127.0.0.1:8000/query-stream",
        help="URL of the FastAPI streaming endpoint"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Explain quantum computing in simple terms",
        help="Query text to send to the server"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds"
    )
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.05,
        help="Delay between processing chunks (seconds)"
    )
    return parser.parse_args()


def query_streaming_endpoint(
    endpoint: str,
    query: str,
    timeout: int = 300,
    chunk_delay: float = 0.05
) -> None:
    """
    Query the streaming endpoint and display the response in real-time.

    Args:
        endpoint: URL of the streaming API endpoint.
        query: Text query to send to the server.
        timeout: Maximum time to wait for response (seconds).
        chunk_delay: Delay between processing response chunks (seconds).

    Raises:
        requests.RequestException: If the HTTP request fails.
        KeyboardInterrupt: If the user interrupts the stream.
    """
    print(f"Query: {query}")
    print("=" * 80)
    print("Response:\n")

    try:
        response = requests.get(
            endpoint,
            params={"query": query},
            stream=True,
            timeout=timeout
        )
        response.raise_for_status()

        # Stream and display the response
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
                time.sleep(chunk_delay)

        print("\n")
        print("=" * 80)
        print("✓ Response completed successfully")

    except requests.exceptions.Timeout:
        print("\n✗ Error: Request timed out", file=sys.stderr)
        sys.exit(1)

    except requests.exceptions.ConnectionError:
        print(
            f"\n✗ Error: Could not connect to server at {endpoint}",
            file=sys.stderr
        )
        print("Make sure the server is running.", file=sys.stderr)
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}", file=sys.stderr)
        print(f"Status code: {response.status_code}", file=sys.stderr)
        print(f"Response: {response.text}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n✗ Stream interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the client application.

    Parses command-line arguments and queries the streaming endpoint.
    """
    args = parse_arguments()

    print("=" * 80)
    print("LLM Inference Server - Streaming Client")
    print("=" * 80)
    print(f"Endpoint: {args.endpoint}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 80)
    print()

    query_streaming_endpoint(
        endpoint=args.endpoint,
        query=args.query,
        timeout=args.timeout,
        chunk_delay=args.chunk_delay
    )


if __name__ == "__main__":
    main()
