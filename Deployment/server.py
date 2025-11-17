"""
FastAPI-based LLM Inference Server with Streaming Support.

This module provides a production-ready FastAPI server for serving Large Language Models
with streaming text generation capabilities. Supports both quantized and full-precision models.

Author: Ruslan Magana
Website: https://ruslanmv.com
License: Apache-2.0
"""

import argparse
import asyncio
import logging
from queue import Queue
from threading import Thread
from typing import Optional, Dict, Any, AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomStreamer(TextStreamer):
    """
    Custom text streamer that outputs generated tokens to a queue.

    This class extends TextStreamer to enable asynchronous streaming of generated text
    through a queue-based mechanism, suitable for real-time API responses.

    Attributes:
        _queue (Queue): Queue for storing generated text chunks.
        stop_signal (None): Signal to indicate end of generation.
        timeout (int): Timeout for queue operations in seconds.
    """

    def __init__(
        self,
        queue: Queue,
        tokenizer: PreTrainedTokenizer,
        skip_prompt: bool,
        **decode_kwargs: Any
    ) -> None:
        """
        Initialize the custom streamer.

        Args:
            queue: Queue object to store generated text chunks.
            tokenizer: Tokenizer for decoding tokens.
            skip_prompt: Whether to skip the input prompt in the output.
            **decode_kwargs: Additional keyword arguments for token decoding.
        """
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._queue = queue
        self.stop_signal = None
        self.timeout = 1

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """
        Callback method invoked when text generation produces finalized text.

        Args:
            text: The generated text chunk.
            stream_end: Flag indicating if this is the last chunk.
        """
        self._queue.put(text)
        if stream_end:
            self._queue.put(self.stop_signal)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for server configuration.

    Returns:
        Parsed command-line arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="FastAPI LLM Inference Server with Streaming Support"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Hugging Face model ID to use for inference"
    )
    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Enable 4-bit quantization using BitsAndBytes"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation"
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_id: str,
    quantization: bool,
    device: str = "cuda:0"
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the language model and tokenizer.

    Args:
        model_id: Hugging Face model identifier.
        quantization: Whether to load the model with 4-bit quantization.
        device: Target device for model inference.

    Returns:
        Tuple containing the loaded model and tokenizer.

    Raises:
        RuntimeError: If model loading fails.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if quantization:
            logger.info(f"Loading quantized model: {model_id}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map=device,
            )
        else:
            logger.info(f"Loading full-precision model: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map=device,
            )

        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def create_app(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int,
    temperature: float,
    device: str = "cuda:0"
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        model: Pre-trained language model for inference.
        tokenizer: Tokenizer corresponding to the model.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
        device: Device for tensor operations.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="LLM Inference Server",
        description="Production-ready streaming inference server for Large Language Models",
        version="1.0.0"
    )

    streamer_queue = Queue()
    streamer = CustomStreamer(streamer_queue, tokenizer, skip_prompt=True)

    @app.get("/")
    async def root() -> Dict[str, str]:
        """
        Root endpoint providing API information.

        Returns:
            Dictionary containing API metadata.
        """
        return {
            "message": "LLM Inference Server",
            "version": "1.0.0",
            "endpoints": ["/query-stream/"],
        }

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """
        Health check endpoint.

        Returns:
            Dictionary indicating server health status.
        """
        return {"status": "healthy", "model_loaded": model is not None}

    @app.get("/query-stream/")
    async def stream_query(query: str) -> StreamingResponse:
        """
        Stream text generation for a given query.

        This endpoint accepts a text query and returns a streaming response with
        generated text chunks as they are produced by the language model.

        Args:
            query: Input text prompt for the model.

        Returns:
            Streaming response containing generated text.

        Raises:
            HTTPException: If query processing fails.
        """
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Received query: {query[:100]}...")

        try:
            inputs = tokenizer([query], return_tensors="pt").to(device)
            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
            }

            # Start generation in a separate thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            async def response_generator() -> AsyncGenerator[str, None]:
                """Generate response chunks asynchronously."""
                while True:
                    value = streamer_queue.get()
                    if value is None:
                        break
                    yield value
                    streamer_queue.task_done()
                    await asyncio.sleep(0.01)

            return StreamingResponse(
                response_generator(),
                media_type="text/event-stream"
            )

        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return app


def main() -> None:
    """
    Main entry point for the server application.

    Parses command-line arguments, loads the model, and starts the FastAPI server.
    """
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("Starting LLM Inference Server")
    logger.info("=" * 60)
    logger.info(f"Model ID: {args.model_id}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Running on CPU may be extremely slow.")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        args.quantization,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Create FastAPI app
    app = create_app(
        model,
        tokenizer,
        args.max_new_tokens,
        args.temperature
    )

    # Start server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
