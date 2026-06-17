import json
from typing import AsyncIterator, List, Dict, Any

import httpx
from loguru import logger

from .agent_interface import AgentInterface
from ..output_types import SentenceOutput
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource


class LettaAgent(AgentInterface):
    """
    Custom Letta class to interface with the Letta server.
    """

    def __init__(
        self,
        live2d_model,
        id,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        host: str = "localhost",
        port: int = 8283,
    ):
        super().__init__()
        self.url = f"http://{host}:{port}"
        self.id = id
        # Initialize decorator parameters
        self._tts_preprocessor_config = tts_preprocessor_config
        self._live2d_model = live2d_model
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method

        # Delay decorator application
        self.chat = tts_filter(self._tts_preprocessor_config)(
            display_processor()(
                actions_extractor(self._live2d_model)(
                    sentence_divider(
                        faster_first_response=self._faster_first_response,
                        segment_method=self._segment_method,
                        valid_tags=["think"],
                    )(self.chat)
                )
            )
        )

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        # The Letta Server automatically stores historical messages, so this part is not needed
        pass

    def handle_interrupt(self, heard_response: str) -> None:
        pass

    async def chat(self, input_data: BatchInput) -> AsyncIterator[SentenceOutput]:
        messages = self._to_messages(input_data)
        payload = {"messages": messages, "stream_tokens": True}
        path = f"/v1/agents/{self.id}/messages/stream"

        async with httpx.AsyncClient(base_url=self.url, timeout=None) as client:
            async with client.stream("POST", path, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    mtype = event.get("message_type")
                    if mtype == "assistant_message":
                        content = event.get("content", "")
                        if isinstance(content, list):
                            content = "".join(
                                part.get("text", "")
                                for part in content
                                if isinstance(part, dict)
                            )
                        if content:
                            yield content
                    elif mtype == "error_message":
                        logger.error(
                            f"Letta error: {event.get('detail') or event.get('message')}"
                        )
                        break

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """
        Format BatchInput into a prompt string for the LLM.

        Args:
            input_data: BatchInput - The input data containing texts

        Returns:
            str - Formatted message string
        """
        message_parts = []

        # Process text inputs in order
        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(f"[Clipboard content: {text_data.content}]")

        return "\n".join(message_parts)

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """
        Prepare messages list without image support.
        """
        messages = []

        if input_data.images:
            content = []
            text_content = self._to_text_prompt(input_data)
            content.append({"type": "text", "text": text_content})
            user_message = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": self._to_text_prompt(input_data)}

        messages.append(user_message)

        return messages
