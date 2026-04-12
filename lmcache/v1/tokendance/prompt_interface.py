# SPDX-License-Identifier: Apache-2.0
"""
Round-Aware Prompt Interface for TokenDance.

Provides utilities for multi-agent applications that follow the All-Gather
communication pattern.  The application inserts a reserved separator token
``<TTSEP>`` between adjacent logical blocks so the runtime can recognise
shared blocks even when they appear at different absolute positions across
agent requests.

Typical usage::

    builder = RoundAwarePromptBuilder(tokenizer)
    prompt_tokens = builder.build_round_prompt(
        private_history_text="You are a helpful agent...",
        shared_blocks=["Agent A said ...", "Agent B said ..."],
        round_task_text="Respond to the discussion.",
    )

Reference: TokenDance paper, Section 4.1 — Round-Aware Prompt Interface.
"""

# Standard
from typing import List, Optional, Sequence, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Default separator string.  The tokenizer encodes this into the token IDs
# that act as the ``<TTSEP>`` boundary markers.  Applications may override
# via ``RoundAwarePromptBuilder(sep_str=...)``.
# ---------------------------------------------------------------------------
TTSEP_DEFAULT_STR: str = "<TTSEP>"


class RoundAwarePromptBuilder:
    """Build round-aware prompts with ``<TTSEP>`` separators.

    This helper constructs the token sequence for a single agent's prompt in
    one All-Gather round.  The resulting token stream preserves logical block
    boundaries via separator tokens so that the segment-based hash table in
    :class:`RoundAwareSegmentDatabase` can split, hash, and index each block
    independently of its absolute position.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer (must support
            ``encode(text) -> list[int]``).
        sep_str: The separator string.  Defaults to ``"<TTSEP>"``.
            If the tokenizer has a dedicated ``<TTSEP>`` special token you
            may pass that string directly.
    """

    def __init__(
        self,
        tokenizer: object,
        sep_str: str = TTSEP_DEFAULT_STR,
    ) -> None:
        self.tokenizer = tokenizer
        self.sep_str = sep_str

        # Encode the separator.  Some tokenizers prepend a BOS token — we
        # strip it by taking everything from index 1 when the first token is
        # the BOS.
        raw_ids: list[int] = tokenizer.encode(sep_str)  # type: ignore[union-attr]
        if hasattr(tokenizer, "bos_token_id") and len(raw_ids) > 1:
            if raw_ids[0] == tokenizer.bos_token_id:  # type: ignore[union-attr]
                raw_ids = raw_ids[1:]
        self.sep_tokens: list[int] = raw_ids
        self.sep_tensor: torch.Tensor = torch.tensor(
            self.sep_tokens, dtype=torch.long, device="cpu"
        )
        logger.info(
            "RoundAwarePromptBuilder: sep_str=%r  sep_token_ids=%s",
            sep_str,
            self.sep_tokens,
        )

    # --------------------------------------------------------------------- #
    #  Public helpers                                                         #
    # --------------------------------------------------------------------- #

    def build_round_prompt(
        self,
        private_history_text: str,
        shared_blocks: Sequence[str],
        round_task_text: Optional[str] = None,
    ) -> torch.Tensor:
        """Tokenize and concatenate blocks with ``<TTSEP>`` separators.

        The returned token tensor has the structure::

            [private_history] <TTSEP> [block_0] <TTSEP> [block_1] ...
                              <TTSEP> [round_task]

        Args:
            private_history_text: The agent's private history / system prompt.
            shared_blocks: Ordered list of shared output blocks from the
                previous round.  Each string is tokenized independently.
            round_task_text: Optional trailing task prompt for the current
                round.

        Returns:
            A 1-D ``torch.LongTensor`` on CPU containing the full prompt.
        """
        segments: list[torch.Tensor] = []

        # Private history
        history_ids = self._encode(private_history_text)
        segments.append(history_ids)

        # Shared output blocks
        for block_text in shared_blocks:
            segments.append(self.sep_tensor)
            segments.append(self._encode(block_text))

        # Round task (optional)
        if round_task_text is not None:
            segments.append(self.sep_tensor)
            segments.append(self._encode(round_task_text))

        return torch.cat(segments, dim=0)

    def build_round_prompt_from_token_ids(
        self,
        private_history_ids: Union[List[int], torch.Tensor],
        shared_block_ids: Sequence[Union[List[int], torch.Tensor]],
        round_task_ids: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Concatenate pre-tokenized blocks with ``<TTSEP>`` separators.

        Same layout as :meth:`build_round_prompt` but accepts raw token IDs
        instead of text strings.

        Args:
            private_history_ids: Token IDs for the agent's private history.
            shared_block_ids: Sequence of token-ID sequences for the shared
                output blocks.
            round_task_ids: Optional token IDs for the round task.

        Returns:
            A 1-D ``torch.LongTensor`` on CPU containing the full prompt.
        """
        segments: list[torch.Tensor] = []

        segments.append(self._to_tensor(private_history_ids))

        for block_ids in shared_block_ids:
            segments.append(self.sep_tensor)
            segments.append(self._to_tensor(block_ids))

        if round_task_ids is not None:
            segments.append(self.sep_tensor)
            segments.append(self._to_tensor(round_task_ids))

        return torch.cat(segments, dim=0)

    def get_separator_tokens(self) -> list[int]:
        """Return the token IDs that represent the ``<TTSEP>`` separator.

        Returns:
            A list of integer token IDs.
        """
        return list(self.sep_tokens)

    # --------------------------------------------------------------------- #
    #  Private helpers                                                        #
    # --------------------------------------------------------------------- #

    def _encode(self, text: str) -> torch.Tensor:
        """Encode *text* and return a 1-D LongTensor (stripping leading BOS
        when present)."""
        ids: list[int] = self.tokenizer.encode(text)  # type: ignore[union-attr]
        if hasattr(self.tokenizer, "bos_token_id") and len(ids) > 1:
            if ids[0] == self.tokenizer.bos_token_id:  # type: ignore[union-attr]
                ids = ids[1:]
        return torch.tensor(ids, dtype=torch.long, device="cpu")

    @staticmethod
    def _to_tensor(ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """Ensure *ids* is a 1-D CPU LongTensor."""
        if isinstance(ids, torch.Tensor):
            return ids.to(device="cpu", dtype=torch.long).view(-1)
        return torch.tensor(ids, dtype=torch.long, device="cpu")
