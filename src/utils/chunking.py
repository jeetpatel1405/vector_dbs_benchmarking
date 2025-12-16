"""Text chunking strategies for RAG systems."""
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    text: str
    metadata: Dict[str, Any]
    start_index: int
    end_index: int

    def __len__(self):
        """Return chunk length."""
        return len(self.text)


class ChunkingStrategy:
    """Base class for chunking strategies."""

    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        """
        Initialize chunking strategy.

        Args:
            chunk_size: Target size of each chunk (in characters or tokens)
            chunk_overlap: Number of characters/tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: Text to chunk
            doc_id: Document ID for metadata

        Returns:
            List of chunks
        """
        raise NotImplementedError


class FixedSizeChunking(ChunkingStrategy):
    """Fixed-size chunking with optional overlap."""

    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """
        Chunk text into fixed-size pieces.

        Args:
            text: Text to chunk
            doc_id: Document ID for metadata

        Returns:
            List of chunks
        """
        chunks = []
        text_length = len(text)
        start = 0
        chunk_num = 0

        # Minimum chunk size to avoid tiny final chunks (typically overlap size or 10% of chunk_size)
        min_chunk_size = max(self.chunk_overlap, self.chunk_size // 10)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_length = end - start

            # Skip very small final chunks - merge with previous chunk instead
            if chunk_length < min_chunk_size and len(chunks) > 0:
                break

            chunk_text = text[start:end]
            chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"

            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata={
                    'chunk_num': chunk_num,
                    'doc_id': doc_id,
                    'strategy': 'fixed_size',
                    'chunk_size': self.chunk_size,
                    'overlap': self.chunk_overlap
                },
                start_index=start,
                end_index=end
            )

            chunks.append(chunk)
            chunk_num += 1

            # Move start position, accounting for overlap
            start = start + self.chunk_size - self.chunk_overlap

        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-aware chunking that tries to keep sentences intact."""

    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        """Initialize sentence chunking."""
        super().__init__(chunk_size, chunk_overlap)
        # Simple sentence splitter (can be improved with NLTK or spaCy)
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """
        Chunk text by sentences, respecting chunk size.

        Args:
            text: Text to chunk
            doc_id: Document ID for metadata

        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = self.sentence_pattern.split(text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 0
        start_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds chunk size and we have content
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"

                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        'chunk_num': chunk_num,
                        'doc_id': doc_id,
                        'strategy': 'sentence',
                        'target_chunk_size': self.chunk_size,
                        'num_sentences': len(current_chunk)
                    },
                    start_index=start_index,
                    end_index=start_index + len(chunk_text)
                )

                chunks.append(chunk)
                chunk_num += 1

                # Handle overlap
                if self.chunk_overlap > 0:
                    # Keep some sentences for overlap
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text, sentence]
                    start_index = start_index + len(chunk_text) - self.chunk_overlap
                    current_size = len(overlap_text) + sentence_size
                else:
                    current_chunk = [sentence]
                    start_index += len(chunk_text)
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"

            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata={
                    'chunk_num': chunk_num,
                    'doc_id': doc_id,
                    'strategy': 'sentence',
                    'target_chunk_size': self.chunk_size,
                    'num_sentences': len(current_chunk)
                },
                start_index=start_index,
                end_index=start_index + len(chunk_text)
            )

            chunks.append(chunk)

        return chunks


class ParagraphChunking(ChunkingStrategy):
    """Paragraph-aware chunking."""

    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """
        Chunk text by paragraphs, respecting chunk size.

        Args:
            text: Text to chunk
            doc_id: Document ID for metadata

        Returns:
            List of chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 0
        start_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_size = len(paragraph)

            # If adding this paragraph exceeds chunk size and we have content
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"

                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        'chunk_num': chunk_num,
                        'doc_id': doc_id,
                        'strategy': 'paragraph',
                        'target_chunk_size': self.chunk_size,
                        'num_paragraphs': len(current_chunk)
                    },
                    start_index=start_index,
                    end_index=start_index + len(chunk_text)
                )

                chunks.append(chunk)
                chunk_num += 1

                # Reset for next chunk
                current_chunk = [paragraph]
                start_index += len(chunk_text)
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_num}" if doc_id else f"chunk_{chunk_num}"

            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata={
                    'chunk_num': chunk_num,
                    'doc_id': doc_id,
                    'strategy': 'paragraph',
                    'target_chunk_size': self.chunk_size,
                    'num_paragraphs': len(current_chunk)
                },
                start_index=start_index,
                end_index=start_index + len(chunk_text)
            )

            chunks.append(chunk)

        return chunks


class SemanticChunking(ChunkingStrategy):
    """
    Semantic chunking using embeddings (placeholder).

    This would require embeddings to group semantically similar content.
    For now, falls back to sentence chunking.
    """

    def chunk(self, text: str, doc_id: str = None) -> List[Chunk]:
        """
        Chunk text semantically (placeholder implementation).

        Args:
            text: Text to chunk
            doc_id: Document ID for metadata

        Returns:
            List of chunks
        """
        # For now, use sentence chunking
        # In production, this would use embeddings to group similar sentences
        sentence_chunker = SentenceChunking(self.chunk_size, self.chunk_overlap)
        chunks = sentence_chunker.chunk(text, doc_id)

        # Update metadata
        for chunk in chunks:
            chunk.metadata['strategy'] = 'semantic'

        return chunks


def get_chunking_strategy(strategy_name: str, chunk_size: int, chunk_overlap: int = 0) -> ChunkingStrategy:
    """
    Factory function to get chunking strategy.

    Args:
        strategy_name: Name of strategy ('fixed', 'sentence', 'paragraph', 'semantic')
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap size

    Returns:
        ChunkingStrategy instance
    """
    strategies = {
        'fixed': FixedSizeChunking,
        'sentence': SentenceChunking,
        'paragraph': ParagraphChunking,
        'semantic': SemanticChunking
    }

    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")

    return strategy_class(chunk_size, chunk_overlap)
