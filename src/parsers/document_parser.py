"""Document parsing utilities for RAG benchmarking."""
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib
import xml.etree.ElementTree as ET


@dataclass
class Document:
    """Represents a parsed document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str

    @classmethod
    def from_file(cls, file_path: str, content: str, metadata: Dict[str, Any] = None):
        """Create document from file."""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()
        return cls(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            source=file_path
        )


class DocumentParser:
    """Parse various document formats for RAG ingestion."""

    def __init__(self):
        """Initialize document parser."""
        self.supported_formats = ['.txt', '.md', '.pdf', '.html', '.json', '.xml']

    def parse_txt(self, file_path: str) -> Document:
        """Parse plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {
            'format': 'txt',
            'size_bytes': len(content.encode('utf-8')),
            'filename': Path(file_path).name
        }

        return Document.from_file(file_path, content, metadata)

    def parse_markdown(self, file_path: str) -> Document:
        """Parse markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = {
            'format': 'markdown',
            'size_bytes': len(content.encode('utf-8')),
            'filename': Path(file_path).name
        }

        return Document.from_file(file_path, content, metadata)

    def parse_xml(self, file_path: str) -> Document:
        """
        Parse XML file.

        Supports both MediaWiki XML dumps and generic XML files.
        For MediaWiki dumps, extracts text content from page elements.
        For other XML, extracts all text content.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Check if this is a MediaWiki XML dump
        if 'mediawiki' in root.tag.lower():
            # Extract all page text content from MediaWiki dump
            pages = []

            # Find all page elements (try with and without namespace)
            page_elements = root.findall('.//{http://www.mediawiki.org/xml/export-0.11/}page')
            if not page_elements:
                # Fallback: try without namespace
                page_elements = root.findall('.//page')

            for page in page_elements:
                # Extract title (try with namespace first, then without)
                title_elem = page.find('{http://www.mediawiki.org/xml/export-0.11/}title')
                if title_elem is None:
                    title_elem = page.find('title')

                # Extract text from revision/text (try with namespace first, then without)
                text_elem = page.find('.//{http://www.mediawiki.org/xml/export-0.11/}revision/{http://www.mediawiki.org/xml/export-0.11/}text')
                if text_elem is None:
                    text_elem = page.find('.//revision/text')

                if title_elem is not None and text_elem is not None:
                    title = title_elem.text or ''
                    text = text_elem.text or ''
                    if title and text:
                        pages.append(f"Title: {title}\n\n{text}")

            content = '\n\n---\n\n'.join(pages)
            num_pages = len(pages)
        else:
            # For generic XML, extract all text content
            content = ET.tostring(root, encoding='unicode', method='text')
            num_pages = 1

        metadata = {
            'format': 'xml',
            'size_bytes': Path(file_path).stat().st_size,
            'filename': Path(file_path).name,
            'num_pages': num_pages,
            'xml_type': 'mediawiki' if 'mediawiki' in root.tag.lower() else 'generic'
        }

        return Document.from_file(file_path, content, metadata)

    def parse_pdf(self, file_path: str) -> Document:
        """
        Parse PDF file.

        Note: Requires PyPDF2 or pdfplumber for production use.
        This is a placeholder implementation.
        """
        try:
            import PyPDF2

            content = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content.append(page.extract_text())

            full_content = '\n'.join(content)
            metadata = {
                'format': 'pdf',
                'size_bytes': Path(file_path).stat().st_size,
                'num_pages': len(pdf_reader.pages),
                'filename': Path(file_path).name
            }

            return Document.from_file(file_path, full_content, metadata)

        except ImportError:
            raise ImportError("PyPDF2 required for PDF parsing. Install with: pip install PyPDF2")

    def parse_file(self, file_path: str) -> Document:
        """
        Parse file based on extension.

        Args:
            file_path: Path to file

        Returns:
            Parsed Document object
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.txt':
            return self.parse_txt(file_path)
        elif suffix in ['.md', '.markdown']:
            return self.parse_markdown(file_path)
        elif suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix == '.xml':
            return self.parse_xml(file_path)
        else:
            # Default: treat as text
            return self.parse_txt(file_path)

    def parse_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """
        Parse all supported documents in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively

        Returns:
            List of parsed documents
        """
        documents = []
        path = Path(directory)

        if recursive:
            files = path.rglob('*')
        else:
            files = path.glob('*')

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self.parse_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

        return documents


@dataclass
class IngestionMetrics:
    """Metrics for document ingestion."""
    num_documents: int
    num_chunks: int
    total_parsing_time: float
    total_embedding_time: float
    total_insertion_time: float
    avg_parsing_time_per_doc: float
    avg_embedding_time_per_chunk: float
    avg_insertion_time_per_chunk: float
    total_size_bytes: int
    chunk_sizes: List[int]
    # Optional resource metrics collected during ingestion
    ingestion_resource_metrics: Any = None

    @property
    def parsing_time(self) -> float:
        """Convenience property for backward compatibility."""
        return self.total_parsing_time

    @property
    def embedding_time(self) -> float:
        """Convenience property for backward compatibility."""
        return self.total_embedding_time

    @property
    def insertion_time(self) -> float:
        """Convenience property for backward compatibility."""
        return self.total_insertion_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_documents': self.num_documents,
            'num_chunks': self.num_chunks,
            'total_parsing_time': self.total_parsing_time,
            'total_embedding_time': self.total_embedding_time,
            'total_insertion_time': self.total_insertion_time,
            'avg_parsing_time_per_doc': self.avg_parsing_time_per_doc,
            'avg_embedding_time_per_chunk': self.avg_embedding_time_per_chunk,
            'avg_insertion_time_per_chunk': self.avg_insertion_time_per_chunk,
            'total_size_bytes': self.total_size_bytes,
            'avg_chunk_size': sum(self.chunk_sizes) / len(self.chunk_sizes) if self.chunk_sizes else 0,
            'min_chunk_size': min(self.chunk_sizes) if self.chunk_sizes else 0,
            'max_chunk_size': max(self.chunk_sizes) if self.chunk_sizes else 0,
            'ingestion_resources': (
                self.ingestion_resource_metrics.to_dict() if hasattr(self.ingestion_resource_metrics, 'to_dict') and self.ingestion_resource_metrics is not None else None
            )
        }
