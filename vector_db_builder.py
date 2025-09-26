import chromadb
from chromadb.utils import embedding_functions
import os
import docx
import hashlib
import json
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDBBuilder:
    """Handles building and managing vector database for document retrieval"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "exoplanet_docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.hash_file = "file_hashes.json"
        
    def setup_chromadb(self) -> bool:
        """Initialize ChromaDB client and embedding function"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            logger.info("ChromaDB setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up ChromaDB: {str(e)}")
            return False
    
    def extract_text_from_docx(self, file_path: str) -> Optional[str]:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text.strip())
            
            extracted_text = '\n'.join(text)
            logger.info(f"Successfully extracted {len(extracted_text)} characters from {file_path}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English"""
        if not text:
            return "Unknown"
        
        arabic_chars = sum(1 for char in text[:500] if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in text[:500] if char.isalpha()])
        
        if total_chars == 0:
            return "Unknown"
        
        arabic_ratio = arabic_chars / total_chars
        return "Arabic" if arabic_ratio > 0.3 else "English"
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks with smart boundary detection"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            if end < text_length:
                last_double_newline = text.rfind('\n\n', start, end)
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                if last_double_newline > start + chunk_size // 3:
                    end = last_double_newline + 2
                elif last_period > start + chunk_size // 2:
                    end = last_period + 1
                elif last_newline > start + chunk_size // 2:
                    end = last_newline + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50: 
                chunks.append(chunk)
            
            start = end - overlap if end < text_length else end
        
        return chunks
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""
    
    def load_stored_hashes(self) -> Dict[str, str]:
        """Load previously stored file hashes"""
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading hash file: {str(e)}")
        return {}
    
    def save_file_hashes(self, hashes: Dict[str, str]) -> bool:
        """Save current file hashes"""
        try:
            with open(self.hash_file, 'w', encoding='utf-8') as f:
                json.dump(hashes, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved hashes for {len(hashes)} files")
            return True
        except Exception as e:
            logger.error(f"Error saving hash file: {str(e)}")
            return False
    
    def check_files_changed(self, document_paths: List[str], force_rebuild: bool = False) -> bool:
        """Check if any files have changed since last build"""
        if force_rebuild:
            logger.info("Force rebuild requested")
            return True
        
        stored_hashes = self.load_stored_hashes()
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                logger.warning(f"File not found: {doc_path}")
                continue
                
            current_hash = self.get_file_hash(doc_path)
            if doc_path not in stored_hashes or stored_hashes[doc_path] != current_hash:
                logger.info(f"File changed: {doc_path}")
                return True
        
        logger.info("No file changes detected")
        return False
    
    def delete_existing_collection(self) -> bool:
        """Delete existing collection if it exists"""
        try:
            existing_collections = self.client.list_collections()
            collection_exists = any(col.name == self.collection_name for col in existing_collections)
            
            if collection_exists:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
        
        return True
    
    def create_collection(self) -> bool:
        """Create new ChromaDB collection"""
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            return False
    
    def process_documents(self, document_paths: List[str]) -> tuple[List[str], List[Dict], List[str]]:
        """Process all documents and return chunks, metadata, and IDs"""
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                logger.warning(f"File not found: {doc_path}")
                continue
            
            logger.info(f"Processing {doc_path}...")
            
            text = self.extract_text_from_docx(doc_path)
            if not text:
                logger.warning(f"No text extracted from {doc_path}")
                continue
            
            language = self.detect_language(text)
            
            chunks = self.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from {doc_path}")
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"{os.path.basename(doc_path)}_chunk_{j}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": os.path.basename(doc_path),
                    "chunk_index": j,
                    "language": language,
                    "file_path": doc_path,
                    "chunk_length": len(chunk),
                    "processed_at": datetime.now().isoformat()
                })
                all_ids.append(chunk_id)
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks, all_metadatas, all_ids
    
    def add_to_database(self, chunks: List[str], metadatas: List[Dict], ids: List[str]) -> bool:
        """Add processed chunks to ChromaDB"""
        if not chunks:
            logger.error("No chunks to add to database")
            return False
        
        try:
            batch_size = 100
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch_num = i // batch_size + 1
                logger.info(f"Adding batch {batch_num}/{total_batches} to database...")
                
                batch_chunks = chunks[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Successfully added {len(chunks)} chunks to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            return False
    
    def build_database(self, document_paths: List[str], force_rebuild: bool = False) -> bool:
        """Main method to build the vector database"""
        logger.info("Starting vector database build process...")
        
        if not self.setup_chromadb():
            return False
        
        if not force_rebuild:
            files_changed = self.check_files_changed(document_paths, force_rebuild)
            if not files_changed:
                try:
                    existing_collections = self.client.list_collections()
                    collection_exists = any(col.name == self.collection_name for col in existing_collections)
                    if collection_exists:
                        logger.info("No changes detected and collection exists. Skipping rebuild.")
                        return True
                except:
                    pass
        
        if not self.delete_existing_collection():
            return False
        
        if not self.create_collection():
            return False
        
        chunks, metadatas, ids = self.process_documents(document_paths)
        
        if not chunks:
            logger.error("No valid documents processed")
            return False
        
        if not self.add_to_database(chunks, metadatas, ids):
            return False
        
        current_hashes = {}
        for doc_path in document_paths:
            if os.path.exists(doc_path):
                current_hashes[doc_path] = self.get_file_hash(doc_path)
        
        if not self.save_file_hashes(current_hashes):
            logger.warning("Failed to save file hashes")
        
        logger.info("Vector database build completed successfully!")
        return True
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database"""
        try:
            if not self.client:
                self.setup_chromadb()
            
            collections = self.client.list_collections()
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                collection = self.client.get_collection(self.collection_name)
                count = collection.count()
                
                return {
                    "exists": True,
                    "collection_name": self.collection_name,
                    "document_count": count,
                    "db_path": self.db_path
                }
            else:
                return {
                    "exists": False,
                    "collection_name": self.collection_name,
                    "db_path": self.db_path
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            return {"exists": False, "error": str(e)}

def find_document_files(directory: str = ".") -> List[str]:
    """Find all .docx files in the given directory"""
    doc_files = []
    for file in os.listdir(directory):
        if file.endswith('.docx') and not file.startswith('~'):  
            doc_files.append(os.path.join(directory, file))
    return doc_files

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Build vector database for exoplanet project documents")
    parser.add_argument("--docs", nargs="+", help="Paths to document files")
    parser.add_argument("--directory", "-d", default=".", help="Directory to search for .docx files")
    parser.add_argument("--force", "-f", action="store_true", help="Force rebuild even if no changes detected")
    parser.add_argument("--db-path", default="./chroma_db", help="Path for ChromaDB storage")
    parser.add_argument("--collection", default="exoplanet_docs", help="Collection name")
    parser.add_argument("--info", action="store_true", help="Show database info only")
    
    args = parser.parse_args()
    
    builder = VectorDBBuilder(db_path=args.db_path, collection_name=args.collection)
    
    if args.info:
        info = builder.get_database_info()
        print("Database Information:")
        print(f"Exists: {info['exists']}")
        print(f"Collection: {info['collection_name']}")
        print(f"Path: {info['db_path']}")
        if info['exists']:
            print(f"Document Count: {info['document_count']}")
        return
    
    if args.docs:
        document_paths = args.docs
    else:
        document_paths = find_document_files(args.directory)
        if not document_paths:
            logger.error(f"No .docx files found in {args.directory}")
            return
    
    logger.info(f"Found {len(document_paths)} document(s) to process:")
    for doc in document_paths:
        logger.info(f"  - {doc}")
    
    success = builder.build_database(document_paths, force_rebuild=args.force)
    
    if success:
        logger.info("✅ Database build completed successfully!")
        info = builder.get_database_info()
        logger.info(f"📊 Total chunks in database: {info.get('document_count', 'Unknown')}")
    else:
        logger.error("❌ Database build failed!")

if __name__ == "__main__":
    main()