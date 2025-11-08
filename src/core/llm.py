"""
LLM Answering Module for Mini RAG System
Retrieves top-k chunks and generates answers with citations
"""

import re
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.core.retrieval import RetrievalSystem

# Load environment variables from .env file
load_dotenv()


class LLMAnswerGenerator:
    """Generates answers using LLM with retrieved context and citations"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 base_url: Optional[str] = None):
        """
        Initialize LLM answer generator
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model name (default: gpt-3.5-turbo)
            base_url: Custom API base URL (for OpenAI-compatible APIs)
        """
        if OpenAI is None:
            raise ImportError(
                "openai is required. Install it with: pip install openai"
            )
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def _create_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Create prompt with context"""
        # format chunks with citation numbers
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source_file = chunk['metadata'].get('source_file', 'Unknown')
            chunk_index = chunk['metadata'].get('chunk_index', i - 1)
            
            context_text += f"\n[Citation {i}] Source: {source_file}, Chunk {chunk_index}\n"
            context_text += f"{chunk['text']}\n"
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
Your answers must be accurate and cite specific evidence from the context using citation numbers.

Context:
{context_text}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents"
3. Cite your sources using [Citation X] format when referencing specific information
4. Be concise but complete
5. If multiple citations support your answer, cite all relevant ones

Answer:"""
        
        return prompt
    
    def generate_answer(self, 
                       query: str, 
                       context_chunks: List[Dict],
                       temperature: float = 0.7,
                       max_tokens: int = 500) -> Dict:
        """Generate answer with citations"""
        if not context_chunks:
            return {
                'answer': "I cannot find relevant information to answer this question.",
                'citations': [],
                'sources': []
            }
        
        prompt = self._create_prompt(query, context_chunks)
        
        try:
            # call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers with citations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # extract citation numbers from answer text
            citations = self._extract_citations(answer)
            
            # get unique source files
            sources = []
            for citation_num in citations:
                if 1 <= citation_num <= len(context_chunks):
                    source = context_chunks[citation_num - 1]['metadata'].get('source_file', 'Unknown')
                    if source not in sources:
                        sources.append(source)
            
            return {
                'answer': answer,
                'citations': citations,
                'sources': sources,
                'context_chunks': [
                    {
                        'text': chunk['text'],
                        'source_file': chunk['metadata'].get('source_file', 'Unknown'),
                        'chunk_index': chunk['metadata'].get('chunk_index', 0),
                        'score': chunk.get('score', 0.0)
                    }
                    for chunk in context_chunks
                ]
            }
        
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
    
    def _extract_citations(self, text: str) -> List[int]:
        """
        Extract citation numbers from answer text
        
        Args:
            text: Answer text that may contain citations like [Citation 1]
            
        Returns:
            List of citation numbers found
        """
        # Pattern to match [Citation X] or [CitationX] or (Citation X)
        patterns = [
            r'\[Citation\s+(\d+)\]',
            r'\[Citation(\d+)\]',
            r'\(Citation\s+(\d+)\)',
            r'Citation\s+(\d+)',
        ]
        
        citations = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                citations.add(int(match))
        
        return sorted(list(citations))


class RAGAnswerSystem:
    """RAG system: retrieval + LLM"""
    
    def __init__(self, 
                 retrieval_system: RetrievalSystem,
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 base_url: Optional[str] = None):
        """Initialize RAG system"""
        self.retrieval_system = retrieval_system
        self.llm_generator = LLMAnswerGenerator(
            api_key=api_key,
            model=model,
            base_url=base_url
        )
    
    def answer(self,
               query: str,
               k: int = 5,
               mode: str = "semantic",
               semantic_weight: float = 0.7,
               bm25_weight: float = 0.3,
               temperature: float = 0.7,
               max_tokens: int = 500) -> Dict:
        """Answer question using RAG"""
        # retrieve chunks
        retrieved_chunks = self.retrieval_system.search(
            query=query,
            k=k,
            mode=mode,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
        
        # generate answer
        result = self.llm_generator.generate_answer(
            query=query,
            context_chunks=retrieved_chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return result


def main():
    """Example usage"""
    import sys
    from src.core.embeddings import EmbeddingVectorStore
    
    if len(sys.argv) < 2:
        print("Usage: python llm_answering.py <question>")
        print("Example: python llm_answering.py 'What is a workflow job?'")
        print("\nNote: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    question = sys.argv[1]
    
    # Load vector store
    try:
        vector_store = EmbeddingVectorStore()
        vector_store.load("faiss_index.bin", "faiss_metadata.pkl")
        
        # Initialize retrieval system
        from src.core.retrieval import RetrievalSystem
        retrieval = RetrievalSystem(vector_store)
        
        # Load chunks
        import pickle
        with open("faiss_metadata.pkl", 'rb') as f:
            chunks = pickle.load(f)
        retrieval.update_chunks(chunks)
        
        # Initialize RAG system
        rag = RAGAnswerSystem(retrieval)
        
        # Generate answer
        print(f"Question: {question}\n")
        print("Retrieving relevant chunks...")
        result = rag.answer(question, k=3)
        
        print("\n" + "=" * 70)
        print("Answer:")
        print("=" * 70)
        print(result['answer'])
        print("\n" + "=" * 70)
        print("Citations:")
        print("=" * 70)
        for citation_num in result['citations']:
            if 1 <= citation_num <= len(result['context_chunks']):
                chunk = result['context_chunks'][citation_num - 1]
                print(f"\n[Citation {citation_num}]")
                print(f"Source: {chunk['source_file']}")
                print(f"Text: {chunk['text'][:100]}...")
        
        print("\n" + "=" * 70)
        print("Sources:")
        print("=" * 70)
        for source in result['sources']:
            print(f"  - {source}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

