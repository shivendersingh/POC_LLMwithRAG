import asyncio
from pdf_text_processor import create_pdf_processor

async def main():
    # Create PDF processor
    processor, query_engine = create_pdf_processor()

    # Extract text from PDF
    text = await processor._extract_text_from_pdf('E:/POC_LLMwithRAG/DemoData/knowledge.pdf')

    print("=== EXTRACTED TEXT FROM YOUR PDF ===")
    print(text[:1000])
    print("...")
    print(f"Total characters: {len(text)}")

if __name__ == "__main__":
    asyncio.run(main())