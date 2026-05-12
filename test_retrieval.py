import asyncio
from src.retrieval.retrieval_service import retrieve_chunks

async def test():
    result = await retrieve_chunks('severe malaria')
    print(f'Retrieved {len(result)} chunks')
    print('Top sources:')
    for i, r in enumerate(result[:3]):
        print(f'{i+1}. {r["source"]}')

if __name__ == '__main__':
    asyncio.run(test())