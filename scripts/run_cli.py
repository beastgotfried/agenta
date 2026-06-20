import asyncio 
from app.agent.models import make_model

async def main():
    model=make_model()
    response = await model.ainvoke("say hello in a short sentence")
    print(response.content)
    
if __name__ == "__main__":
    asyncio.run(main())
    
    