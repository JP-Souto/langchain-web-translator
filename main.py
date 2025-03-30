from transformers import pipeline
from fastapi import FastAPI #to populate input field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # convert responde to string
from langserve import add_routes # synthesize a bridge between langchain and fast api
from langchain_huggingface import HuggingFacePipeline
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("translation_EN_to_YY",
                 model="google-t5/t5-base",
                 device=0,
                 )
print(model("Translate this text from english to deutsch: I like soccer"))
llm = HuggingFacePipeline(pipeline=model)

# prompt template
system_template = "Translate the following into {target_language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# parser
parser = StrOutputParser()

#create chain
chain = prompt_template | llm | parser

#fast api app
app = FastAPI(
    title='langchain web translator',
    version='1.0',
    description='Translate text into another language using langchain and llms',
)

# add a root endpoint
@app.get('/')
async def read_root():
    return {'message':'translator home'}

add_routes(
    app,
    chain,
    path='/chain',
)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=8000)
