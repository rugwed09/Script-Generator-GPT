# Dependencies
import os
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Set your OpenAI API Key
apikey = "ENTER YOUR API KEY"  # Replace with your actual API key
os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 Script GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write me a YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title'],
    template='Write me a YouTube video script based on this title TITLE: {title}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# Display on screen
if prompt:
    title = title_chain.run(prompt)
    script = script_chain.run(title=title)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)
