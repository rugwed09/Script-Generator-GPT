#Dependencies 
import os
from apkikey import apikey

import streamlit as st
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import wikipedia

os.environ['OPENAI_API_KEY'] = apikey

#App framework 
st.title('🦜🔗Script GPT Creator')
prompt = st.text_input('Plug in your prompt here')

#prompt template
title_template = PromptTemplate(
    input_variable = ['topic'],
    template = 'Write me a youtube video title about {topic}'
)

scipt_template = PromptTemplate(
    input_variable = ['title', 'wikipedia_research'],
    template = 'Write me a youtube video script vased on this title TITLE: {title} While leveraging this wikipedia research:{wikipedia_research}'
)

#Memory

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history') 
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

#llms
llm = openai(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=scipt_template, verbose=True, output_key='script', memory= script_memory)
#sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables= ['title', 'script'], verbose=True)
wiki = wikipedia()

#Display on screen
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wiki_research = wiki_research)
   
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research History'):
        st.info(wiki_research)    
