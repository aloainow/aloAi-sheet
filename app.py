import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.llms import GooglePalm, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_pandas_dataframe_agent
import streamlit as st
from streamlit_chat import message
import statsmodels as sm
import seaborn as sns
import os
import sys
from io import StringIO, BytesIO
from sklearn.linear_model import LinearRegression


#api_key1 = st.secrets["GOOGLE_API_KEY"]
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] ="AIzaSyD29fEos3V6S2L-AGSQgNu03GqZEIgJads"
os.environ["OPENAI_API_KEY"] = api_key
llm1 = OpenAI(temperature=temperature,max_tokens= 2048)  
llm2 = GooglePalm(temperature=0.1, max_output_tokens= 2048,verbose=True,streaming=True)





st.set_page_config(page_title="aloAi", page_icon="chart_with_upwards_trend")

st.image("white_logo.png", width= 250)


about = st.sidebar.expander("🧠 About")
sections = [r"""
Encontre e compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial .
Faça análises de times e jogadores, identificando oportunidades de mercado insights para o seu time.
As possibilidades são infinitas." 
    """]
for section in sections:
    about.write(section)
    

TEMPERATURE_MIN_VALUE = 0.0
TEMPERATURE_MAX_VALUE = 1.0
TEMPERATURE_DEFAULT_VALUE = 0.5
TEMPERATURE_STEP = 0.01
with st.sidebar.expander("🛠️Tools", expanded=False):
    temperature = st.slider(
        label="Temperature",
        min_value=TEMPERATURE_MIN_VALUE,
        max_value=TEMPERATURE_MAX_VALUE,
        value=TEMPERATURE_DEFAULT_VALUE,
        step=TEMPERATURE_STEP,
    )
    st.session_state["temperature"] = temperature





def generate_code(prompt, data_type, missing, shape):
    

    prompt_template = PromptTemplate(
    input_variables=['prompt','data_type', 'shape', 'missing'],
        template="Your a football data analyst who understands protuguese. Football Data is loaded as 'df', column names and their types: {data_type}\n\
        df.shape: {shape}\
        missing values: {missing}\
        instructions: Please provide short python code for the user, user knows python, include correct column names.\
        query: {prompt}\
        Answer: \
        " 
    )
    about_chain = LLMChain(llm=llm1, prompt=prompt_template, output_key="about")


    chain = SequentialChain(chains=[about_chain], input_variables=["prompt","data_type", "shape", "missing"], output_variables=["about"])

    response = chain.run({'prompt': prompt, 'data_type': data_type, 'shape': shape, 'missing':missing})
    return response
    

  
folder_path = "./files"

# Initialize an empty list to store the file paths
file_paths = []

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    # Create the full file path by joining the folder path and the filename
    file_path = os.path.join(folder_path, filename)
    # Append the file path to the list
    file_paths.append(file_path)


folder_path = "./files/"

# List all files in the folder
files_in_folder = os.listdir(folder_path)

# Check if there are any files in the folder
if files_in_folder:
    # Select the first file in the list
    selected_file = files_in_folder[0]

    # Construct the full path to the selected file
    uploaded_file = os.path.join(folder_path, selected_file)

    # Now 'uploaded_file' contains the path to the selected file
    print(f"Selected file: {uploaded_file}")
else:
    print("No files found in the folder.")
    


if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'responses' not in st.session_state:
    st.session_state.responses = []



file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data()
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

if "messages" not in st.session_state or st.sidebar.button("Limpar histórico de conversa"):
    st.session_state["messages"] = [{"role": "assistant", "content": r""" Olá, através de prompts, encontre e compare jogadores, através da combinação entre estatísticas e todo o poder da Inteligência artificial .
Faça análises de times e jogadores, identificando oportunidades de mercado insights para o seu time.
As possibilidades são infinitas.
"""}]
    st.session_state['history']  = []

if uploaded_file is not None:
    df = load_data(uploaded_file)

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.chat_message("assistant", avatar= "https://raw.githubusercontent.com/aloainow/images/main/logo.png").write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])


    if prompt := st.chat_input(placeholder="Inicie aqui seu chat!"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        
        data = df.head()
        missing = df.isnull().sum()
        shape = df.shape
        columns= df.columns

        variable_type_info = []

        for key, value in data.items():
            variable_type = type(value)
            data_type1 = f"'{key}' is of type: {variable_type}"
            variable_type_info.append(data_type1)
        data_type = "\n".join(variable_type_info)
        
        prompt1 =  generate_code(prompt, missing, shape, columns) 
      
        
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "", ai_prefix= "")

        for i in range(0, len(st.session_state.messages), 2):
            if i + 1 < len(st.session_state.messages):
                current_message = st.session_state.messages[i]
                next_message = st.session_state.messages[i + 1]
                
                current_role = current_message["role"]
                current_content = current_message["content"]
                
                next_role = next_message["role"]
                next_content = next_message["content"]
                
                # Concatenate role and content for context and output
                context = f"{current_role}: {current_content}\n{next_role}-said: {next_content}"
                
                memory.save_context({"question": context}, {"output": ""})
        
        #llm1= ChatOpenAI(temperature=0.7,  model="gpt-3.5-turbo-0613", streaming=True, verbose = True) #incase we need openai
        
        #agent = create_pandas_dataframe_agent(llm1 ,df, agent_type=AgentType.OPENAI_FUNCTIONS
        agent = create_pandas_dataframe_agent(llm2 ,df , agent = AgentType.OPENAI_FUNCTIONS
                                              ,prefix=r"""You are an expert football data analyst. You need to perform analysis on players' data.
                                                                                            
"""
        ,handle_parsing_errors=True, number_of_head_rows= 2
        )

        
        with st.chat_message("assistant", avatar= "https://raw.githubusercontent.com/aloainow/images/main/logo.png"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            try:
                # Your code that may raise an error here
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                response = agent.run(prompt1, callbacks=[st_cb])
                fig = plt.gcf()
                if fig.get_axes():
                            # Adjust the figure size
                    fig.set_size_inches(12, 6)

                    # Adjust the layout tightness
                    plt.tight_layout()
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    
                    #message.write("Hello human")
                    st.image(buf, caption={prompt},use_column_width=True)
                    
                                        
                    st.session_state.messages.append({"role": "assistant", "content": f"Image Removed -{prompt}"})
                    st.stop()  

                sys.stdout = old_stdout
            
            except Exception as e:
                # Handle the error here
                st.error("Problem in the Data! Please Try Again with a different question.")
                st.stop()  # Stop execution to prevent further code execution
            st.session_state.messages.append({"role": "assistant", "content": response})     
            st.markdown(response)

       
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
            




