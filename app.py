import os
import openai
import base64
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from io import BytesIO
from lida import Manager, TextGenerationConfig, llm


# Define a function that will create an image from base64 string
def base64_to_image(base64_string):
    # Decode the data and returns the image 
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))


# Using the openAI API key
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create an instance of LIDA
lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.1, model="gpt-4", use_cache=True)



# Creation of the streamlit website here
st.subheader("Perform Analysis and Visualisation on your Data")
file_uploader = st.file_uploader(label = "Upload your CSV file")

# Check through all the files uploaded
if file_uploader:
    # Load CSV file into a DataFrame
    df = pd.read_csv(file_uploader)

    path_to_save = "ncs.csv"

    # Get the value within the csv
    with open(path_to_save, "wb") as f:
        f.write(file_uploader.getvalue())

    # Get the input prompt from the user
    user_input = st.text_area("Query your Data to Generate Insights", height = 150)

    # Once user clicks generate
    if st.button("Generate Insights"):
        if len(user_input) > 0:

            # Display the query
            st.info("Your query: " + user_input)

            # Get the summary from the csv file
            summary = lida.summarize(path_to_save, summary_method = "default", textgen_config = textgen_config)

            # Get the goals (i.e the possible questions that users can ask) from the data
            goals = lida.goals(summary = summary, n = 3, textgen_config = textgen_config)

            # Visualise the data
            charts = lida.visualize(summary = summary, goal = user_input, textgen_config = textgen_config, library = "seaborn")
            img = base64_to_image(charts[0].raster)
            st.image(image = img)

            # Performs the explaination
            code = charts[0].code
            explainations = lida.explain(code = code, textgen_config = textgen_config, library = "seaborn")

            # # Convert DataFrame to text
            # text_data = df.to_string(index=False)

            # # Use LLM to generate insights based on user input
            # generated_text = lida.text_gen.generate(text_data + "\nQuery: " + user_input, textgen_config = textgen_config)

            # # Display generated insights
            # st.subheader("Generated Insights")
            # st.write(generated_text)

            # Display the explaination
            st.subheader("Graph Explaination")

            for row in explainations[0]:
                st.write(f"**{row['section']}**:\n{row['explanation']}")

            # Display some of the possible questions
            st.subheader("Some Possible Questions to Consider")

            for goal in goals:
                st.write(goal)


