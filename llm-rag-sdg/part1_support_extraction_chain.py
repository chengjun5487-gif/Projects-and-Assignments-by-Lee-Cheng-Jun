"""
Instructions
1. Define and instantiate all objects and attributes in this .py file.
2. Invocation shoud be separated from this file and tested in A1_P1_testrun notebook (https://colab.research.google.com/drive/1cBB2QLSf5x7KprbVdntsYRhwSCNb-G7e?usp=drive_link).
3. Only single invocation is allowed, make sure all process is encapsulated in the final "extraction_chain".
4. You may build your chain and test the answer in a notebook first, then copy the answer to this .py file after finalizing.
6. You can edit this .py file in VS Code or any Python IDE before upload to the A1_P1_testrun notebook.
6. Any error raised when running this .py file will be subjected to mark reduction.
7. Rename this .py file in format: P1_<FULLNAME>.py, e.g., P1_PNG_WEN_HAO.py
"""

#################  Student Details  #####################
student_name = "LEE CHENG JUN"   # All capital letters
student_id = "2206342"        # Without Alphabets (numeric only)
#################  End of Details  #######################

##################  Import Libraries  ####################
# Make sure you import all necessary libraries 
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
###################  End of Import  ######################

##################  Start of Code  #######################
# Include all your classes, functions, runnables, and variables here
# Include model instantiation as well
# DO NOT include userdata.get('OPENAI_API_KEY') in this file

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

extraction_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract support info. Return JSON: "
            "user_name, "
            "product_name (brand only e.g. MacBook not MacBook Pro), "
            "model_name (variant, e.g. Pro 16-inch, Air M4), "
            "serial_number (exact only; ignore if unsure), "
            "issue (fault ≤5 words, no qualifiers), "
            "issue_description (user words), "
            "inquiry (explicit question/request). "
            "Empty if missing. No inference."
        ),
        ("human", "{query}"),
    ]
)

EMPTY_RESULT = {
    "user_name": "", "product_name": "", "model_name": "",
    "serial_number": "", "issue": "", "issue_description": "", "inquiry": ""
}

def parse_output(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not text:
        return EMPTY_RESULT
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return EMPTY_RESULT

def normalise_input(input):
    if isinstance(input, str):
        return {"query": input}
    return input

extraction_chain = (
    RunnableLambda(normalise_input)
    | extraction_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(parse_output)
)

###################  End of Code  ########################
