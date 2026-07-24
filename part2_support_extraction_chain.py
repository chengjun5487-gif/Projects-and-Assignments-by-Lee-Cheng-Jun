"""
Instructions: 
1. Define and instantiate all objects and attributes in this .py file.
2. Invocation shoud be separated from this file and tested in A1_P2_testrun notebook (https://colab.research.google.com/drive/1IV-2Y_cgWNpow61HQpMRAJysyXQ6K1sh?usp=sharing).
3. Only single invocation is allowed, make sure all process is encapsulated in the final "full_chain".
4. Remember to includes the "extraction_chain" and all previoes codes and imports from Part 1 in this file, this file should run idependently as a whole.
5. You may build your chain and test the answer in a notebook first, then copy the answer to this .py file after finalizing.
6. You can edit this .py file in VS Code or any Python IDE before upload to the A1_P2_testrun notebook.
7. Any error raised when running this .py file will be subjected to mark reduction.
8. Rename this .py file in format: P2_<FULLNAME>.py, e.g., P2_PNG_WEN_HAO.py
"""

#################  Student Details  #####################
student_name = "LEE CHENG JUN"   # All capital letters
student_id = "2206342"        # Without Alphabets (numeric only)
#################  End of Details  #######################

##################  Import Libraries  ####################
# Make sure you import all necessary libraries (include those from Part 1 Answer)
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
###################  End of Import  ######################

##################  Start of Code  ########################
# Include all your classes, functions, runnables, and variables here
# Include all your answer from Part 1 in this .py file
# Include model instantiation as well
# DO NOT include userdata.get('OPENAI_API_KEY') in this file

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Part 1: Extraction Chain

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

# Part 2: Validation Chain

REQUIRED_FIELDS = ["user_name", "product_name", "model_name", "serial_number", "issue"]

FIELD_LABELS = {
    "user_name":     "User Name",
    "product_name":  "Product Name",
    "model_name":    "Model Name",
    "serial_number": "Serial Number",
    "issue":         "Issue",
}

def get_missing_fields(extracted: dict) -> list:
    return [
        field
        for field in REQUIRED_FIELDS
        if not extracted.get(field, "").strip()
    ]

# auto_feedback: invoked when exactly 1 required field is missing
def auto_feedback(extracted: dict) -> str:
    missing = get_missing_fields(extracted)
    label = FIELD_LABELS[missing[0]]
    return f"Hi, could you please provide me your {label.lower()} before I process your request."

# model_feedback: chain invoked when 2 or more required fields are missing
model_feedback_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ask for missing required info, bold list, offer help"),
        ("human", "{missing_fields_list}"),
    ]
)

model_feedback_chain = model_feedback_prompt | llm | StrOutputParser()

# normal_chain: invoked when all required fields are present
normal_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support specialist. "
            "Start response with 'Hi [name],' followed by a blank line. "
            "Then empathise with the issue and provide helpful next steps. "
            "No formal letter format, no sign-off or signature."
        ),
        (
            "human",
            "Customer Details:\n"
            "- Name: {user_name}\n"
            "- Product: {product_name} {model_name}\n"
            "- Serial Number: {serial_number}\n"
            "- Issue: {issue}\n"
            "- Issue Description: {issue_description}\n"
            "- Inquiry: {inquiry}\n\n"
            "Please provide a comprehensive customer support response."
        ),
    ]
)

normal_chain = normal_response_prompt | llm | StrOutputParser()

def route_validation(extracted: dict) -> str:
    missing = get_missing_fields(extracted)
    count = len(missing)

    if count == 0:
        # All required fields present — generate support response
        return normal_chain.invoke(extracted)
    elif count == 1:
        # Exactly 1 missing field — use hard-coded auto_feedback
        return auto_feedback(extracted)
    else:
        # 2 or more missing fields — use model_feedback chain
        missing_fields_list = "\n".join(
            f"{i + 1}. **{FIELD_LABELS[f]}**:" for i, f in enumerate(missing)
        )
        return model_feedback_chain.invoke({"missing_fields_list": missing_fields_list})

validation_chain = RunnableLambda(route_validation)

full_chain = extraction_chain | validation_chain

###################  End of Code  ########################
