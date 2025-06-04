import boto3
import json
import ipywidgets as widgets
from IPython.display import display

# Setup Model
my_session = boto3.session.Session()
my_region = my_session.region_name
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=my_region,
    )
def genai(prompt):
    # Define model ID
    model_id = "meta.llama3-8b-instruct-v1:0"

    # Create request body
    body = json.dumps({
        "prompt": prompt,
    })

    # Send request to Bedrock
    response = bedrock.invoke_model(
        modelId=model_id,
        body=body
    )

    # Process the response
    response_body = json.loads(response['body'].read())
    generated_text = response_body['generation']
    generated_text = generated_text.lstrip('?').strip()
    return generated_text

# Create widgets
prompt_input = widgets.Text(description="Prompt:")
submit_button = widgets.Button(description="Submit")
output_area = widgets.Output()

# Define button click event
def on_button_click(b):
    with output_area:
        output_area.clear_output()
        print("Generating response...")
        response = genai(prompt_input.value)
        print("Response:")
        print(response)

# Attach the event to the button
submit_button.on_click(on_button_click)

def display_widgets():
    """
    Function to display the widgets in the notebook.
    Call this function to show the interactive prompt interface.
    """
    display(prompt_input, submit_button, output_area)
