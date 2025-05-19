import gemini_helper.build as builder
import google.generativeai as genai
import ipywidgets as widgets

_model = None # Private module-level variable for the model

def _initialize_model():
    """Initializes the Gemini model if not already initialized."""
    global _model
    if _model is None:
        try:
            key = builder.get_api_key()
            genai.configure(api_key=key)
            _model = genai.GenerativeModel("gemini-1.5-flash")
            print("Gemini model initialized successfully.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            _model = None # Ensure it's None if initialization fails
    return _model

# --- Widget Creation and Logic ---
def create_gemini_chat_widget():
    """
    Creates and returns the Ipywidgets for a Gemini chat interface.
    The widgets should be displayed in the calling environment (e.g., Jupyter Notebook).
    """
    model = _initialize_model()
    if not model:
        # If model failed to initialize, return a message widget
        error_message = widgets.HTML("Failed to initialize Gemini model. Please check API key and configuration.")
        return (error_message,) # Return as a tuple for consistency with display

    # Create a text box for input
    prompt_input = widgets.Text(
        value='',
        placeholder='Type your prompt here',
        description='Prompt:',
        disabled=False
    )

    # Create an output area for the response
    response_output = widgets.Output()

    # Create a button to submit the prompt
    submit_button = widgets.Button(description="Submit")

    # Define the function to handle the button click
    def on_submit_button_clicked(b):
        # Ensure the model is available (it should be if we reached here)
        if not _model:
            with response_output:
                response_output.clear_output()
                print("Model is not available.")
            return

        with response_output:
            response_output.clear_output() # Clear previous output
            prompt_text = prompt_input.value
            if not prompt_text.strip():
                print("Please enter a prompt.")
                return

            # Indicate processing
            print("Generating response...")
            try:
                # Generate the response using the Gemini model
                response = _model.generate_content(prompt_text)
                response_output.clear_output() # Clear "Generating response..."
                print(response.text)
            except Exception as e:
                response_output.clear_output() # Clear "Generating response..."
                print(f"Error generating content: {e}")

    # Attach the click event handler to the button
    submit_button.on_click(on_submit_button_clicked)

    # Return the widgets to be displayed by the caller
    return prompt_input, submit_button, response_output

def run_gemini_widget():
    """
    An example "empty" function to create and immediately display the widget.
    """
    from IPython.display import display # Import display here, as it's for notebook environment

    widgets_to_display = create_gemini_chat_widget()
    if widgets_to_display: # Check if widgets were created
        # If only one widget is returned (e.g., an error message), display it
        if isinstance(widgets_to_display, tuple) and len(widgets_to_display) > 1:
            display(*widgets_to_display)
        else:
            display(widgets_to_display)
    else:
        print("Could not create Gemini widget.")