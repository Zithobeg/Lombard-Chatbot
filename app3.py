import os
from typing import Optional, Tuple
from threading import Lock
import gradio as gr
#from query_data import get_custom_prompt_qa_chain
from query_data import get_qa_with_sources_chain

# Define your company's branding colors
brand_primary_color = "navy"
brand_secondary_color = "lightgreen"

def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = get_qa_with_sources_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain({"question": inp})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

# Define custom CSS for Light Mode
custom_light_css = f"""
.gradio-container {{
    background-color: #F9F9F9;
    color: #000;
    box-shadow: none;
}}

.gradio-title {{
    color: #000;
    font-size: 24px;
    margin-bottom: 10px;
}}

.gradio-input {{
    background-color: #FFF;
    border: 1px solid #D9D9D9;
    padding: 10px;
    font-size: 16px;
    box-shadow: none;
}}

.gradio-button {{
    background-color: {brand_primary_color};
    color: #FFF;
    font-size: 18px;
    border: none;
    padding: 10px 20px;
    margin-top: 10px;
}}

.gradio-output {{
    background-color: #FFF;
    border: 1px solid #D9D9D9;
    padding: 10px;
    font-size: 16px;
}}
"""

# Define custom CSS for Dark Mode
custom_dark_css = f"""
.gradio-container {{
    background-color: #333;
    color: #FFF;
    box-shadow: none;
}}

.gradio-title {{
    color: #FFF;
    font-size: 24px;
    margin-bottom: 10px;
}}

.gradio-input {{
    background-color: #444;
    border: 1px solid #666;
    padding: 10px;
    font-size: 16px;
    box-shadow: none;
}}

.gradio-button {{
    background-color: {brand_secondary_color};
    color: #FFF;
    font-size: 18px;
    border: none;
    padding: 10px 20px;
    margin-top: 10px;
}}

.gradio-output {{
    background-color: #444;
    border: 1px solid #666;
    padding: 10px;
    font-size: 16px;
}}
"""

# Initial state for Dark Mode
#dark_mode_enabled = True

#def toggle_dark_mode(dark_mode_checkbox):
#    global dark_mode_enabled
#    dark_mode_enabled = dark_mode_checkbox


theme = 'gstaff/xkcd'
# Create Gradio interface with Dark Mode toggle
block = gr.Blocks(theme=theme)

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Lombard Insurance Chatbot</center></h3>")

        # Add Dark Mode Toggle Checkbox
#       dark_mode_checkbox = gr.Checkbox(
#            label="Dark Mode",
#            default=False,
 #           action=toggle_dark_mode
 #       )

        openai_api_key_textbox = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=True,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="User Input",
            placeholder="Ask questions about Lombard Insurance",
            lines=3,
        )
        submit = gr.Button(value="Submit")

    gr.Examples(
        examples=[
            "What goes into the Underwriting Process at Lombard Insurance?",
        ],
        inputs=message,
    )

    #gr.HTML("Demo application of a LangChain chain.")

    #gr.HTML(
    #    "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    #)

    state = gr.State()
    agent_state = gr.State()

    def update_chat_history(chat_history, history):
        # Update the chat history display
        chat_history.value = "\n\n".join([f"You: {q}\nBot: {a}" for q, a in history])

    submit.click(chat, inputs=[openai_api_key_textbox, message,
                 state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

#    if dark_mode_enabled:
#        block.css = custom_dark_css

block.launch(debug=True)
