import openai
import gradio as gr
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import numpy as np
from transformers import pipeline

# Load environment variables
load_dotenv()

# Initialize OpenAI client and Whisper transcriber
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# Set assistant ID from environment variable
ASSISTANT_ID = os.getenv('ASSISTANT_ID', 'asst_vrzCP6Kjz6lBo1PoHCZ7ghBa')

def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        current_price = info.get('regularMarketPrice', 'N/A')
        previous_close = info.get('regularMarketPreviousClose', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('volume', 'N/A')
        
        # Get historical data for the last 5 days
        hist = stock.history(period='5d')
        
        return f"""
        Stock: {symbol}
        Current Price: ${current_price}
        Previous Close: ${previous_close}
        Market Cap: ${market_cap:,}
        Volume: {volume:,}
        5-day Price History:
        {hist['Close'].to_string()}
        """
    except Exception as e:
        return f"Error fetching data for {symbol}: {str(e)}"

def create_thread_and_run(user_input, thread_id=None):
    if thread_id is None:
        thread = client.beta.threads.create()
    else:
        thread = type('obj', (object,), {'id': thread_id})

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_stock_info",
                "description": "Get real-time stock information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol (e.g., AAPL, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }]
    )

    return thread, run

def get_assistant_response(thread_id, run_id):
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )

    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

        if run.status == "requires_action":
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            
            for tool_call in tool_calls:
                if tool_call.function.name == "get_stock_info":
                    # Parse the arguments as JSON
                    arguments = json.loads(tool_call.function.arguments)
                    symbol = arguments.get("symbol")
                    result = get_stock_info(symbol)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": result
                    })
            
            if tool_outputs:
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run_id,
                    tool_outputs=tool_outputs
                )

    messages = client.beta.threads.messages.list(thread_id=thread_id)
    assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]

    if assistant_messages:
        return assistant_messages[-1].content[0].text.value
    else:
        return "No assistant response found."

def transcribe_audio(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def chat(message, audio, history):
    if audio is not None:
        # Convert audio to text using Whisper
        message = transcribe_audio(audio)
    
    thread_id = None
    
    # Check if message contains stock symbol request
    if "$" in message:
        # Extract stock symbols (words starting with $)
        symbols = [word.strip('$') for word in message.split() if word.startswith('$')]
        stock_data = ""
        for symbol in symbols:
            stock_data += "\n" + get_stock_info(symbol)
        message = message + "\n\nCurrent Stock Data:" + stock_data

    # If there's history, get the thread_id from the last interaction
    if history:
        last_interaction = history[-1]
        if hasattr(last_interaction, 'thread_id'):
            thread_id = last_interaction.thread_id

    # Create thread and run
    thread, run = create_thread_and_run(message, thread_id)
    
    # Get assistant response
    response = get_assistant_response(thread.id, run.id)
    
    # Store thread_id for next interaction
    response = type('obj', (object,), {'thread_id': thread.id, 'text': response})
    
    return response.text

# Create Gradio interface with both text and audio inputs
iface = gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.Audio(label="Speak your message")
    ],
    title="Profit Pilot",
    description="Chat with your AI assistant. Use voice or text input. Use $SYMBOL (e.g., $AAPL) to get real-time stock information.",
    examples=[
        ["How is $AAPL performing today?"],
        ["Compare $GOOGL and $MSFT"]
    ],  # Changed from flat list to nested list
    theme="default"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
