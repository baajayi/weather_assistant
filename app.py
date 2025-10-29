import requests
import os
from datetime import datetime, timezone
import pytz
import traceback
import time
import json
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
from flask_cors import CORS
import logging
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from langfuse import observe, get_client

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse client for trace updates
langfuse = get_client()

# Get the OpenWeatherMap API key from environment variables
api_key = os.getenv('WEATHER_API_KEY')
def get_weather_by_city(city_name, api_key, country_code=None, state_code=None, exclude=None):
    """
    Fetches weather data for a city by name using OpenWeatherMap's Geocoding and One Call APIs.

    Parameters:
    city_name (str): Name of the city (e.g., "London").
    api_key (str): OpenWeatherMap API key.
    country_code (str, optional): Country code (e.g., "GB" for United Kingdom).
    state_code (str, optional): State code (e.g., "CA" for California).
    exclude (list or str, optional): Parts to exclude from One Call API response.

    Returns:
    dict: Weather data from One Call API, or None if an error occurs.
    """
    # Step 1: Get coordinates using Geocoding API
    geocoding_url = "https://api.openweathermap.org/geo/1.0/direct?"
    q = city_name
    if state_code:
        q += f",{state_code}"
    if country_code:
        q += f",{country_code}"
    
    params = {
        'q': q,
        'limit': 1,
        'appid': os.getenv('WEATHER_API_KEY'),
    }
    
    try:
        # Fetch coordinates
        geo_response = requests.get(geocoding_url, params=params)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            print(f"City '{city_name}' not found.")
            return None
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        # Step 2: Fetch weather data using One Call API
        return get_openweather_onecall(lat, lon, api_key, exclude)
    
    except requests.exceptions.RequestException as e:
        print(f"Geocoding API error: {e}")
        return None


# Reuse the existing One Call API function
def get_openweather_onecall(lat, lon, api_key, exclude=None):
    """
    (The original One Call API function from earlier)
    """
    url = os.getenv('OPENWEATHER_ONECALL_API_URL', 'https://api.openweathermap.org/data/2.5/weather')
    params = {'lat': lat, 'lon': lon, 'appid': os.getenv('WEATHER_API_KEY')}
    
    if exclude:
        params['exclude'] = exclude if isinstance(exclude, str) else ','.join(exclude)
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"One Call API error: {e}")
        return None
    


def format_weather_response(raw_text: str) -> str:
    """Convert API response to formatted markdown"""
    # Add your custom formatting logic here
    formatted = (
        raw_text
        .replace('**', '*')  # Convert to single asterisks for italic
        .replace('Fahrenheit', '°F')
        .replace('Celsius', '°C')
        .replace(' - ', '\n- ')
    )
    
    # Add table formatting for historical data
    if 'historical data' in formatted.lower():
        formatted = formatted.replace('|', '\n|').replace('---', '---')
    
    return formatted


def get_historical_weather(lat: float, lon: float, api_key: str, 
                          start: int, cnt: int, data_type: str = 'hour') -> dict:
    """
    Fetches historical weather data from OpenWeatherMap's History API.
    
    Parameters:
    lat (float): Latitude of the location
    lon (float): Longitude of the location
    api_key (str): OpenWeatherMap API key with History API access
    start (int): Start time in UNIX timestamp (UTC)
    cnt (int): Number of data points to retrieve (max 24 for hourly, 30 for daily)
    data_type (str): Type of data - 'hour' or 'day' (default: 'hour')

    Returns:
    dict: JSON response or None if error occurs
    """
    url = "https://api.openweathermap.org/data/2.5/history/city"
    
    params = {
        'lat': lat,
        'lon': lon,
        'type': data_type,
        'start': start,
        'cnt': cnt,
        'appid': api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    return None

# Helper function to convert datetime to UNIX timestamp
def datetime_to_utc_timestamp(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

system_prompt = """
You are a weather assistant that retrieves data from OpenWeatherMap.
You can respond to questions about coordinates, weather conditions, and historical weather data.
Only respond to weather-related questions and address exactly what was asked - nothing more.
When asked about temperature, provide only temperature information.
When asked about precipitation, provide only precipitation information.
When asked about wind, provide only wind information.
And so on for other weather conditions.
Use the available API tools to fetch accurate and current weather data.
If asked about non-weather topics, politely explain that you can only provide weather information."""


# Initialize OpenAI client with Langfuse tracing
client = OpenAI()

weather_assistant = client.beta.assistants.create(
    instructions=system_prompt,
    name="Weather Assistant",
    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_city",
            "description": "Fetches current weather data for a city using OpenWeatherMap Geocoding and Weather API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "Name of the city (e.g., 'London')"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "OpenWeatherMap API key"
                    },
                    "country_code": {
                        "type": "string",
                        "description": "Country code (e.g., 'GB' for United Kingdom)",
                        "nullable": True
                    },
                    "state_code": {
                        "type": "string",
                        "description": "State code (e.g., 'CA' for California)",
                        "nullable": True
                    },
                    "exclude": {
                        "type": ["string", "array"],
                        "description": "Parts to exclude from the weather data (e.g., 'minutely,hourly')",
                        "items": {
                            "type": "string"
                        },
                        "nullable": True
                    }
                },
                "required": ["city_name", "api_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_openweather_onecall",
            "description": "Fetches weather data from OpenWeatherMap One Call API by coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude"
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "OpenWeatherMap API key"
                    },
                    "exclude": {
                        "type": ["string", "array"],
                        "description": "Data parts to exclude (e.g., 'minutely,hourly')",
                        "items": {
                            "type": "string"
                        },
                        "nullable": True
                    }
                },
                "required": ["lat", "lon", "api_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_weather",
            "description": "Fetches historical weather data from OpenWeatherMap History API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude"
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "OpenWeatherMap API key with History access"
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start time in UNIX timestamp (UTC)"
                    },
                    "cnt": {
                        "type": "integer",
                        "description": "Number of data points to retrieve"
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["hour", "day"],
                        "description": "Type of data to retrieve (hour or day)"
                    }
                },
                "required": ["lat", "lon", "api_key", "start", "cnt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "datetime_to_utc_timestamp",
            "description": "Converts a datetime string to a UTC UNIX timestamp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dt": {
                        "type": "string",
                        "description": "Datetime string in ISO 8601 format (e.g., '2023-04-08T14:30:00')"
                    }
                },
                "required": ["dt"]
            }
        }
    }
],
    model="gpt-4o-mini"
)

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='None',
    
)
Session(app)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
@app.route('/ask', methods=['POST'])
@observe()
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Log user input to Langfuse
    langfuse.update_current_trace(
        name="weather_assistant_conversation",
        input={"question": question},
        metadata={"endpoint": "/ask"}
    )

    # Create a new thread for the conversation
    if "thread_id" not in session:
        thread = client.beta.threads.create()
        session["thread_id"] = thread.id
        session.modified = True  # Explicitly mark session as modified
        print("New thread created:", thread.id)
    thread_id = session["thread_id"]
    print("Using thread:", thread_id)

    # Add thread_id to trace metadata
    langfuse.update_current_trace(
        session_id=thread_id,
        user_id=session.get("user_id", "anonymous")
    )

    # Add user message to the existing thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=question
    )
    # Start a new run and block until complete
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=weather_assistant.id
    )
    while True:
        if run.status == "requires_action":
            print(f"Tool calls required: {run.required_action.submit_tool_outputs.tool_calls}")
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            
            for tool_call in tool_calls:
                print(f"Processing tool call: {tool_call.id}")
                print(f"- Function: {tool_call.function.name}")
                print(f"- Arguments: {tool_call.function.arguments}")
                
                result = get_outputs_for_tools(tool_call)
                
                # Validate output format
                if "output" not in result and "error" not in result:
                    result["error"] = "Invalid tool response format"
                    
                # Force all outputs to be JSON strings
                if "output" in result:
                    try:
                        json.loads(result["output"])
                    except json.JSONDecodeError:
                        result["output"] = json.dumps({"result": result["output"]})
                        
                tool_outputs.append(result)
                print(f"Tool output: {json.dumps(result, indent=2)}")
                
            print(f"Submitting {len(tool_outputs)} tool outputs")
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=[
                    {
                        "tool_call_id": output["tool_call_id"],
                        "output": output.get("output") or json.dumps({"error": output.get("error")})
                    }
                    for output in tool_outputs
                ]
            )
        
        # Check if run is complete
        if run.status in ["completed", "failed", "expired"]:
            print(f"Run status: {run.status}")
            break
            
        # If we're here, the run is still in progress
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        print(f"Run status: {run.status}")


    final_message = format_weather_response(client.beta.threads.messages.list(thread_id=thread_id).data[0].content[0].text.value)
    print(f"Final message: {final_message}")

    # Update trace with the final response
    langfuse.update_current_trace(
        output={"response": final_message},
        metadata={
            "thread_id": thread_id,
            "run_id": run.id,
            "status": run.status
        }
    )

    return jsonify({
        "thread_id": thread_id,
        "run_id": run.id,
        "status": run.status,
        "response": final_message
    })

@observe()
def get_outputs_for_tools(tool_call):
    """
    Execute the requested weather-related tool call and return the result in the expected format.

    Args:
        tool_call: The tool call object from the LLM containing function name and arguments

    Returns:
        dict: A dictionary with tool_call_id and either output (success) or error (failure)
    """
    try:
        tool_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            error_result = {
                "tool_call_id": tool_call.id,
                "error": f"Invalid JSON in arguments: {str(e)}"
            }
            langfuse.update_current_span(
                name=f"tool_execution_{tool_name}",
                input={"tool_call_id": tool_call.id, "raw_arguments": tool_call.function.arguments},
                output=error_result,
                level="ERROR"
            )
            return error_result

        print(f"Executing tool: {tool_name}")
        print(f"With arguments: {arguments}")

        # Log tool execution start
        langfuse.update_current_span(
            name=f"tool_execution_{tool_name}",
            input={
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call.id
            }
        )

        # Tool function mapping
        tool_dispatcher = {
            "get_weather_by_city": lambda args: get_weather_by_city(
                city_name=args.get("city_name"),
                api_key=args.get("api_key"),
                country_code=args.get("country_code"),
                state_code=args.get("state_code"),
                exclude=args.get("exclude")
            ),
            "get_openweather_onecall": lambda args: get_openweather_onecall(
                lat=args.get("lat"),
                lon=args.get("lon"),
                api_key=args.get("api_key"),
                exclude=args.get("exclude")
            ),
            "get_historical_weather": lambda args: get_historical_weather(
                lat=args.get("lat"),
                lon=args.get("lon"),
                api_key=args.get("api_key"),
                start=args.get("start"),
                cnt=args.get("cnt"),
                data_type=args.get("data_type", "hour")
            ),
            "datetime_to_utc_timestamp": lambda args: datetime_to_utc_timestamp(
                datetime.fromisoformat(args.get("dt"))
            )
        }

        if tool_name not in tool_dispatcher:
            error_result = {
                "tool_call_id": tool_call.id,
                "error": f"Unknown tool: {tool_name}"
            }
            langfuse.update_current_span(
                output=error_result,
                level="ERROR"
            )
            return error_result

        result = tool_dispatcher[tool_name](arguments)

        output = {
            "tool_call_id": tool_call.id,
            "output": json.dumps(result, default=str)
        }

        # Log successful tool execution
        langfuse.update_current_span(
            output={"result": result},
            level="DEFAULT"
        )

        return output

    except Exception as e:
        error_details = traceback.format_exc()
        error_msg = f"Error executing tool: {str(e)}"

        print(f"{error_msg}\n{error_details}")

        error_result = {
            "tool_call_id": tool_call.id,
            "error": error_msg
        }

        # Log error in Langfuse
        langfuse.update_current_span(
            output={"error": error_msg, "traceback": error_details},
            level="ERROR"
        )

        return error_result

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
