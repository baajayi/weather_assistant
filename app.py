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
from openai import OpenAIError

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse client for trace updates
langfuse = get_client()


def trace_update(**kwargs):
    """
    Best-effort Langfuse trace update.

    Tracing is observability, not application logic, so an SDK mismatch must never
    fail the request. langfuse 4.x removed Langfuse.update_current_trace(), which
    previously surfaced as an HTTP 500 from /ask on any host that resolved a newer
    version than the pinned one.
    """
    try:
        langfuse.update_current_trace(**kwargs)
    except Exception as e:
        print(f"Langfuse trace update skipped: {type(e).__name__}: {e}")


def span_update(**kwargs):
    """Best-effort Langfuse span update. See trace_update()."""
    try:
        langfuse.update_current_span(**kwargs)
    except Exception as e:
        print(f"Langfuse span update skipped: {type(e).__name__}: {e}")

# Get the OpenWeatherMap API key from environment variables
api_key = os.getenv('WEATHER_API_KEY')

# OpenWeatherMap reports temperature and wind in units that depend on the `units`
# parameter, but the JSON payload itself carries no unit labels. Without them the
# model guesses (and mislabels imperial wind speed as m/s), so every weather tool
# returns its payload wrapped with the units that were actually requested.
UNIT_LABELS = {
    'imperial': {'temperature': '°F', 'wind_speed': 'mph'},
    'metric': {'temperature': '°C', 'wind_speed': 'm/s'},
    'standard': {'temperature': 'K', 'wind_speed': 'm/s'},
}


def annotate_units(data, units='imperial'):
    """
    Wraps an OpenWeatherMap payload with explicit unit labels.

    Parameters:
    data (dict): Raw API response, or None if the request failed.
    units (str): 'imperial', 'metric', or 'standard'.

    Returns:
    dict: {'units', 'temperature_unit', 'wind_speed_unit', ..., 'data'}, or None if data is None.
    """
    if data is None:
        return None

    labels = UNIT_LABELS.get(units, UNIT_LABELS['imperial'])
    return {
        'units': units,
        'temperature_unit': labels['temperature'],
        'wind_speed_unit': labels['wind_speed'],
        # These are fixed by the API regardless of the `units` parameter.
        'wind_direction_unit': 'degrees',
        'precipitation_unit': 'mm',
        'pressure_unit': 'hPa',
        'humidity_unit': '%',
        'visibility_unit': 'm',
        'data': data,
    }
def get_weather_by_city(city_name, api_key, country_code=None, state_code=None, exclude=None, units='imperial'):
    """
    Fetches current weather for a city by name using OpenWeatherMap's Geocoding API
    to resolve coordinates, then the free-tier current weather endpoint.

    Parameters:
    city_name (str): Name of the city (e.g., "London").
    api_key (str): OpenWeatherMap API key.
    country_code (str, optional): Country code (e.g., "GB" for United Kingdom).
    state_code (str, optional): State code (e.g., "CA" for California).
    exclude (list or str, optional): Accepted for backwards compatibility and ignored;
        the current weather endpoint has no excludable sections.
    units (str): 'imperial' (°F), 'metric' (°C), or 'standard' (Kelvin).

    Returns:
    dict: Unit-annotated current weather, or None if an error occurs.
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

        # Step 2: Fetch current conditions. This deliberately does NOT use
        # get_openweather_onecall: One Call 2.5 is deprecated and returns 401 on
        # free-tier keys, which made every city lookup fail.
        return get_current_weather(lat, lon, units)

    except requests.exceptions.RequestException as e:
        print(f"Geocoding API error: {e}")
        return None


# Reuse the existing One Call API function
def get_openweather_onecall(lat, lon, api_key, exclude=None, units='imperial'):
    """
    Fetches current weather + forecast from OpenWeatherMap One Call API by coordinates.
    """
    url = os.getenv('OPENWEATHER_ONECALL_API_URL', 'https://api.openweathermap.org/data/2.5/onecall')
    params = {
        'lat': lat,
        'lon': lon,
        'appid': os.getenv('WEATHER_API_KEY'),
        'units': units,
    }

    if exclude:
        params['exclude'] = exclude if isinstance(exclude, str) else ','.join(exclude)

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return annotate_units(response.json(), units)
    except requests.exceptions.RequestException as e:
        print(f"One Call API error: {e}")
        return None


def get_current_weather(lat, lon, units='imperial'):
    """
    Fetches current weather conditions from OpenWeatherMap's free-tier /data/2.5/weather endpoint.
    """
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {
        'lat': lat,
        'lon': lon,
        'appid': os.getenv('WEATHER_API_KEY'),
        'units': units,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return annotate_units(response.json(), units)
    except requests.exceptions.RequestException as e:
        print(f"Current weather API error: {e}")
        return None


def get_forecast(lat, lon, api_key, units='imperial', cnt=40):
    """
    Fetches a 5-day / 3-hour forecast from OpenWeatherMap Forecast API.
    This is available on the free tier and returns up to 40 data points
    (5 days x 8 intervals per day).

    Parameters:
    lat (float): Latitude
    lon (float): Longitude
    api_key (str): OpenWeatherMap API key
    units (str): 'imperial' (°F), 'metric' (°C), or 'standard' (Kelvin)
    cnt (int): Number of 3-hour intervals to return (max 40)

    Returns:
    dict: Forecast data or None on error
    """
    url = 'https://api.openweathermap.org/data/2.5/forecast'
    params = {
        'lat': lat,
        'lon': lon,
        'appid': os.getenv('WEATHER_API_KEY'),
        'units': units,
        'cnt': cnt,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return annotate_units(response.json(), units)
    except requests.exceptions.RequestException as e:
        print(f"Forecast API error: {e}")
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
                          start: int, cnt: int, data_type: str = 'hour',
                          units: str = 'imperial') -> dict:
    """
    Fetches historical weather data from OpenWeatherMap's History API.

    Parameters:
    lat (float): Latitude of the location
    lon (float): Longitude of the location
    api_key (str): OpenWeatherMap API key with History API access
    start (int): Start time in UNIX timestamp (UTC)
    cnt (int): Number of data points to retrieve (max 24 for hourly, 30 for daily)
    data_type (str): Type of data - 'hour' or 'day' (default: 'hour')
    units (str): 'imperial' (°F), 'metric' (°C), or 'standard' (Kelvin)

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
        'appid': api_key,
        'units': units,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return annotate_units(response.json(), units)
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
If asked about non-weather topics, politely explain that you can only provide weather information.

Unit rules:
- Every weather tool returns the readings under a "data" key, alongside explicit unit
  labels such as "temperature_unit" and "wind_speed_unit".
- Always report values using those labels. Never infer or assume a unit from the value
  itself, and never convert between unit systems unless the user asks you to.

Tool selection rules:
- For CURRENT weather at a lat/lon: use get_current_weather.
- For a FORECAST at a lat/lon: use get_forecast.
- For weather by city name: use get_weather_by_city.
- Do NOT use get_openweather_onecall; it is deprecated and will fail."""


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
                    },
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric", "standard"],
                        "description": "Unit system: imperial (°F), metric (°C), standard (Kelvin). Default: imperial."
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
            "description": "Fetches current weather + 7-day daily forecast from OpenWeatherMap One Call API by coordinates.",
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
                    },
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric", "standard"],
                        "description": "Unit system: imperial (°F), metric (°C), standard (Kelvin). Default: imperial."
                    }
                },
                "required": ["lat", "lon", "api_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Fetches current weather conditions by coordinates using the free-tier OpenWeatherMap API. Use this for current weather requests when lat/lon are provided.",
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
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric", "standard"],
                        "description": "Unit system: imperial (°F), metric (°C), standard (Kelvin). Default: imperial."
                    }
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Fetches a 5-day / 3-hour step forecast from OpenWeatherMap Forecast API (free tier). Use this for multi-day or 7-day forecast requests.",
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
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric", "standard"],
                        "description": "Unit system: imperial (°F), metric (°C), standard (Kelvin). Default: imperial."
                    },
                    "cnt": {
                        "type": "integer",
                        "description": "Number of 3-hour forecast intervals to return (max 40 = 5 days). Default: 40."
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
                    },
                    "units": {
                        "type": "string",
                        "enum": ["imperial", "metric", "standard"],
                        "description": "Unit system: imperial (°F), metric (°C), standard (Kelvin). Default: imperial."
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
@app.errorhandler(OpenAIError)
def handle_openai_error(error):
    """
    Returns OpenAI SDK failures as JSON instead of an opaque HTTP 500.

    The thread/run calls in ask() can raise for reasons the caller needs to see
    (bad key, quota, an unknown thread id, an already-active run). Without this
    the reason only reaches the server's stderr.
    """
    message = str(error)
    print(f"OpenAI API error: {type(error).__name__}: {message}")
    print(traceback.format_exc())

    # A thread whose previous run never finished (e.g. the server was restarted
    # mid-poll) rejects every subsequent message. Drop the stored thread so the
    # next request starts a fresh conversation instead of failing forever.
    thread_reset = "while a run" in message and "is active" in message
    if thread_reset:
        session.pop("thread_id", None)
        session.modified = True
        message += (
            " The stored conversation thread was stuck and has been cleared - "
            "please send your message again."
        )

    return jsonify({
        "error": message,
        "error_type": type(error).__name__,
        "thread_reset": thread_reset,
    }), 502


@app.route('/ask', methods=['POST'])
@observe()
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Log user input to Langfuse
    trace_update(
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
    trace_update(
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
        if run.status in ["completed", "failed", "expired", "cancelled", "incomplete"]:
            print(f"Run status: {run.status}")
            if run.last_error:
                print(f"Run last_error: {run.last_error.code}: {run.last_error.message}")
            incomplete_details = getattr(run, "incomplete_details", None)
            if incomplete_details:
                print(f"Run incomplete_details: {incomplete_details}")
            break
            
        # If we're here, the run is still in progress
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        print(f"Run status: {run.status}")


    # If the run never completed there is no assistant reply to read. Returning the
    # newest thread message here would hand back the user's own question.
    if run.status != "completed":
        incomplete_details = getattr(run, "incomplete_details", None)
        if run.last_error:
            error_code = run.last_error.code
            error_message = run.last_error.message
        elif incomplete_details:
            # 'incomplete' runs report a reason here rather than in last_error.
            error_code = getattr(incomplete_details, "reason", None)
            error_message = f"run stopped early (reason: {error_code})"
        else:
            error_code = None
            error_message = "The API did not report an error message."
        return _error_response(
            thread_id,
            run,
            f"The assistant run {run.status}: {error_message}",
            error_code=error_code,
        )

    assistant_message = next(
        (
            message
            for message in client.beta.threads.messages.list(thread_id=thread_id, order="desc").data
            if message.role == "assistant" and message.run_id == run.id
        ),
        None,
    )
    text_block = next(
        (block for block in (assistant_message.content if assistant_message else []) if block.type == "text"),
        None,
    )
    if text_block is None:
        return _error_response(
            thread_id,
            run,
            "The run completed but produced no assistant text reply.",
        )

    final_message = format_weather_response(text_block.text.value)
    print(f"Final message: {final_message}")

    # Update trace with the final response
    trace_update(
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


def _error_response(thread_id, run, message, error_code=None):
    """Log a failed run to stdout and Langfuse, and build the JSON error response."""
    print(f"Returning error for run {run.id}: {message}")

    trace_update(
        output={"error": message},
        metadata={
            "thread_id": thread_id,
            "run_id": run.id,
            "status": run.status,
            "error_code": error_code,
            "error_message": message,
        }
    )
    span_update(level="ERROR", status_message=message)

    return jsonify({
        "thread_id": thread_id,
        "run_id": run.id,
        "status": run.status,
        "error": message,
    }), 502

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
            span_update(
                name=f"tool_execution_{tool_name}",
                input={"tool_call_id": tool_call.id, "raw_arguments": tool_call.function.arguments},
                output=error_result,
                level="ERROR"
            )
            return error_result

        print(f"Executing tool: {tool_name}")
        print(f"With arguments: {arguments}")

        # Log tool execution start
        span_update(
            name=f"tool_execution_{tool_name}",
            input={
                "tool_name": tool_name,
                "arguments": arguments,
                "tool_call_id": tool_call.id
            }
        )

        # Tool function mapping
        tool_dispatcher = {
            "get_current_weather": lambda args: get_current_weather(
                lat=args.get("lat"),
                lon=args.get("lon"),
                units=args.get("units", "imperial")
            ),
            "get_weather_by_city": lambda args: get_weather_by_city(
                city_name=args.get("city_name"),
                api_key=args.get("api_key"),
                country_code=args.get("country_code"),
                state_code=args.get("state_code"),
                exclude=args.get("exclude"),
                units=args.get("units", "imperial")
            ),
            "get_openweather_onecall": lambda args: get_openweather_onecall(
                lat=args.get("lat"),
                lon=args.get("lon"),
                api_key=args.get("api_key"),
                exclude=args.get("exclude"),
                units=args.get("units", "imperial")
            ),
            "get_forecast": lambda args: get_forecast(
                lat=args.get("lat"),
                lon=args.get("lon"),
                api_key=args.get("api_key"),
                units=args.get("units", "imperial"),
                cnt=args.get("cnt", 40)
            ),
            "get_historical_weather": lambda args: get_historical_weather(
                lat=args.get("lat"),
                lon=args.get("lon"),
                api_key=args.get("api_key"),
                start=args.get("start"),
                cnt=args.get("cnt"),
                data_type=args.get("data_type", "hour"),
                units=args.get("units", "imperial")
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
            span_update(
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
        span_update(
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
        span_update(
            output={"error": error_msg, "traceback": error_details},
            level="ERROR"
        )

        return error_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    session.pop('thread_id', None)
    session.modified = True
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
