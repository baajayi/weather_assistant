```markdown
# Weather Assistant 🌦️

A smart weather information system powered by OpenAI and OpenWeatherMap APIs, providing current conditions, forecasts, and historical data through natural language queries.

## Features ✨

- **Natural Language Processing**  
  Ask weather questions in plain English
- **Multi-Source Data Integration**  
  Combines current, forecast, and historical weather data
- **Global Coverage**  
  Supports any city worldwide with automatic geocoding
- **AI-Powered Insights**  
  GPT-4o interpretation of raw weather data
- **Web Interface**  
  Responsive UI with markdown-formatted responses

## Tech Stack 🛠️

- **Backend**: Python 3.9+, Flask
- **AI**: OpenAI Assistants API (gpt-4o-mini)
- **Weather Data**: OpenWeatherMap APIs
- **Frontend**: Bootstrap 5, Markdown rendering
- **Security**: DOMPurify, secure session management

## Installation 💻

### Prerequisites
- Python 3.9+
- OpenAI API key
- OpenWeatherMap API key (with History API subscription)

### Setup

1. Clone repository:
   ```bash
   git clone https://github.com/baajayi/weather-assistant.git
   cd weather-assistant
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your API keys:
   ```ini
   OPENAI_API_KEY=your_openai_key_here
   OPENWEATHER_API_KEY=your_owm_key_here
   ```

5. Run application:
   ```bash
   flask run
   ```

## Usage 🚀

### Web Interface
Access at `http://localhost:5000`

Example queries:
- "What's the weather in Tokyo right now?"
- "Show historical data for London last week"
- "Will it rain in New York tomorrow afternoon?"

### API Endpoints

**POST /ask**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "Current weather in Paris"}' \
  http://localhost:5000/ask
```

Sample response:
```json
{
  "response": "**Current Weather in Paris, FR**\n\n🌡️ Temperature: 22°C (71.6°F)\n💧 Humidity: 65%\n🌬️ Wind: 12 km/h NW\n☁️ Conditions: Partly cloudy"
}
```

## Configuration ⚙️

| Environment Variable       | Required | Description                          |
|----------------------------|----------|--------------------------------------|
| `OPENAI_API_KEY`           | Yes      | OpenAI API key                       |
| `OPENWEATHER_API_KEY`      | Yes      | OpenWeatherMap API key               |
| `FLASK_ENV`                | No       | Set to "production" for deployment   |
| `FLASK_SECRET_KEY`         | No       | Secret key for session encryption    |

## API Reference 📚

### Available Tools
```python
get_weather_by_city(city_name, country_code=None, state_code=None)
get_historical_weather(lat, lon, start, cnt, data_type='hour')
datetime_to_utc_timestamp(datetime_str)
```

### Rate Limits
- OpenAI: 3,000 RPM / 10,000 TPM
- OpenWeatherMap: 60 calls/minute (free tier)

## Troubleshooting 🔧

**Common Issues**  
`401 Unauthorized` Error:
- Verify API keys in `.env`
- Ensure OpenWeatherMap History API subscription
- Check account activation status

`City Not Found` Error:
- Use country/state codes for ambiguous cities
- Example: "Springfield,US,MO"

## Contributing 🤝

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License 📄

MIT License

---

**Note**: Historical weather data requires OpenWeatherMap paid subscription. Demo mode available with limited functionality.
```

