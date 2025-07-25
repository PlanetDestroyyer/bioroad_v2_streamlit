import requests
from datetime import datetime, timedelta

def get_pune_weather_forecast(location):
    url = "https://weather-api167.p.rapidapi.com/api/weather/forecast"
    querystring = {f"place": {location}, "units": "metric"}
    headers = {
        "x-rapidapi-key": "46d33ff5a0mshe40b3178c84a8b4p1f5cf6jsnb84963ce1cd1",
        "x-rapidapi-host": "weather-api167.p.rapidapi.com",
        "Accept": "application/json"
    }

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    def calculate_gdd(temp_min, temp_max, base_temp=10):
        avg_temp = (temp_min + temp_max) / 2
        gdd = max(0, avg_temp - base_temp)
        return round(gdd, 2)

    def estimate_et(temp, humidity):
        et = max(0, (temp - 10) * (1 - humidity / 100) * 0.1)
        return round(et, 2)

    results = []
    if response.status_code == 200 and 'list' in data:
        for entry in data['list']:
            try:
                forecast_date = datetime.strptime(entry['dt_txt'], '%Y-%m-%d %H:%M:%S').date()
            except (ValueError, KeyError):
                return [{"error": "Invalid date format or missing dt_txt in response"}]

            if forecast_date in [yesterday, today, tomorrow]:
                main = entry['main']
                wind = entry['wind']
                clouds = entry['clouds']
                rain = entry.get('rain', {})
                weather_desc = entry['weather'][0]['description']

                temp = main['temprature']
                temp_min = main['temprature_min']
                temp_max = main['temprature_max']
                feels_like = main['temprature_feels_like']
                humidity = main['humidity']
                precipitation = rain.get('amount', 0)

                gdd = calculate_gdd(temp_min, temp_max)
                et = estimate_et(temp, humidity)
                frost_warning = "Yes" if temp_min <= 0 else "No"
                severe_weather = "Yes" if wind['speed'] > 10 or precipitation > 10 else "No"

                weather_data = {
                    "date_time": entry['dt_txt'],
                    "temperature": round(temp, 2),
                    "feels_like": round(feels_like, 2),
                    "temp_min": round(temp_min, 2),
                    "temp_max": round(temp_max, 2),
                    "humidity": humidity,
                    "precipitation": precipitation,
                    "wind_speed": wind['speed'],
                    "wind_direction": f"{wind['direction']} ({wind['degrees']}°)",
                    "cloud_cover": clouds['cloudiness'],
                    "frost_warning": frost_warning,
                    "gdd": gdd,
                    "evapotranspiration": et,
                    "severe_weather": severe_weather,
                    "description": weather_desc
                }
                results.append(weather_data)
        return results
    else:
        return [{"error": f"Error fetching data: {data.get('message', 'Unknown error')}"}]

# Run and print result
def format_weather_for_ai(weather_data, location):
    """Format weather data for AI consumption"""
    if not weather_data or weather_data[0].get('error'):
        return f"Weather data for {location} is currently unavailable."
    
    weather_summary = f"Weather forecast for {location}:\n"
    for day_data in weather_data:
        weather_summary += f"""
Date: {day_data['date_time']}
Temperature: {day_data['temperature']}°C (Min: {day_data['temp_min']}°C, Max: {day_data['temp_max']}°C)
Humidity: {day_data['humidity']}%
Precipitation: {day_data['precipitation']}mm
Wind: {day_data['wind_speed']} m/s
Growing Degree Days: {day_data['gdd']}
Evapotranspiration: {day_data['evapotranspiration']}mm
Frost Warning: {day_data['frost_warning']}
Severe Weather: {day_data['severe_weather']}
Description: {day_data['description']}
---
"""
    return weather_summary


city = "Pune"
weather_data = get_pune_weather_forecast(city)
print(weather_data)
formatted_weather = format_weather_for_ai(weather_data, city)
print( format_weather_for_ai(weather_data, city))


