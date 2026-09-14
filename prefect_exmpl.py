import httpx
from prefect import flow, task

@task(retries=2, retry_delay_seconds=5)
def fetch_weather(city: str) -> dict:
    """Task 1: Fetch data from a public API."""
    url = f"https://wttr.in{city}?format=j1"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()

@task
def parse_temperature(weather_data: dict) -> str:
    """Task 2: Process the fetched data."""
    current_condition = weather_data["current_condition"][0]
    temp_c = current_condition["temp_C"]
    feel_c = current_condition["FeelsLikeC"]
    return f"It is currently {temp_c}°C (Feels like {feel_c}°C)."

@flow(name="Weather Monitoring Pipeline")
def weather_workflow(city: str = "Vancouver"):
    """The main flow that orchestrates the tasks."""
    # 1. Run the first task
    raw_data = fetch_weather(city)
    
    # 2. Pass the output directly into the second task
    report = parse_temperature(raw_data)
    
    # 3. Print the final result
    print(report)

if __name__ == "__main__":
    # Run the flow locally just like a regular Python function
    weather_workflow(city="Toronto")
