# Nigerian Economic Policy Simulator

This project presents an interactive Nigerian Economic Policy Simulator. It combines a powerful large-language-model (LLM) for natural language policy input and an agent-based computational economics (ACE) core for simulation. The goal is to provide a transparent, rapid-feedback tool for evaluating the complex, cross-sectoral impacts of fiscal and social policies in Nigeria.

The simulator allows users to enter plain-language policy proposals (e.g., "remove the fuel subsidy and re-invest 50% of the savings into agricultural support"). **Google's Gemini 1.5 Pro** translates these proposals into quantitative model parameters. The core simulator, built with Mesa, models heterogeneous households (stratified by income) and businesses. The model includes dynamic unemployment and government finance mechanisms, creating feedback where policy effects are constrained by real-world economic data.

The entire system is deployed as a Streamlit web application, enabling real-time, interactive policy experiments.

-------
## App Showcase Description

### Nigerian Economic Policy Simulator
An advanced agent-based modeling tool that simulates how economic policies ripple through Nigeria's economy. Using autonomous agents representing households and businesses, it models complex economic interactions to predict policy impacts on employment, income distribution, and GDP growth. Input your policy in plain language, and the AI-powered simulator will generate detailed forecasts and risk analyses, providing valuable insights for evidence-based decision making.

## Features

* **Natural Language Policy Input:** Use plain English to define economic policies, powered by Google Gemini.
* **Agent-Based Modeling:** Simulates individual household and business behaviors based on real-world data.
* **Dynamic Economic Indicators:** Tracks GDP, unemployment, inflation, government budget, and population demographics.
* **Interactive Dashboard:** Visualizes simulation results in real-time using Streamlit.
* **Sensitivity Analysis:** Adjust macroeconomic and agent-level parameters to explore different scenarios.

## Data sources
**Inflation** Central Bank Of Nigeria: https://www.cbn.gov.ng/rates/inflrates.html
**GDP Growth Rate** World Bank: https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=NG
**Unemployment** Nigerian Bureau of Statistics: https://www.nigerianstat.gov.ng/pdfuploads/NLFS_Q1_2024_Report.pdf
**Tax Rate** https://tradingeconomics.com/nigeria/personal-income-tax-rate#:~:text=The%20Personal%20Income%20Tax%20Rate,of%2024.00%20percent%20in%202012.

## Running the Application

To run the Nigerian Economic Policy Simulator on your machine, follow these steps:

### 1. Prerequisites

* Python 3.8 or higher
* `pip` (Python package installer)
* `git` (for cloning the repository)

### 2. Clone the Repository

```bash
git clone <repository-url> # Replace <repository-url> with the actual URL of your Git repository
cd <repository-directory>
```

### 3. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Configure Your Google AI API Key

This project uses the Google AI API to interpret natural language policy descriptions. You will need to provide your own API key.

The application expects the key to be available as an environment variable named `GOOGLE_API_KEY`.

**On macOS/Linux:**

```bash
export GOOGLE_API_KEY='Your-own-API-key-here'
```

**On Windows (Command Prompt):**

```bash
set GOOGLE_API_KEY=Your-own-API-key-here
```

### 6. Run the Streamlit Application

Once the dependencies are installed and the API key is set, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will typically open the application in your default web browser at `http://localhost:8501`.

## Project Structure

* `app.py`: Main Streamlit application file containing the UI logic.
* `model.py`: Contains the Mesa agent-based model logic (`NigerianAgent`, `NigeriaModel`).
* `requirements.txt`: Lists Python dependencies for the project.
* `README.md`: This file.

## Future Work

* **Regional Granularity:** Introduce state or geopolitical zone-level analysis for more localized policy insights.
* **External Shock Modeling:** Add a module to test policy resilience against external shocks like oil price crashes or droughts.
* **Comparative Policy Analysis:** Implement a side-by-side comparison feature to evaluate multiple policy scenarios simultaneously.
* **Expanded Sector Detail:** Break down the business sector into more specific industries (e.g., Agriculture, Tech, Manufacturing) for more targeted simulations.


## License

GPL
