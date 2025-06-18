# Nigerian Economic Policy Simulator

This project presents an interactive Nigerian Economic Policy Simulator. It combines a large-language-model (LLM) for natural language policy input, an agent-based computational economics (ACE) core for simulation, and real-time fiscal feedback loops. The goal is to provide a transparent, rapid-feedback tool for evaluating the complex, cross-sectoral impacts of fiscal and social policies in Nigeria.

The simulator allows users to enter plain-language policy proposals (e.g., "remove the fuel subsidy and re-invest 50% of the savings into agricultural support"). OpenAI's GPT-4 translates these proposals into quantitative model parameters. The core simulator, built with Mesa, models heterogeneous households (stratified by income) and businesses on a 2D grid. The model includes dynamic unemployment and government finance mechanisms, creating feedback where policy effects are constrained by a balanced budget approach.

The entire system is deployed as a Streamlit web application, enabling real-time, interactive policy experiments. Sensitivity sliders for core behavioral assumptions allow for robust exploration of model uncertainty.

## Features

*   **Natural Language Policy Input:** Use plain English to define economic policies.
*   **Agent-Based Modeling:** Simulates individual household and business behaviors.
*   **Dynamic Economic Indicators:** Tracks GDP, unemployment, inflation, government budget, and population demographics.
*   **Interactive Dashboard:** Visualizes simulation results in real-time using Streamlit.
*   **Sensitivity Analysis:** Adjust model parameters to explore different scenarios.

## Scientific Paper and Live Demo

For a detailed explanation of the methodology, findings, and implications of this work, please refer to our scientific paper:
*   **Paper:** (You can link to the `paper.pdf` in your repository if you commit it, or to an online version if you host it elsewhere.)

Experience the simulator live:
*   **Live Demo on Deepnote:** [https://deepnote.com/streamlit-apps/cc9b3ebf-0139-46f5-bdda-06a8160b7227](https://deepnote.com/streamlit-apps/cc9b3ebf-0139-46f5-bdda-06a8160b7227) (View in Firefox or Chrome)

## Running the Application Locally

To run the Nigerian Economic Policy Simulator on your local machine, follow these steps:

### 1. Prerequisites

*   Python 3.8 or higher
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### 2. Clone the Repository

```bash
git clone <repository-url> # Replace <repository-url> with the actual URL of your Git repository
cd policy_simulation_app # Or your repository's root directory name
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

### 5. Set Up OpenAI API Key

The application uses the OpenAI API to interpret natural language policy inputs. You need to provide your OpenAI API key.

1.  Create a directory named `.streamlit` in the root of the project directory if it doesn't already exist:
    ```bash
    mkdir .streamlit
    ```
2.  Inside the `.streamlit` directory, create a file named `secrets.toml`.
3.  Add your OpenAI API key to `secrets.toml` in the following format:
    ```toml
    OPENAI_API_KEY = "your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

### 6. Run the Streamlit Application

Once the dependencies are installed and the API key is set up, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will typically open the application in your default web browser. If not, the command will output a local URL (usually `http://localhost:8501`) that you can open manually.

## Project Structure

*   `app.py`: Main Streamlit application file.
*   `model.py`: Contains the Mesa agent-based model logic (`NigerianAgent`, `NigeriaModel`).
*   `requirements.txt`: Lists Python dependencies.
*   `.streamlit/secrets.toml`: Stores the OpenAI API key (ensure this is in your `.gitignore` if the repository is public and contains a real key).
*   `README.md`: This file.

## License

GPL
