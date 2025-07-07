import streamlit as st
import nest_asyncio

nest_asyncio.apply()
import os
import json
import pandas as pd
import pydantic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from model import NigeriaModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Nigerian Economic Policy Simulator",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Theme ---
st.markdown('<style>h1{color: #2E8B57;}</style>', unsafe_allow_html=True)

# --- Session State Initialization ---
def init_session_state():
    """Initializes all required session state variables with default values."""
    defaults = {
        'api_key': None,
        'policy_text': "",
        'ai_params': None,
        'simulation_results': None,
        'analysis_complete': False,
        # Economic Scenario
        'gdp_growth': 3.4,
        'inflation_rate': 22.97,
        'unemployment_rate': 5.3,
        'tax_rate': 24.0,
        # Advanced Assumptions
        'low_income_inflation_sensitivity': 0.7,
        'medium_income_inflation_sensitivity': 0.4,
        'high_income_inflation_sensitivity': 0.1,
        'unemployment_sensitivity': 0.5,
        'welfare_support_effectiveness': 0.3,
        'gdp_growth_effectiveness': 0.2,
        'low_income_expansion_propensity': 0.05,
        'medium_income_expansion_propensity': 0.1,
        'high_income_expansion_propensity': 0.15
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- AI & Simulation Functions ---

def get_ai_parameters(policy_text, api_key):
    """Uses LangChain with Google Gemini to extract structured economic parameters."""
    class PolicyParameters(pydantic.BaseModel):
        household_welfare_support: float = pydantic.Field(description="A float between 0.0 and 1.0 for direct household support.")
        key_sector_investment: float = pydantic.Field(description="A float between 0.0 and 1.0 for investment in a key sector.")
        target_sector_name: str = pydantic.Field(description="The name of the targeted economic sector.")
        rationale: str = pydantic.Field(description="A brief explanation for the chosen parameter values.")

    parser = JsonOutputParser(pydantic_object=PolicyParameters)
    prompt = ChatPromptTemplate.from_template(
        """You are an expert economic policy analysis model. Your task is to read a user's policy description for Nigeria and extract numerical parameters for a simulation.
        
        Policy Description: "{policy_text}"
        
        Respond with a JSON object matching this format: {format_instructions}
        """
    )
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.0,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        chain = prompt | llm | parser
        params = chain.invoke({
            "policy_text": policy_text,
            "format_instructions": parser.get_format_instructions()
        })
        return params
    except Exception as e:
        st.error(f"AI Parameter Extraction Failed: {e}")
        return None

def get_ai_analysis(policy_text, sim_results, api_key):
    """Generates a deep analysis of the simulation results using Google Gemini."""
    # Extract metrics from session state and simulation results
    gdp_growth = st.session_state.gdp_growth
    inflation_rate = st.session_state.inflation_rate
    initial_unemployment_rate = st.session_state.unemployment_rate
    tax_rate = st.session_state.tax_rate
    """Generates a deep analysis of the simulation results using Google Gemini."""
    # Extract final metrics for the prompt
    final_unemployment = sim_results['time_series_data']['Unemployment Rate'].iloc[-1]
    unemployment_change = final_unemployment - initial_unemployment_rate
    sector_growth = sim_results['target_sector_growth_percent']
    gov_budget = sim_results['time_series_data']['Government Budget'].iloc[-1]
    low_income_change = sim_results['final_low_income'] - sim_results['initial_low_income']
    high_income_change = sim_results['final_high_income'] - sim_results['initial_high_income']
    target_sector = sim_results['target_sector_name']

    prompt_template = """
    **Persona:** You are a senior economic advisor. You are known for your candid, clear-eyed, and rigorous assessments. Your task is to provide a deep, explicit, and unvarnished analysis of a proposed economic policy based on a simulation. Go beyond the surface-level numbers to explain the second and third-order effects. Your goal is to give the user a complete picture of the potential consequences, both positive and negative.

    **Policy Under Review:**
    "{policy_text}"

    **Initial Economic Conditions:**
    - GDP Growth: {gdp_growth:.1f}%
    - Inflation Rate: {inflation_rate:.1f}%
    - Initial Unemployment Rate: {initial_unemployment_rate:.1f}%
    - Government Tax Rate: {tax_rate:.1f}%

    **Key Simulation Results:**
    - Final Unemployment Rate: {final_unemployment:.2f}% (a change of {unemployment_change:+.2f}%)
    - Target Sector ('{target_sector}') Growth: {sector_growth:.2f}%
    - Final Government Budget: {gov_budget:,.0f}
    - Change in Low-Income Population: {low_income_change:.0f}
    - Change in High-Income Population: {high_income_change:.0f}

    **Your Mandate: A Deep and Candid Analysis**
    Provide a structured analysis in Markdown. Be direct. Also you do not need to format it like it is a letter. You can just write it as a normal brief.

    1.  **Executive Summary:** A concise, top-line summary. What is the single most important outcome of this policy, and what is the most significant risk?

    2.  **Deep Dive: Causal Analysis of Key Impacts:**
        - **Employment and Economic Structure:** The unemployment rate changed by {unemployment_change:+.2f}%. Explain the precise mechanism. Did the investment in '{target_sector}' create enough high-quality jobs to offset other effects? Or was the impact marginal? Explain *why* the number of businesses changed as it did.
        - **Household Prosperity and Inequality:** How did this policy impact the different income strata? Was the welfare support a temporary relief or did it foster genuine upward mobility? Explicitly state whether the policy likely increased or decreased income inequality and explain the causal chain.

    3.  **Explicit Risks and Negative Consequences:** This is the most critical section. Do not be vague. Provide a detailed, clear-eyed assessment of the potential downsides.
        - **Fiscal Sustainability:** Analyze the final government budget. Is the policy fiscally sustainable? If it creates a deficit, explain the long-term consequences (e.g., increased debt service, crowding out private investment, future austerity).
        - **Sectoral Imbalance & 'Dutch Disease' Risk:** The target sector grew by {sector_growth:.2f}%. Is there a risk of neglecting other vital sectors of the economy? Could this lead to an over-reliance on '{target_sector}', creating vulnerabilities?
        - **Inflationary Pressure:** Could the combination of welfare support and investment spending lead to significant demand-pull inflation that harms the very households the policy aims to help? Explain this risk clearly.

    4.  **Actionable Recommendation:** Conclude with a clear, decisive, and data-driven recommendation. Should the policy be **Approved**, **Amended**, or **Rejected**? If Amended, provide specific, actionable changes required to mitigate the risks you identified.
    """
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.5,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        formatted_prompt = prompt_template.format(
            policy_text=policy_text, gdp_growth=gdp_growth, inflation_rate=inflation_rate,
            initial_unemployment_rate=initial_unemployment_rate, tax_rate=tax_rate,
            final_unemployment=final_unemployment, unemployment_change=unemployment_change,
            sector_growth=sector_growth, gov_budget=gov_budget, low_income_change=low_income_change,
            high_income_change=high_income_change, target_sector=target_sector
        )
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        st.error(f"AI Analysis Failed: {e}")
        return "The AI analysis could not be completed due to an error."

def run_nigeria_simulation(steps=50):
    """Runs the agent-based model simulation."""
    
    # --- Define Population Scale ---
    # Each agent represents a larger group to model the ~200M population
    TARGET_POPULATION = 200_000_000
    POPULATION_SCALE = 10_000 # Each agent represents 10,000 people/entities

    # Collect all necessary parameters from session state
    sim_params = {
        "target_population": TARGET_POPULATION,
        "population_scale": POPULATION_SCALE,
        "household_to_business_ratio": 5.0,
        "household_welfare_support": st.session_state.ai_params['household_welfare_support'],
        "key_sector_investment": st.session_state.ai_params['key_sector_investment'],
        "gdp_growth": st.session_state.gdp_growth,
        "inflation_rate": st.session_state.inflation_rate,
        "unemployment_rate": st.session_state.unemployment_rate,
        "tax_rate": st.session_state.tax_rate,
    }
    # Add advanced parameters
    for key in [k for k in st.session_state.keys() if 'sensitivity' in k or 'propensity' in k or 'effectiveness' in k]:
        sim_params[key] = st.session_state[key]

    # The model now calculates its own grid size based on population
    model = NigeriaModel(**sim_params)

    # The model now calculates its own grid size based on population

    
    for _ in range(steps):
        model.step()
    
    model_data = model.datacollector.get_model_vars_dataframe()
    
    # Calculate results
    initial_households = model_data.iloc[0][['Low Income', 'Medium Income', 'High Income']].sum()
    final_households = model_data.iloc[-1][['Low Income', 'Medium Income', 'High Income']].sum()
    initial_businesses = model_data.iloc[0]['Businesses']
    final_businesses = model_data.iloc[-1]['Businesses']
    
    return {
        "time_series_data": model_data,
        "target_sector_growth_percent": ((final_businesses - initial_businesses) / initial_businesses) * 100 if initial_businesses > 0 else 0,
        "population_growth_percent": ((final_households - initial_households) / initial_households) * 100 if initial_households > 0 else 0,
        "initial_low_income": model_data.iloc[0]['Low Income'],
        "final_low_income": model_data.iloc[-1]['Low Income'],
        "initial_medium_income": model_data.iloc[0]['Medium Income'],
        "final_medium_income": model_data.iloc[-1]['Medium Income'],
        "initial_high_income": model_data.iloc[0]['High Income'],
        "final_high_income": model_data.iloc[-1]['High Income'],
        "target_sector_name": st.session_state.ai_params['target_sector_name']
    }

# --- UI Rendering Functions ---

def render_sidebar():
    """Renders the sidebar for user input, updating session_state directly."""
    with st.sidebar:
        
        
        # --- Step 1: Economic Scenario ---
        st.markdown("#### Step 1: Configure Economic Scenario")
        st.caption("Defaults have been set with current economic data. Only change if you wish to experiment with different scenarios.")
        with st.expander("Adjust Parameters", expanded=False):
            st.session_state.gdp_growth = st.slider("GDP Growth Rate (%)", -5.0, 10.0, st.session_state.gdp_growth, 0.1)
            st.session_state.inflation_rate = st.slider("Inflation Rate (%)", 0.0, 50.0, st.session_state.inflation_rate, 0.01)
            st.session_state.unemployment_rate = st.slider("Initial Unemployment Rate (%)", 0.0, 40.0, st.session_state.unemployment_rate, 0.1)
            st.session_state.tax_rate = st.slider("Government Tax Rate (%)", 5.0, 50.0, st.session_state.tax_rate, 0.5)

        

        # --- Step 2: Advanced Assumptions ---
        st.markdown("#### Step 2: Advanced Assumptions")
        st.caption("Adjust agent sensitivity and model behavior. :red[Warning: Do not touch unless you're an economist.]")
        with st.expander("Adjust Parameters", expanded=False):
            for key in [k for k in st.session_state.keys() if 'sensitivity' in k or 'propensity' in k or 'effectiveness' in k]:
                st.session_state[key] = st.slider(
                    key.replace('_', ' ').title(), 0.0, 1.0, st.session_state[key], 0.05
                )
        
        

        # --- Step 3: Propose a Policy ---
        st.markdown("#### Step 3: Propose a Policy")
        example_policies = {
            "Select an example...": "",
            "Agricultural Tech Investment": "Invest heavily in agricultural technology and infrastructure to boost food production, reduce reliance on imports, and create jobs in rural areas.",
            "Digital Skills Grant": "Launch a nationwide grant to fund digital skills training for young people, aiming to grow the tech sector and improve youth employment.",
            "Small Business Tax Cut": "Implement a 10% tax cut for small and medium-sized enterprises (SMEs) to encourage entrepreneurship and stimulate local economies."
        }
        
        def on_example_change():
            st.session_state.policy_text = example_policies[st.session_state.example_policy]

        st.selectbox(
            "Load an Example Policy (Optional)",
            options=list(example_policies.keys()),
            key='example_policy',
            on_change=on_example_change
        )
        
        st.session_state.policy_text = st.text_area(
            "**Propose or Modify an Economic Policy**",
            st.session_state.policy_text,
            height=150,
            placeholder="e.g., 'Invest in agricultural tech to boost food production and create jobs.'"
        )

        if st.button("Analyze Policy Intent"):
            if st.session_state.policy_text:
                with st.spinner("🤖 AI is interpreting your policy..."):
                    st.session_state.ai_params = get_ai_parameters(st.session_state.policy_text, st.session_state.api_key)
                    st.session_state.simulation_results = None # Reset results
                    st.session_state.analysis_complete = False
            else:
                st.warning("Please describe a policy first.")

def render_main_content():
    """Renders the main content area based on the app's state."""
    st.title("🇳🇬 Nigerian Economic Policy Simulator")

    if not st.session_state.ai_params:
        st.info("👈 Set the economic scenario, adjust assumptions, and propose a policy in the sidebar to begin.")
        with st.expander("Learn about the Model Methodology"):
            st.markdown("""
            ### Agent-Based Model (ABM)
            The simulation is powered by an Agent-Based Model (ABM) that models two primary agent types: Households and Businesses. Their interactions and decisions, influenced by your policy and macroeconomic factors, produce the emergent outcomes you see.
            ### AI Integration
            We use Google's Gemini Pro to interpret your natural language policy proposals and to provide a qualitative analysis of the simulation's results.
            """)
        return

    # Display AI-interpreted parameters
    params = st.session_state.ai_params
    st.subheader("AI-Interpreted Policy Parameters")
    st.caption(f"**Rationale:** *{params['rationale']}*" )
    
    cols = st.columns(3)
    cols[0].metric("Household Welfare Support", f"{params['household_welfare_support']:.0%}")
    cols[1].metric("Key Sector Investment", f"{params['key_sector_investment']:.0%}")
    cols[2].metric("Target Sector", params['target_sector_name'])

    if st.button("Run Simulation & Generate Analysis", type="primary"):
        with st.spinner("📈 Setting up country environment and running simulation..."):
            st.session_state.simulation_results = run_nigeria_simulation()
        with st.spinner("✍️ Gemini is drafting the policy briefing..."):
            st.session_state.ai_analysis = get_ai_analysis(
                st.session_state.policy_text, 
                st.session_state.simulation_results,
                st.session_state.api_key
            )
            st.session_state.analysis_complete = True

    if st.session_state.analysis_complete:
        
        st.subheader("Simulation Results & AI Briefing")
        
        sim_results = st.session_state.simulation_results
        tab1, tab2, tab3 = st.tabs(["Briefing", "Population Charts", "Economic Charts"])
        
        with tab1:
            st.markdown(st.session_state.ai_analysis, unsafe_allow_html=True)
        with tab2:
            st.subheader("Agent Population Over Time")
            chart_data_pop = sim_results['time_series_data'][['Low Income', 'Medium Income', 'High Income', 'Businesses']]
            st.line_chart(chart_data_pop)
        with tab3:
            st.subheader("Macroeconomic Indicators Over Time")
            chart_data_gov = sim_results['time_series_data'][['Government Budget', 'Unemployment Rate']]
            st.line_chart(chart_data_gov)
            
            st.subheader("Export Data")
            csv_data = sim_results['time_series_data'].to_csv().encode('utf-8')
            st.download_button(
                label="Download Simulation Data (CSV)",
                data=csv_data,
                file_name='nigeria_simulation_results.csv',
                mime='text/csv',
            )

# --- Main App Execution ---
def main():
    """Main function to run the Streamlit app."""
    init_session_state()
    
    # Check for API key from Hugging Face Secrets
    st.session_state.api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not st.session_state.api_key:
        st.error("🚨 I can't run without a Google AI API Key buddy!")
        st.info("Please add your Google AI API Key as a secret in your Hugging Face or Deepnote Space. Which is where i assume you're running this.. Name the secret 'GOOGLE_API_KEY'.If you're running this on your local machine, run GOOGLE_API_KEY='Your-own-API-key-here' streamlit run app.py")
        st.stop()
    
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()