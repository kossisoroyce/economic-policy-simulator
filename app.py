import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import json
from model import NigeriaModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Nigerian Policy Simulator",
    page_icon="🇳🇬",
    layout="wide"
)

# --- OpenAI API Setup ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("OpenAI API key not found. Please add it to your secrets.toml file.")
    st.stop()

# --- AI-Powered Functions ---
def get_ai_parameters(policy_text):
    """
    Uses OpenAI to parse natural language and extract simulation parameters.
    It's designed to return a JSON object, which makes it reliable.
    """
    prompt = f"""
    You are an economic policy analysis model. Your task is to read a user's description of a policy for Nigeria and extract specific numerical parameters for a simulation.

    The user's policy description is:
    '{policy_text}'

    The user's policy is: '{policy_text}'

    Your task is to:
    1.  **Analyze the policy**: What is the core intent? Does it primarily target household welfare, a specific economic sector, or both?
    2.  **Translate to parameters**: Convert this intent into values for `household_welfare_support` and `key_sector_investment`.
    3.  **Provide a rationale**: Briefly explain your reasoning in a `rationale` field. Why did you choose these parameter values based on the user's policy?

    Respond with ONLY a JSON object with four fields: `household_welfare_support`, `key_sector_investment`, `target_sector_name` (e.g., 'Agriculture', 'Tech Sector', or 'General Business' if no specific sector is targeted or investment is zero), and `rationale`. 
    
    Example for the policy 'Increase the healthcare budget by 10%':
    {{
        "household_welfare_support": 0.3,
        "key_sector_investment": 0.0,
        "rationale": "Increased healthcare spending primarily boosts household welfare by reducing out-of-pocket health expenses. I have modeled this as a low-to-moderate increase in household support. It has no direct impact on key sector investment."
    }}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful economic analyst that only responds with JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        params = json.loads(response.choices[0].message.content)
        return params
    except Exception as e:
        st.error(f"Error extracting parameters with AI: {e}")
        return None

def get_ai_analysis(policy_text, sim_results, gdp_growth, inflation_rate, initial_unemployment_rate, tax_rate):
    """
    Uses OpenAI's GPT-4 to generate a qualitative analysis of the policy's impact, now including fiscal sustainability.
    """
    # Calculate the percentage change for each income group.
    low_income_change = ((sim_results['final_low_income'] - sim_results['initial_low_income']) / sim_results['initial_low_income']) * 100 if sim_results['initial_low_income'] > 0 else 0
    medium_income_change = ((sim_results['final_medium_income'] - sim_results['initial_medium_income']) / sim_results['initial_medium_income']) * 100 if sim_results['initial_medium_income'] > 0 else 0
    high_income_change = ((sim_results['final_high_income'] - sim_results['initial_high_income']) / sim_results['initial_high_income']) * 100 if sim_results['initial_high_income'] > 0 else 0
    final_budget = sim_results['time_series_data']['Government Budget'].iloc[-1]
    final_unemployment = sim_results['time_series_data']['Unemployment Rate'].iloc[-1]

    prompt = f"""
    You are a senior economic advisor to the President of Nigeria, with deep expertise in development economics, inequality, and policy impact analysis. You are delivering a concise, professional briefing.

    **Context:**
    - **User's Proposed Policy:** "{policy_text}"
    - **Initial Economic & Fiscal Conditions:** GDP Growth at {gdp_growth}%, Inflation at {inflation_rate}%, Initial Unemployment at {initial_unemployment_rate}%, Tax Rate at {tax_rate}%.
    - **AI Policy Interpretation:** The policy was translated into a {sim_results['household_welfare_support']*100:.0f}% focus on household welfare and a {sim_results['key_sector_investment']*100:.0f}% focus on key sector investment.

    **Simulation Results (50 steps):**
    - **Overall Population Growth:** {sim_results['population_growth_percent']:.1f}%.
    - **Target Sector Growth:** {sim_results['target_sector_growth_percent']:.1f}%.
    - **Impact on Income Classes:**
        - Low-Income Households: {low_income_change:+.1f}% change.
        - Medium-Income Households: {medium_income_change:+.1f}% change.
        - High-Income Households: {high_income_change:+.1f}% change.
    - **Final Government Budget:** {final_budget:.2f} units.
    - **Final Unemployment Rate:** {final_unemployment:.1f}%.

    **Your Task:**
    Provide a brief, insightful analysis (4-5 paragraphs) of the likely real-world impacts of this policy. Structure your response with the following markdown headers:

    ### Executive Summary
    (Provide a one-sentence summary of the key outcome, focusing on its effect on social strata and fiscal health.)

    ### Analysis of Key Impacts
    (Discuss the main effects on businesses and households. Did the policy disproportionately benefit one income group? Link the simulation results directly to the policy's intent.)

    ### Dynamic Feedbacks & Stability
    (The unemployment rate is now DYNAMIC. It started at {initial_unemployment_rate}% and ended at {final_unemployment:.1f}%. Did the policy create a virtuous cycle (e.g., growth -> lower unemployment -> more growth) or a vicious cycle (e.g., population growth outpacing GDP, leading to higher unemployment)? Analyze the stability of the system you've created. Is this a stable path or a boom-bust cycle?)

    ### Fiscal Sustainability Analysis
    (Analyze the government's final budget. Was the policy self-funding, or did it create a deficit? At a {tax_rate}% tax rate, is this policy sustainable long-term? **Crucially, remember that if the budget becomes negative, all policy spending stops.** Did this happen? Comment on the trade-off between the policy's goals and its fiscal cost.)

    ### Risks & Social Considerations
    (What are the potential unintended consequences, especially regarding social cohesion? Could this policy lead to resentment? Is the growth in one sector leaving others behind?)

    Be direct, use professional language, and provide actionable insights for a policymaker concerned with both economic growth and social equity.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a world-class economic analyst for the Nigerian government, specializing in inequality and fiscal policy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=700
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating AI analysis: {e}" 

# --- Mesa Model Simulation ---
def run_nigeria_simulation(household_welfare_support, key_sector_investment, target_sector_name, gdp_growth, inflation_rate, unemployment_rate, tax_rate, steps=50, **kwargs):
    """
    This function runs our Mesa-based Nigeria model and returns the key results, including time-series data.
    """
    st.info(f"Running simulation for {steps} steps with Household Support: {household_welfare_support*100:.0f}%, Sector Investment: {key_sector_investment*100:.0f}%...")
    model = NigeriaModel(
        household_welfare_support=household_welfare_support,
        key_sector_investment=key_sector_investment,
        gdp_growth=gdp_growth / 100.0,
        inflation_rate=inflation_rate / 100.0,
        unemployment_rate=unemployment_rate / 100.0,
        tax_rate=tax_rate / 100.0, # Add tax rate
        **kwargs # Pass all the advanced assumptions to the model
    )

    # Run the simulation for the specified number of steps.
    for i in range(steps):
        model.step()

    # Get the complete data from the simulation.
    model_data = model.datacollector.get_model_vars_dataframe()

    # Calculate summary statistics.
    initial_households = model_data.iloc[0]["Households"]
    initial_business = model_data.iloc[0]["Businesses"]
    final_households = model_data.iloc[-1]["Households"]
    final_business = model_data.iloc[-1]["Businesses"]
    
    pop_growth = ((final_households - initial_households) / initial_households) * 100 if initial_households > 0 else 0
    target_sector_growth = ((final_business - initial_business) / initial_business) * 100 if initial_business > 0 else 0

    # Generate the final land use map.
    land_use_map = np.zeros((model.grid.width, model.grid.height))
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            content = model.grid.get_cell_list_contents([(x, y)])
            if content:
                if any(a.agent_type == "Business" for a in content):
                    land_use_map[x, y] = 2
                else:
                    land_use_map[x, y] = 1
            else:
                land_use_map[x, y] = 3
            
    # Return all the results in a dictionary.
    return {
        "target_sector_growth_percent": round(target_sector_growth, 2),
        "population_growth_percent": round(pop_growth, 2),
        "land_use_map": land_use_map,
        "household_welfare_support": household_welfare_support,
        "key_sector_investment": key_sector_investment,
        "target_sector_name": target_sector_name,
        "time_series_data": model_data,
        "initial_low_income": model_data.iloc[0]["Low Income"],
        "final_low_income": model_data.iloc[-1]["Low Income"],
        "initial_medium_income": model_data.iloc[0]["Medium Income"],
        "final_medium_income": model_data.iloc[-1]["Medium Income"],
        "initial_high_income": model_data.iloc[0]["High Income"],
        "final_high_income": model_data.iloc[-1]["High Income"],
    }

# --- Streamlit User Interface ---
st.title("🇳🇬 Nigerian Economic Policy Simulator")
st.write("Describe a policy in your own words, and let our AI-powered model simulate its potential impact.")

# We use session_state to store the extracted parameters across reruns
if 'params' not in st.session_state:
    st.session_state.params = None
if 'policy_text' not in st.session_state:
    st.session_state.policy_text = ""

with st.sidebar:
    st.header("1. Set Economic & Fiscal Scenario")
    # Sliders for setting the economic conditions
    gdp_growth = st.slider("GDP Growth Rate (%)", -5.0, 10.0, 3.0, 0.1)
    inflation_rate = st.slider("Inflation Rate (%)", 0.0, 50.0, 26.5, 0.5)
    unemployment_rate = st.slider("Initial Unemployment Rate (%)", 0.0, 25.0, 8.0, 0.1)
    tax_rate = st.slider("Tax Rate on Economic Activity (%)", 0.0, 50.0, 10.0, 0.5)

    st.header("2. Describe Your Policy")
    policy_description = st.text_area(
        "Describe the economic policy you want to test:", 
        height=150,
        value="Implement a 30% subsidy for agriculture and remove the fuel subsidy entirely to boost food production."
    )

    if st.button("Analyze Policy Intent"):
        with st.spinner("🤖 AI is interpreting your policy..."):
            st.session_state.params = get_ai_parameters(policy_description)
            st.session_state.policy_text = policy_description # Save the policy text for later

    if st.session_state.params:
        st.header("2. Review Extracted Parameters")
        st.write("The AI has interpreted your policy as follows:")
        st.metric("Household Welfare Support", f"{st.session_state.params['household_welfare_support']*100:.0f}%")
        st.metric("Key Sector Investment", f"{st.session_state.params['key_sector_investment']*100:.0f}%")
        st.metric("Identified Target Sector", st.session_state.params.get('target_sector_name', 'General Business'))
        st.info(f"**AI's Rationale:** {st.session_state.params['rationale']}")

    # --- Advanced Settings --- 
    with st.sidebar.expander("**Advanced Economic Assumptions**"):
        st.write("Control the underlying agent behavior. Use for sensitivity analysis.")
        low_income_inflation_sensitivity = st.slider("Low-Income Inflation Sensitivity", 0.0, 1.0, 0.7, 0.05)
        medium_income_inflation_sensitivity = st.slider("Medium-Income Inflation Sensitivity", 0.0, 1.0, 0.4, 0.05)
        high_income_inflation_sensitivity = st.slider("High-Income Inflation Sensitivity", 0.0, 1.0, 0.1, 0.05)
        unemployment_sensitivity = st.slider("Unemployment Sensitivity", 0.0, 1.0, 0.5, 0.05)
        welfare_support_effectiveness = st.slider("Welfare Support Effectiveness", 0.0, 1.0, 0.3, 0.05)
        gdp_growth_effectiveness = st.slider("GDP Growth Effectiveness", 0.0, 1.0, 0.2, 0.05)
        low_income_expansion_propensity = st.slider("Low-Income Expansion Propensity", 0.0, 0.5, 0.05, 0.01)
        medium_income_expansion_propensity = st.slider("Medium-Income Expansion Propensity", 0.0, 0.5, 0.1, 0.01)
        high_income_expansion_propensity = st.slider("High-Income Expansion Propensity", 0.0, 0.5, 0.15, 0.01)

# Main content area
if st.session_state.params:
    st.header("3. Run Simulation & See Results")
    if st.button("Run Simulation & Analyze", type="primary"):
        # We need to filter out the 'rationale' before passing params to the simulation
        sim_params = {k: v for k, v in st.session_state.params.items() if k != 'rationale'}
        sim_results = run_nigeria_simulation(
            **sim_params, 
            gdp_growth=gdp_growth, 
            inflation_rate=inflation_rate, 
            unemployment_rate=unemployment_rate,
            tax_rate=tax_rate, # Pass tax rate
            # Pass advanced parameters
            low_income_inflation_sensitivity=low_income_inflation_sensitivity,
            medium_income_inflation_sensitivity=medium_income_inflation_sensitivity,
            high_income_inflation_sensitivity=high_income_inflation_sensitivity,
            unemployment_sensitivity=unemployment_sensitivity,
            welfare_support_effectiveness=welfare_support_effectiveness,
            gdp_growth_effectiveness=gdp_growth_effectiveness,
            low_income_expansion_propensity=low_income_expansion_propensity,
            medium_income_expansion_propensity=medium_income_expansion_propensity,
            high_income_expansion_propensity=high_income_expansion_propensity
        )
        
        st.header("Simulation Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Target Sector Growth", f"{sim_results['target_sector_growth_percent']}%")
            st.metric("Household (Population) Growth", f"{sim_results['population_growth_percent']}%")
            
            st.write("**Final Land Use Map**")
            fig, ax = plt.subplots()
            cmap = plt.get_cmap('terrain', 3)
            im = ax.imshow(sim_results['land_use_map'], cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ticks=[1, 2, 3])
            cbar.ax.set_yticklabels(['Household', sim_results.get('target_sector_name', 'Business'), 'Undeveloped'])
            st.pyplot(fig)

        with col2:
            st.write("**AI-Powered Policy Analysis**")
            with st.spinner("🤖 Analyzing the policy's impact on the Nigerian economy..."):
                analysis = get_ai_analysis(st.session_state.policy_text, sim_results, gdp_growth, inflation_rate, unemployment_rate, tax_rate)
                st.markdown(analysis)

        # --- Time Series Charts ---
        st.header("Simulation Timeline")
        st.write("These charts show the evolution of the simulation over 50 steps.")
        
        st.subheader("Population Dynamics")
        chart_data_pop = sim_results['time_series_data'][['Low Income', 'Medium Income', 'High Income', 'Businesses']]
        st.line_chart(chart_data_pop)

        st.subheader("Government Fiscal Position")
        chart_data_gov = sim_results['time_series_data'][['Government Budget']]
        st.line_chart(chart_data_gov)

        st.subheader("Macroeconomic Dynamics")
        chart_data_macro = sim_results['time_series_data'][['Unemployment Rate']]
        st.line_chart(chart_data_macro)
else:
    st.info("Describe a policy in the sidebar and click 'Analyze Policy Intent' to begin.")
