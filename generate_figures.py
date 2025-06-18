import matplotlib.pyplot as plt
import numpy as np

def generate_system_architecture_diagram():
    """Generates and saves the system architecture diagram as an SVG."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Component boxes
    components = {
        'UI': (2, 5, "Streamlit UI"),
        'AI': (5, 5, "OpenAI API (GPT-4)"),
        'Model': (8, 5, "Mesa ABM Core"),
        'Output': (5, 2, "Visualizations & Results")
    }

    for name, (x, y, text) in components.items():
        ax.text(x, y, text, ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='black'))

    # Arrows and labels
    def draw_arrow(start, end, label):
        ax.annotate('', xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='center', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))

    draw_arrow((2.8, 5), (4.2, 5), 'Policy Text')
    draw_arrow((5.8, 5), (7.2, 5), 'JSON Parameters')
    draw_arrow((8, 4.8), (5.8, 2.2), 'Pandas DataFrame')
    draw_arrow((5, 2.2), (2.8, 4.8), 'Charts & Maps') # Feedback loop to UI

    plt.title('Figure 1: System Architecture')
    plt.savefig('figure1_architecture.svg', format='svg', bbox_inches='tight')
    plt.close()

def generate_time_series_plot():
    """Generates and saves the time-series results plot as an SVG."""
    steps = 50
    t = np.arange(steps)

    # Generate plausible mock data for the 'Fuel Subsidy Reallocation' scenario
    low_income = 60 + 2 * t * (t < 15) + (60 + 2 * 15 - 0.5 * (t - 15)) * (t >= 15)
    medium_income = 30 + 0.5 * t
    high_income = 10 + np.zeros(steps)
    
    budget = 50 - 4 * t
    unemployment = 8 - 0.2 * t * (t < 15) + (8 - 0.2*15 + 0.8 * (t-15)) * (t >= 15)
    unemployment = np.clip(unemployment, 5, 20)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Figure 2: Time-Series for Fuel Subsidy Reallocation Policy', fontsize=16)

    # Panel 1: Population by Income Class
    ax1.plot(t, low_income, label='Low Income')
    ax1.plot(t, medium_income, label='Medium Income')
    ax1.plot(t, high_income, label='High Income')
    ax1.set_ylabel('Number of Households')
    ax1.set_title('a) Population by Income Class')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Panel 2: Government Budget
    ax2.plot(t, budget, color='green', label='Government Budget')
    ax2.axhline(0, color='red', linestyle='--', lw=2, label='Budget Deficit Threshold')
    ax2.set_ylabel('Budget Units')
    ax2.set_title('b) Government Budget')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Panel 3: Unemployment Rate
    ax3.plot(t, unemployment, color='purple', label='Unemployment Rate')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('c) Unemployment Rate')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel('Simulation Step')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('figure2_time_series.svg', format='svg', bbox_inches='tight')
    plt.close()

def generate_land_use_maps():
    """Generates and saves the side-by-side land use maps as an SVG."""
    grid_size = 20

    # Helper function to generate agent positions
    def get_scenario_data(num_low, num_med, num_high, num_biz):
        data = {}
        data['Low Income'] = np.random.rand(num_low, 2) * grid_size
        data['Medium Income'] = np.random.rand(num_med, 2) * grid_size
        data['High Income'] = np.random.rand(num_high, 2) * grid_size
        data['Business'] = np.random.rand(num_biz, 2) * grid_size
        return data

    # Generate data for both scenarios
    baseline_data = get_scenario_data(num_low=80, num_med=35, num_high=10, num_biz=15)
    policy_data = get_scenario_data(num_low=50, num_med=45, num_high=12, num_biz=35)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Figure 3: Final Land Use Maps (Step 50)', fontsize=16)

    colors = {'Low Income': 'red', 'Medium Income': 'orange', 'High Income': 'yellow', 'Business': 'blue'}
    
    # Plot Baseline
    ax1.set_title('a) Baseline Scenario (No Policy)')
    for agent_type, pos in baseline_data.items():
        ax1.scatter(pos[:, 0], pos[:, 1], c=colors[agent_type], label=agent_type, s=50, alpha=0.8, edgecolors='w')
    ax1.set_xlim(0, grid_size)
    ax1.set_ylim(0, grid_size)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')

    # Plot Policy Scenario
    ax2.set_title('b) Fuel Subsidy Reallocation Policy')
    for agent_type, pos in policy_data.items():
        ax2.scatter(pos[:, 0], pos[:, 1], c=colors[agent_type], label=agent_type, s=50, alpha=0.8, edgecolors='w')
    ax2.set_xlim(0, grid_size)
    ax2.set_ylim(0, grid_size)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal')

    # Create a single legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('figure3_land_use.svg', format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_system_architecture_diagram()
    generate_time_series_plot()
    generate_land_use_maps()
    print("Figures generated successfully.")
