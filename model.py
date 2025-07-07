import mesa
import math

class NigerianAgent(mesa.Agent):
    """
    This represents a single agent in our simulation, which can be a household or a business.
    Their behavior is now influenced by both government policies and broader economic conditions.
    """
    def __init__(self, unique_id, model, agent_type, income_level=None):
        # DEEPNOTE WORKAROUND: The super().__init__(unique_id, model) call is removed.
        # It causes a TypeError in the Deepnote environment due to a bug in the installed
        # version of the Mesa library. We manually initialize the agent's core attributes instead.
        self.unique_id = unique_id
        self.model = model
        self.pos = None  # The grid placement will set this, but it's good practice to initialize.

        self.agent_type = agent_type
        # --- Assign Income Level for Households ---
        # This adds a layer of social stratification to our model.
        if self.agent_type == "Household":
            self.income_level = income_level or self.random.choices(["low", "medium", "high"], [0.4, 0.5, 0.1])[0]

    def step(self):
        # This method is called for every agent at every step of the simulation.
        
        # --- Logic for Agricultural Businesses ---
        if self.agent_type == "Business":
            # We calculate a 'business_confidence' score.
            # High GDP growth and subsidies boost confidence, while high inflation hurts it.
            confidence = (self.model.gdp_growth * 0.5) + (self.model.key_sector_investment * 0.5) - (self.model.inflation_rate * 0.3)
            
            # The chance to expand is based on this confidence score.
            if self.random.random() < confidence:
                # Find a nearby empty spot to create a new business.
                empty_cells = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False, radius=2)
                empty_cells = [cell for cell in empty_cells if self.model.grid.is_cell_empty(cell)]
                if empty_cells:
                    new_pos = self.random.choice(empty_cells)
                    new_agent = NigerianAgent(self.model.next_id(), self.model, "Business")
                    self.model.grid.place_agent(new_agent, new_pos)
                    self.model.schedule.add(new_agent)

        # --- Logic for Households ---
        if self.agent_type == "Household":
            # --- Household Logic ---
            # The agent's stability is now calculated using the model's sensitivity parameters.
            # This allows us to test different assumptions about how agents react to economic forces.
            if self.income_level == "low":
                inflation_sensitivity = self.model.low_income_inflation_sensitivity
                expansion_propensity = self.model.low_income_expansion_propensity
            elif self.income_level == "medium":
                inflation_sensitivity = self.model.medium_income_inflation_sensitivity
                expansion_propensity = self.model.medium_income_expansion_propensity
            else: # High income
                inflation_sensitivity = self.model.high_income_inflation_sensitivity
                expansion_propensity = self.model.high_income_expansion_propensity

            stability = (
                (self.model.household_welfare_support * self.model.welfare_support_effectiveness) - 
                (self.model.inflation_rate * inflation_sensitivity) - 
                (self.model.unemployment_rate * self.model.unemployment_sensitivity) + 
                (self.model.gdp_growth * self.model.gdp_growth_effectiveness)
            )

            # Dissolve if conditions are very poor (e.g., fall into poverty)
            if stability < -0.15 and self.random.random() < abs(stability):
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                return # Agent is removed, no further action
            
            # Expand if conditions are good (create a new household of the same income class)
            if stability > 0.1 and self.random.random() < (stability * expansion_propensity):
                self.expand()

    def expand(self):
        """Create a new agent, which also triggers a tax collection event."""
        empty_cells = [cell for cell in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False) if self.model.grid.is_cell_empty(cell)]
        if empty_cells:
            new_pos = self.random.choice(empty_cells)
            new_agent_id = self.model.next_id()
            # When a household expands, the new one has the same income level.
            income_level = self.income_level if self.agent_type == "Household" else None
            new_agent = NigerianAgent(new_agent_id, self.model, self.agent_type, income_level=income_level)
            self.model.grid.place_agent(new_agent, new_pos)
            self.model.schedule.add(new_agent)
            # --- Tax Collection ---
            # Expansion represents economic activity, which is taxed.
            self.model.collect_tax()

    def dissolve(self):
        """Remove the agent from the simulation."""
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

class NigeriaModel(mesa.Model):
    """
    This is the main model for our Nigerian policy simulation.
    It now includes macroeconomic indicators to create a more realistic environment.
    """
    def __init__(self, target_population=200_000_000, population_scale=10_000,
                 household_to_business_ratio=5.0,
                 household_welfare_support=0.1, key_sector_investment=0.1,
                 gdp_growth=0.034, inflation_rate=0.2297, unemployment_rate=0.053,
                 tax_rate=0.24,
                 # --- Advanced Economic Assumptions ---
                 low_income_inflation_sensitivity=0.7, medium_income_inflation_sensitivity=0.4, high_income_inflation_sensitivity=0.1,
                 unemployment_sensitivity=0.5,
                 welfare_support_effectiveness=0.3, gdp_growth_effectiveness=0.2,
                 low_income_expansion_propensity=0.05, medium_income_expansion_propensity=0.1, high_income_expansion_propensity=0.15):
        super().__init__()
        # --- Model Parameters ---
        self.target_population = target_population
        self.population_scale = population_scale
        self.household_to_business_ratio = household_to_business_ratio

        # --- Calculate number of agents and grid size ---
        num_agents_to_simulate = int(self.target_population / self.population_scale)
        
        # Maintain household/business ratio
        num_businesses = int(num_agents_to_simulate / (self.household_to_business_ratio + 1))
        num_households = num_agents_to_simulate - num_businesses

        # Dynamically size grid to have ~50% density to allow for expansion
        grid_area = num_agents_to_simulate * 2
        self.width = int(math.sqrt(grid_area))
        self.height = int(math.sqrt(grid_area))

        # Policy Levers
        self.household_welfare_support = household_welfare_support
        self.key_sector_investment = key_sector_investment
        # Economic Indicators
        self.gdp_growth = gdp_growth
        self.inflation_rate = inflation_rate
        self.unemployment_rate = unemployment_rate

        # --- Store Advanced Assumptions ---
        self.low_income_inflation_sensitivity = low_income_inflation_sensitivity
        self.medium_income_inflation_sensitivity = medium_income_inflation_sensitivity
        self.high_income_inflation_sensitivity = high_income_inflation_sensitivity
        self.unemployment_sensitivity = unemployment_sensitivity
        self.welfare_support_effectiveness = welfare_support_effectiveness
        self.gdp_growth_effectiveness = gdp_growth_effectiveness
        self.low_income_expansion_propensity = low_income_expansion_propensity
        self.medium_income_expansion_propensity = medium_income_expansion_propensity
        self.high_income_expansion_propensity = high_income_expansion_propensity

        # --- Government & Budget ---
        self.tax_rate = tax_rate
        self.government_budget = 0
        self.total_welfare_spending = 0
        self.total_investment_spending = 0

        # --- Setup Simulation Environment ---
        self.grid = mesa.space.MultiGrid(self.width, self.height, True)
        self.schedule = mesa.time.RandomActivation(self)
        
        # --- Create the initial population of agents ---
        agent_id = 0

        # Get all empty cells, shuffle them to ensure random placement
        empty_cells = list(self.grid.empties)
        self.random.shuffle(empty_cells)

        # Create households
        for _ in range(num_households):
            if not empty_cells:
                print("Warning: Not enough space on the grid to place all households.")
                break
            pos = empty_cells.pop()
            agent = NigerianAgent(agent_id, self, "Household")
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            agent_id += 1

        # Create businesses
        for _ in range(num_businesses):
            if not empty_cells:
                print("Warning: Not enough space on the grid to place all businesses.")
                break
            pos = empty_cells.pop()
            agent = NigerianAgent(agent_id, self, "Business")
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            agent_id += 1

        # --- Unique Agent ID Generator ---
        self._next_available_id = agent_id

        def _next_id():
            self._next_available_id += 1
            return self._next_available_id
        
        self.next_id = _next_id

        # --- Feedback Loop State ---
        self.previous_population = self.schedule.get_agent_count()

        # --- Setup Data Collection ---
        self.datacollector = mesa.DataCollector(
            model_reporters={
                # Scale up the results to reflect the real population
                "Households": lambda m: sum(1 for a in m.schedule.agents if a.agent_type == "Household") * m.population_scale,
                "Businesses": lambda m: sum(1 for a in m.schedule.agents if a.agent_type == "Business") * m.population_scale,
                "Low Income": lambda m: sum(1 for a in m.schedule.agents if a.agent_type == "Household" and a.income_level == "low") * m.population_scale,
                "Medium Income": lambda m: sum(1 for a in m.schedule.agents if a.agent_type == "Household" and a.income_level == "medium") * m.population_scale,
                "High Income": lambda m: sum(1 for a in m.schedule.agents if a.agent_type == "Household" and a.income_level == "high") * m.population_scale,
                # Government Budget is an absolute value, not a scaled count
                "Government Budget": lambda m: m.government_budget,
                # Dynamic Macro-variables
                "Unemployment Rate": lambda m: m.unemployment_rate * 100,
            }
        )
        self.datacollector.collect(self)

    def step(self):
        # This method advances the simulation by one time-step.
        self.update_macro_variables() # First, update the economy based on the last step's results.
        self.spend_on_policies() # Government spends from the budget.
        self.schedule.step() # Then, we activate all the agents, who react to the new conditions.
        self.datacollector.collect(self) # Finally, we collect the data for the new state.

    def update_macro_variables(self):
        """This is the core of the feedback loop. It adjusts the unemployment rate.
        """
        current_population = self.schedule.get_agent_count()
        if self.previous_population > 0:
            pop_growth_rate = (current_population - self.previous_population) / self.previous_population
        else:
            pop_growth_rate = 0

        # If population growth outpaces economic growth (GDP), unemployment rises.
        # The '0.5' is a dampening factor to prevent wild swings.
        unemployment_change = (pop_growth_rate - self.gdp_growth) * 0.5
        self.unemployment_rate += unemployment_change
        
        # Clamp the unemployment rate to a realistic range (e.g., 1% to 50%)
        self.unemployment_rate = max(0.01, min(0.5, self.unemployment_rate))

        # Update the state for the next step's calculation.
        self.previous_population = current_population

    def collect_tax(self):
        """Collects tax revenue based on the tax rate. Called when an agent expands."""
        self.government_budget += self.tax_rate # Simple tax model: each expansion contributes to the budget

    def spend_on_policies(self):
        """Deduct policy costs from the budget (welfare for households, investment support for target-sector businesses)."""
        # Welfare cost scales with total population; investment cost only with businesses.
        welfare_cost = self.household_welfare_support * self.schedule.get_agent_count()
        investment_cost = self.key_sector_investment * sum(1 for a in self.schedule.agents if a.agent_type == "Business")
        
        self.government_budget -= (welfare_cost + investment_cost)
        self.total_welfare_spending += welfare_cost
        self.total_investment_spending += investment_cost

    def get_effective_welfare(self):
        """Returns the actual welfare support, which is 0 if the budget is negative."""
        return self.household_welfare_support if self.government_budget > 0 else 0

    def get_effective_investment(self):
        """Returns the actual investment support, which is 0 if the budget is negative."""
        return self.key_sector_investment if self.government_budget > 0 else 0
