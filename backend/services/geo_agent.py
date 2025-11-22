from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
import json
import logging
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class GeoOutput(BaseModel):
    """Geopolitical simulation output model."""
    crisis: str = Field(..., description="The geopolitical crisis scenario")
    user_choice: str = Field(..., description="The decision made by the user")
    tension_level: int = Field(..., ge=0, le=100, description="Global tension level (0-100)")
    stability_index: int = Field(..., ge=0, le=100, description="Regional stability index (0-100)")
    alliances: List[str] = Field(..., min_items=1, description="Affected alliances and partnerships")
    outcome_summary: str = Field(..., min_length=50, description="Detailed summary of consequences")
    
    @field_validator('tension_level', 'stability_index')
    @classmethod
    def validate_range(cls, v: int) -> int:
        """Ensure values are within 0-100 range."""
        if not 0 <= v <= 100:
            raise ValueError(f"Value must be between 0 and 100, got {v}")
        return v

class GeoState(BaseModel):
    """State management for geopolitical simulation workflow."""
    crisis: str
    user_choice: str
    raw_response: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    final: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# --- Initialize LLM ---
def initialize_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Gemini LLM with configuration."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.4,
        max_output_tokens=2048,
    )

llm = initialize_llm()

# --- Enhanced Prompt Template ---
geo_prompt = PromptTemplate(
    input_variables=["crisis", "user_choice"],
    template="""You are an expert geopolitical analyst and strategic decision simulator.

**CRISIS SCENARIO:**
{crisis}

**USER DECISION:**
{user_choice}

Analyze the geopolitical consequences of this decision comprehensively. Consider:
- Regional and global power dynamics
- Economic implications
- Military tensions and de-escalation factors
- Alliance shifts and diplomatic relationships
- Long-term stability impacts
- Potential for conflict or cooperation

Return your analysis as valid JSON with this exact structure:
{{
  "crisis": "{crisis}",
  "user_choice": "{user_choice}",
  "tension_level": <integer 0-100, where 0=complete peace, 100=imminent war>,
  "stability_index": <integer 0-100, where 0=total chaos, 100=perfect stability>,
  "alliances": ["List of affected nations/alliances", "NATO", "EU", "Bilateral partnerships", "Regional coalitions"],
  "outcome_summary": "A detailed 3-5 sentence summary explaining the cascading effects of this decision, including immediate consequences, medium-term impacts, and potential long-term ramifications for regional and global stability."
}}

Be realistic and nuanced. Provide at least 3-5 affected alliances and a comprehensive outcome summary."""
)

# --- LangGraph Nodes ---
def geo_prompt_node(state: GeoState) -> GeoState:
    """Generate geopolitical analysis using LLM."""
    try:
        logger.info(f"Analyzing crisis: {state.crisis[:50]}...")
        chain = geo_prompt | llm
        response = chain.invoke({
            "crisis": state.crisis,
            "user_choice": state.user_choice
        })
        
        # Extract content from response
        state.raw_response = response.content
        logger.info("Successfully generated geopolitical analysis")
        
    except Exception as e:
        logger.error(f"Error in geo_prompt_node: {str(e)}")
        state.error = f"Generation error: {str(e)}"
    
    return state

def geo_parse_node(state: GeoState) -> GeoState:
    """Parse the JSON response from the LLM."""
    if state.error:
        return state
    
    try:
        # Clean the response
        content = state.raw_response.strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        # Parse JSON
        state.parsed_json = json.loads(content)
        logger.info("Successfully parsed JSON response")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Raw response: {state.raw_response[:200]}...")
        state.error = f"Invalid JSON response: {str(e)}"
    except Exception as e:
        logger.error(f"Error in geo_parse_node: {str(e)}")
        state.error = f"Parsing error: {str(e)}"
    
    return state

def geo_validate_node(state: GeoState) -> GeoState:
    """Validate the parsed JSON against the Pydantic model."""
    if state.error:
        return state
    
    try:
        # Validate with Pydantic
        validated = GeoOutput(**state.parsed_json)
        state.final = validated.model_dump()
        
        logger.info(f"Successfully validated output:")
        logger.info(f"  - Tension Level: {validated.tension_level}/100")
        logger.info(f"  - Stability Index: {validated.stability_index}/100")
        logger.info(f"  - Affected Alliances: {len(validated.alliances)}")
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        state.error = f"Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Error in geo_validate_node: {str(e)}")
        state.error = f"Validation error: {str(e)}"
    
    return state

# --- Build LangGraph ---
def build_graph() -> StateGraph:
    """Construct the LangGraph workflow."""
    graph = StateGraph(GeoState)
    
    # Add nodes
    graph.add_node("prompt", geo_prompt_node)
    graph.add_node("parse", geo_parse_node)
    graph.add_node("validate", geo_validate_node)
    
    # Set up workflow
    graph.set_entry_point("prompt")
    graph.add_edge("prompt", "parse")
    graph.add_edge("parse", "validate")
    graph.add_edge("validate", END)
    
    return graph.compile()

# Compile the graph
geo_agent = build_graph()

# --- Main Function ---
def simulate_geopolitics(crisis: str, user_choice: str) -> Dict[str, Any]:
    """
    Simulate geopolitical consequences of a decision.
    
    Args:
        crisis: Description of the geopolitical crisis
        user_choice: The decision/action being taken
        
    Returns:
        Dictionary containing the geopolitical analysis with:
        - tension_level: Global tension (0-100)
        - stability_index: Regional stability (0-100)
        - alliances: List of affected alliances
        - outcome_summary: Detailed consequence analysis
        
    Raises:
        ValueError: If simulation fails or inputs are invalid
    """
    if not crisis or not user_choice:
        raise ValueError("Both 'crisis' and 'user_choice' must be provided")
    
    if len(crisis) < 10:
        raise ValueError("Crisis description must be at least 10 characters")
    
    if len(user_choice) < 5:
        raise ValueError("User choice must be at least 5 characters")
    
    logger.info(f"Starting geopolitical simulation")
    logger.info(f"Crisis: {crisis[:50]}...")
    logger.info(f"Decision: {user_choice[:50]}...")
    
    initial_state = GeoState(
        crisis=crisis,
        user_choice=user_choice
    )
    
    result = geo_agent.invoke(initial_state)
    
    if result.get("error"):
        raise ValueError(f"Simulation failed: {result['error']}")
    
    if not result.get("final"):
        raise ValueError("No output generated")
    
    return result["final"]

def get_risk_assessment(tension_level: int, stability_index: int) -> str:
    """
    Generate a risk assessment based on tension and stability metrics.
    
    Args:
        tension_level: Global tension level (0-100)
        stability_index: Regional stability index (0-100)
        
    Returns:
        Risk assessment string
    """
    risk_score = tension_level - stability_index
    
    if risk_score > 60:
        return "🔴 CRITICAL RISK - High likelihood of conflict"
    elif risk_score > 30:
        return "🟠 HIGH RISK - Significant tensions present"
    elif risk_score > 0:
        return "🟡 MODERATE RISK - Tensions manageable with diplomacy"
    elif risk_score > -30:
        return "🟢 LOW RISK - Relatively stable situation"
    else:
        return "🔵 MINIMAL RISK - Strong stability indicators"

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Example scenario
        crisis = """
        China has announced a military exercise in the Taiwan Strait with live-fire drills.
        The US has deployed two carrier strike groups to the region. Japan and South Korea
        are on high alert. Taiwan has activated its air defense systems.
        """
        
        user_choice = """
        The US President announces a joint naval exercise with Taiwan, Japan, and South Korea
        in international waters near the Taiwan Strait during China's drills.
        """
        
        result = simulate_geopolitics(crisis, user_choice)
        
        # Display results
        print(f"\n{'='*80}")
        print(f"🌍 GEOPOLITICAL SIMULATION RESULTS")
        print(f"{'='*80}\n")
        
        print(f"📋 CRISIS:")
        print(f"   {result['crisis']}\n")
        
        print(f"⚡ DECISION:")
        print(f"   {result['user_choice']}\n")
        
        print(f"{'─'*80}\n")
        
        print(f"📊 METRICS:")
        print(f"   🌡️  Tension Level:    {result['tension_level']}/100")
        print(f"   ⚖️  Stability Index:  {result['stability_index']}/100")
        print(f"   {get_risk_assessment(result['tension_level'], result['stability_index'])}\n")
        
        print(f"🤝 AFFECTED ALLIANCES:")
        for alliance in result['alliances']:
            print(f"   • {alliance}")
        
        print(f"\n💡 OUTCOME ANALYSIS:")
        print(f"   {result['outcome_summary']}\n")
        
        print(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        print(f"❌ Error: {str(e)}")