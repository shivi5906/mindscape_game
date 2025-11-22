from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import json
import logging

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class TimelineStep(BaseModel):
    """Individual step in the alternate history timeline."""
    year: str = Field(..., description="Year or time period of the event")
    title: str = Field(..., description="Brief title of what changed")
    impact: str = Field(..., description="Detailed impact and consequences")

class HistoryOutput(BaseModel):
    """Complete output structure for the alternate history timeline."""
    event: str = Field(..., description="Original historical event")
    user_change: str = Field(..., description="The hypothetical change made")
    timeline: List[TimelineStep] = Field(..., min_items=1, description="Chronological timeline of changes")

class HistoryState(BaseModel):
    """State management for the LangGraph workflow."""
    event: str
    user_change: str
    raw_response: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    final_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# --- Initialize LLM ---
def initialize_llm():
    """Initialize the Gemini LLM with configuration."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_output_tokens=2048,
    )

llm = initialize_llm()

# --- Prompt Template ---
history_prompt = PromptTemplate(
    input_variables=["event", "user_change"],
    template="""You are an expert geopolitical historian and alternate history analyst.

Your task is to generate a plausible alternate historical timeline based on a changed event.

**Original Event:** {event}
**Hypothetical Change:** {user_change}

Create a detailed alternate timeline showing how this change would have cascading effects through history. Consider:
- Immediate consequences (0-5 years)
- Medium-term impacts (5-20 years)
- Long-term ramifications (20+ years)
- Political, economic, social, and technological effects

Return your response as valid JSON with this exact structure:
{{
    "event": "{event}",
    "user_change": "{user_change}",
    "timeline": [
        {{
            "year": "1945-1950",
            "title": "Immediate Post-War Restructuring",
            "impact": "Detailed description of what changed and why it matters..."
        }},
        {{
            "year": "1950-1970",
            "title": "Next phase title",
            "impact": "Detailed impact explanation..."
        }}
    ]
}}

Provide at least 4-6 timeline steps for a comprehensive alternate history. Be specific and historically grounded."""
)

# --- LangGraph Nodes ---
def prompt_node(state: HistoryState) -> HistoryState:
    """Generate the alternate history timeline using LLM."""
    try:
        logger.info(f"Generating timeline for event: {state.event}")
        chain = history_prompt | llm
        response = chain.invoke({
            "event": state.event,
            "user_change": state.user_change
        })
        
        # Extract content from response
        state.raw_response = response.content
        logger.info("Successfully generated timeline")
        
    except Exception as e:
        logger.error(f"Error in prompt_node: {str(e)}")
        state.error = f"Generation error: {str(e)}"
    
    return state

def parse_node(state: HistoryState) -> HistoryState:
    """Parse the JSON response from the LLM."""
    if state.error:
        return state
    
    try:
        # Clean the response if it has markdown code blocks
        content = state.raw_response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        state.parsed_json = json.loads(content)
        logger.info("Successfully parsed JSON response")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        state.error = f"Invalid JSON response: {str(e)}"
    except Exception as e:
        logger.error(f"Error in parse_node: {str(e)}")
        state.error = f"Parsing error: {str(e)}"
    
    return state

def validate_node(state: HistoryState) -> HistoryState:
    """Validate the parsed JSON against the Pydantic model."""
    if state.error:
        return state
    
    try:
        # Validate with Pydantic
        validated = HistoryOutput(**state.parsed_json)
        state.final_output = validated.model_dump()
        logger.info(f"Successfully validated output with {len(validated.timeline)} timeline steps")
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        state.error = f"Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Error in validate_node: {str(e)}")
        state.error = f"Validation error: {str(e)}"
    
    return state

# --- Build LangGraph ---
def build_graph() -> StateGraph:
    """Construct the LangGraph workflow."""
    graph = StateGraph(HistoryState)
    
    # Add nodes
    graph.add_node("prompt", prompt_node)
    graph.add_node("parse", parse_node)
    graph.add_node("validate", validate_node)
    
    # Set up workflow
    graph.set_entry_point("prompt")
    graph.add_edge("prompt", "parse")
    graph.add_edge("parse", "validate")
    graph.add_edge("validate", END)
    
    return graph.compile()

# Compile the graph
history_agent = build_graph()

# --- Main Function ---
def generate_history_timeline(event: str, user_change: str) -> Dict[str, Any]:
    """
    Generate an alternate history timeline based on a changed event.
    
    Args:
        event: The original historical event
        user_change: The hypothetical change to that event
        
    Returns:
        Dictionary containing the alternate history timeline
        
    Raises:
        ValueError: If generation fails or output is invalid
    """
    if not event or not user_change:
        raise ValueError("Both 'event' and 'user_change' must be provided")
    
    logger.info(f"Starting timeline generation for: {event}")
    
    initial_state = HistoryState(
        event=event,
        user_change=user_change
    )
    
    result = history_agent.invoke(initial_state)
    
    if result.get("error"):
        raise ValueError(f"Timeline generation failed: {result['error']}")
    
    if not result.get("final_output"):
        raise ValueError("No output generated")
    
    return result["final_output"]

# --- Example Usage ---
if __name__ == "__main__":
    try:
        timeline = generate_history_timeline(
            event="The assassination of Archduke Franz Ferdinand in 1914",
            user_change="What if the assassination attempt had failed?"
        )
        
        print(f"\n{'='*60}")
        print(f"EVENT: {timeline['event']}")
        print(f"CHANGE: {timeline['user_change']}")
        print(f"{'='*60}\n")
        
        for step in timeline['timeline']:
            print(f"📅 {step['year']}: {step['title']}")
            print(f"   {step['impact']}\n")
            
    except Exception as e:
        logger.error(f"Failed to generate timeline: {str(e)}")
        print(f"Error: {str(e)}")