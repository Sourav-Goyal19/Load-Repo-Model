import os
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import Dict, List, Any
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from dataset import dataset

# llm = ChatGroq(
#     temperature=0,
#     model_name="llama-3.3-70b-versatile",
#     api_key="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",
# )

llm = ChatAnthropic(
    model="claude-3-5-sonnet-latest",
    temperature=0,
    api_key="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",
)

# llm = ChatGoogleGenerativeAI(
#     model=("gemini-1.5-pro"), api_key="AIzaSyAjJ5MiuN7D4stHeErl_6qeRq7FYN2yIqM"
# )


class State(BaseModel):
    user_input: str = ""
    algorithm: str = ""
    steps: List[str] = []
    current_step_index: int = 0
    current_step_attempt: str = ""
    final_code: List[str] = []
    integrated_code: str = ""


identify_algorithm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert ML algorithm identification system. Your role is to:
            1. Analyze user input to identify the most appropriate machine learning algorithm
            2. Consider factors like:
            - Problem type (classification, regression, clustering, etc.)
            - Data characteristics mentioned
            - Performance requirements
            - Implementation constraints
            3. Return ONLY the algorithm name without explanation
            4. Stick to well-established algorithms that can be effectively visualized""",
        ),
        (
            "human",
            """Based on this user request, identify the single most appropriate ML algorithm.
            If multiple algorithms could work, choose the most fundamental one that's easier to visualize.
            Request: {user_input}""",
        ),
    ]
)

plan_explanation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in breaking down machine learning algorithms into clear, visualizable steps.
            For algorithm {algorithm}, create steps that:
            1. Start with data preparation/input
            2. Include all key mathematical transformations
            3. Show how the algorithm processes data iteratively
            4. Conclude with output/prediction generation
            5. Focus on aspects that can be effectively animated
            Each step should be concrete and visualizable.""",
        ),
        (
            "human",
            """Create a sequence of visualization-friendly steps for {algorithm}. Requirements:
            - Each step should represent a distinct visual state
            - Include data transformations that can be animated
            - Focus on geometric and mathematical operations
            - Ensure steps flow logically from input to output
            - Return ONLY numbered steps, one per line""",
        ),
    ]
)

process_step_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Manim expert specializing in ML algorithm visualization.
            When generating code for {algorithm}, ensure you:
            1. Use appropriate Manim objects (Axes, Dots, Lines, etc.)
            2. Create smooth animations between states
            3. Add clear labels and annotations
            4. Use consistent color schemes
            5. Implement proper scaling and positioning
            6. Handle edge cases gracefully
            **IMPORTANT NOTE** - Use the following examples to learn the new upgraded syntax of manim = {dataset}
            """,
        ),
        (
            "human",
            """Generate production-quality Manim code for this step of {algorithm}:
            Step to visualize: {current_step}

            Requirements:
            - Include all necessary imports
            - Create a complete Scene class
            - Use meaningful variable names
            - Add comments explaining complex animations
            - Ensure proper timing between animations
            - Always use your own custom random data
            - Return ONLY the implementation code""",
        ),
    ]
)

integrate_code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Manim expert tasked with integrating multiple visualization steps into a single cohesive animation.
            Given the algorithm {algorithm} and the user query "{user_input}", combine the following step codes into one Manim script:
            - Ensure all imports are included only once at the top
            - Create a single Scene class that flows logically from one step to the next
            - Maintain continuity between steps (e.g., reuse data, objects, and transformations where appropriate)
            - Use smooth transitions and consistent styling
            - Add comments to separate and explain each step's section
            - Handle timing to ensure a clear, paced animation
            Return ONLY the complete integrated code.
            **IMPORTANT NOTE** - Use the following examples to learn the new upgraded syntax of manim = {dataset}
            """,
        ),
        (
            "human",
            """Integrate these step codes for {algorithm} based on the query "{user_input}":
            Step codes:
            {step_codes}

            Return the fully integrated Manim script.""",
        ),
    ]
)


def identify_algorithm(state: State) -> State:
    chain = identify_algorithm_prompt | llm
    state.algorithm = chain.invoke({"user_input": state.user_input}).content.strip()
    return state


def plan_explanation(state: State) -> State:
    chain = plan_explanation_prompt | llm
    response = chain.invoke({"algorithm": state.algorithm}).content.strip()
    steps = [step.strip() for step in response.split("\n") if step.strip()]
    state.steps = steps
    print(f"Generated steps: {steps}")
    return state


def process_step(state: State) -> State:
    print(
        f"Processing step {state.current_step_index}, Total steps: {len(state.steps)}"
    )
    chain = process_step_prompt | llm
    current_step = state.steps[state.current_step_index]
    state.current_step_attempt = chain.invoke(
        {
            "algorithm": state.algorithm,
            "current_step": current_step,
            "dataset": dataset[-6:],
        }
    ).content.strip()
    state.final_code.append(state.current_step_attempt)
    print(
        f"\nStep {state.current_step_index + 1} visualization code:\n{state.current_step_attempt}"
    )
    state.current_step_index += 1
    return state


def integrate_steps(state: State) -> State:
    chain = integrate_code_prompt | llm
    step_codes = "\n\n".join(
        [f"Step {i+1}:\n{code}" for i, code in enumerate(state.final_code)]
    )
    state.integrated_code = chain.invoke(
        {
            "algorithm": state.algorithm,
            "user_input": state.user_input,
            "step_codes": step_codes,
            "dataset": dataset[-1:],
        }
    ).content.strip()
    return state


def should_continue_steps(state: State) -> str:
    print(
        f"Checking continuation: index={state.current_step_index}, steps={len(state.steps)}"
    )
    if state.current_step_index < len(state.steps):
        return "process_step"
    return "integrate_steps"


workflow = StateGraph(State)
workflow.add_node("identify_algorithm", identify_algorithm)
workflow.add_node("plan_explanation", plan_explanation)
workflow.add_node("process_step", process_step)
workflow.add_node("integrate_steps", integrate_steps)

workflow.set_entry_point("identify_algorithm")
workflow.add_edge("identify_algorithm", "plan_explanation")
workflow.add_edge("plan_explanation", "process_step")
workflow.add_conditional_edges(
    "process_step",
    should_continue_steps,
    {"process_step": "process_step", "integrate_steps": "integrate_steps"},
)
workflow.add_edge("integrate_steps", END)


def run_workflow(user_input: str) -> Dict[str, Any]:
    graph = workflow.compile()
    initial_state = State(
        user_input=user_input,
        algorithm="",
        steps=[],
        current_step_index=0,
        current_step_attempt="",
        final_code=[],
        integrated_code="",
    )
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    try:
        result = run_workflow("Explain linear regression")
        print("Algorithm:", result["algorithm"])
        print("\nSteps:")
        for i, step in enumerate(result["steps"]):
            print(f"{i+1}. {step}")
        print("\nGenerated Code (Summary):")
        for i, code in enumerate(result["final_code"]):
            print(f"\nStep {i+1} visualization code:\n{code}")
        print("\nIntegrated Visualization Code:")
        print(result["integrated_code"])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
