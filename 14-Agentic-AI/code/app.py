import streamlit as st
import random
import time
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import TypedDict, Dict, List, Any, Optional
from langchain.prompts.chat import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Constants
END = "END"
DEBATE_TOPICS = [
    "Science vs Religion",
    "Technology vs Tradition",
    "Freedom vs Security",
    "Capitalism vs Socialism",
    "Education vs Experience",
    "Nature vs Nurture",
    "Individual vs Society",
    "Logic vs Emotion"
]

# --- State Management ---
class DebateState(TypedDict):
    messages: list          # full transcript
    next: str               # which agent speaks next
    round: int              # current round
    max_rounds: int         # stop after N rounds
    topic: str              # the debate topic
    sides: dict             # which side each agent defends
    scores: dict            # track scores per agent
    highlights: dict        # best one-liner or argument from each
    history: list           # history of scores for visualization
    current_message: str    # current message being displayed
    debate_finished: bool   # flag to indicate if debate is complete
    evaluator_notes: list   # detailed notes from evaluator

# --- Workflow Engine ---
class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.entry_point = None
    
    def add_node(self, name, func):
        self.nodes[name] = func
    
    def set_entry_point(self, name):
        self.entry_point = name
    
    def get_node(self, name):
        return self.nodes.get(name)

# --- Agent Functions ---
def science_agent(state: DebateState):
    """Agent that argues based on science, logic, and evidence."""
    st.markdown("### 🧪 Science Agent is thinking...")
    
    with st.spinner("Formulating scientific arguments..."):
        llm = OllamaLLM(model='llama3.2', temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a science debater. Defend the side: {state['sides']['science']}. "
                    "Respond with one short, sharp paragraph using scientific reasoning and facts. "
                    "Be persuasive but concise."),
            ("user", "\n".join(state["messages"]))
        ])
        chain = prompt | llm
        response = chain.invoke({"messages": "\n".join(state["messages"])})
    
    message = f"Science: {response}"
    state["messages"].append(message)
    state["current_message"] = message
    
    state["next"] = "religion"
    return state

def religion_agent(state: DebateState):
    """Agent that argues from religious/spiritual perspective."""
    st.markdown("### 🕊️ Religion Agent is thinking...")
    
    with st.spinner("Contemplating spiritual wisdom..."):
        llm = OllamaLLM(model='llama3.2', temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a religious debater. Defend the side: {state['sides']['religion']}. "
                    "Respond with one short, sharp paragraph using religious, moral, or spiritual reasoning. "
                    "Be persuasive but concise."),
            ("user", "\n".join(state["messages"]))
        ])
        chain = prompt | llm
        response = chain.invoke({"messages": "\n".join(state["messages"])})
    
    message = f"Religion: {response}"
    state["messages"].append(message)
    state["current_message"] = message
    
    state["next"] = "evaluator"
    return state

def evaluator_agent(state: DebateState):
    """Evaluator that scores both sides and picks best highlights."""
    st.markdown("### ⚖️ Evaluator is judging...")
    
    with st.spinner("Evaluating arguments..."):
        # Grab last two debater messages
        last_science = [m for m in state["messages"] if m.startswith("Science:")][-1]
        last_religion = [m for m in state["messages"] if m.startswith("Religion:")][-1]
        
        # For now using random scoring with more realistic distribution
        science_score = random.randint(5, 10)
        religion_score = random.randint(5, 10)
        
        # Generate feedback using descriptive language
        science_feedback = f"The argument was {['somewhat weak', 'fairly strong', 'very compelling'][science_score//4]} in its logical structure."
        religion_feedback = f"The argument was {['somewhat weak', 'fairly strong', 'very compelling'][religion_score//4]} in its moral reasoning."
        
        # Add to evaluator notes
        state["evaluator_notes"].append({
            "round": state["round"],
            "science_score": science_score,
            "science_feedback": science_feedback,
            "religion_score": religion_score,
            "religion_feedback": religion_feedback
        })
    
    # Update scores
    state["scores"]["science"] += science_score
    state["scores"]["religion"] += religion_score
    
    # Track history for visualization
    state["history"].append({
        "round": state["round"],
        "science": science_score,
        "religion": religion_score,
        "science_total": state["scores"]["science"],
        "religion_total": state["scores"]["religion"]
    })
    
    # Pick highlight
    if science_score >= religion_score:
        state["highlights"]["science"] = last_science
        highlight = f"Best this round: Science side → {last_science.replace('Science: ', '')}"
    else:
        state["highlights"]["religion"] = last_religion
        highlight = f"Best this round: Religion side → {last_religion.replace('Religion: ', '')}"
    
    # Log judge decision
    decision = (f"Evaluator: Science scored {science_score}/10, "
                f"Religion scored {religion_score}/10. {highlight}")
    state["messages"].append(decision)
    state["current_message"] = decision
    
    # Advance round or end debate
    if state["round"] >= state["max_rounds"]:
        state["next"] = END
        state["debate_finished"] = True
    else:
        state["round"] += 1
        state["next"] = "science"
    
    return state

# --- UI Helpers ---
def display_message(msg):
    """Display a message with appropriate styling."""
    if msg.startswith("Science:"):
        st.markdown(f"<div style='background-color:#f0f7ff;padding:10px;border-radius:5px;margin-bottom:10px;color:#333;'>"
                   f"<strong>🧪 Science:</strong> {msg.replace('Science: ', '')}</div>", unsafe_allow_html=True)
    elif msg.startswith("Religion:"):
        st.markdown(f"<div style='background-color:#fff7f0;padding:10px;border-radius:5px;margin-bottom:10px;color:#333;'>"
                   f"<strong>🕊️ Religion:</strong> {msg.replace('Religion: ', '')}</div>", unsafe_allow_html=True)
    elif msg.startswith("Evaluator:"):
        st.markdown(f"<div style='background-color:#f5f5f5;padding:10px;border-radius:5px;margin-bottom:10px;color:#333;'>"
                   f"<strong>⚖️ Judge:</strong> {msg.replace('Evaluator: ', '')}</div>", unsafe_allow_html=True)
    else:
        st.text(msg)

def create_score_chart(history):
    """Create a chart to visualize scores over rounds."""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Convert Score column to numeric explicitly to avoid type inference issues
    df['science_total'] = pd.to_numeric(df['science_total'], errors='coerce')
    df['religion_total'] = pd.to_numeric(df['religion_total'], errors='coerce')
    
    chart = alt.Chart(df).transform_fold(
        ['science_total', 'religion_total'],
        as_=['Agent', 'value']
    ).mark_line(point=True).encode(
        x='round:O',
        y=alt.Y('value:Q', title='Score'),
        color='Agent:N',
        tooltip=['round', 'value:Q']
    ).properties(
        title='Cumulative Scores by Round'
    )
    
    return chart

def create_round_chart(history):
    """Create a chart to visualize per-round scores."""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Convert score columns to numeric explicitly
    df['science'] = pd.to_numeric(df['science'], errors='coerce')
    df['religion'] = pd.to_numeric(df['religion'], errors='coerce')
    
    chart = alt.Chart(df).transform_fold(
        ['science', 'religion'],
        as_=['Agent', 'value']
    ).mark_bar().encode(
        x='round:O',
        y=alt.Y('value:Q', title='Score'),
        color='Agent:N',
        tooltip=['round', 'value:Q']
    ).properties(
        title='Individual Round Scores'
    )
    
    return chart

def create_radar_chart(state):
    """Create a radar chart comparing debate performance."""
    if not state["evaluator_notes"]:
        return None
        
    # Calculate average scores for different dimensions
    science_avg = state["scores"]["science"] / state["round"]
    religion_avg = state["scores"]["religion"] / state["round"]
    
    # Create synthetic dimension scores for visual interest
    dimensions = ['Persuasiveness', 'Evidence', 'Clarity', 'Impact', 'Overall']
    
    science_scores = [
        min(10, science_avg + random.uniform(-1, 1)),
        min(10, science_avg + random.uniform(-1.5, 1.5)),
        min(10, science_avg + random.uniform(-0.5, 0.5)),
        min(10, science_avg + random.uniform(-1, 1)),
        science_avg
    ]
    
    religion_scores = [
        min(10, religion_avg + random.uniform(-1, 1)),
        min(10, religion_avg + random.uniform(-1.5, 1.5)),
        min(10, religion_avg + random.uniform(-0.5, 0.5)),
        min(10, religion_avg + random.uniform(-1, 1)),
        religion_avg
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=science_scores,
        theta=dimensions,
        fill='toself',
        name='Science'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=religion_scores,
        theta=dimensions,
        fill='toself',
        name='Religion'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True
    )
    
    return fig

# --- Initialize Streamlit App ---
def init_streamlit_app():
    st.set_page_config(
        page_title="AI Debate Arena",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎭 AI Debate Arena")
    st.markdown("""
    Watch as two AI agents debate a topic of your choice! One agent argues from a scientific/logical perspective, 
    while the other presents arguments from a religious/spiritual viewpoint.
    """)
    
    if 'debate_state' not in st.session_state:
        st.session_state.debate_state = DebateState(
            messages=[],
            next="setup",
            round=1,
            max_rounds=3,
            topic="",
            sides={"science": "", "religion": ""},
            scores={"science": 0, "religion": 0},
            highlights={"science": "", "religion": ""},
            history=[],
            current_message="",
            debate_finished=False,
            evaluator_notes=[]
        )
        
        workflow = StateGraph()
        workflow.add_node("science", science_agent)
        workflow.add_node("religion", religion_agent)
        workflow.add_node("evaluator", evaluator_agent)
        workflow.set_entry_point("science")
        st.session_state.workflow = workflow

# --- Main App Logic ---
def run_app():
    init_streamlit_app()
    
    state = st.session_state.debate_state
    workflow = st.session_state.workflow
    
    # Sidebar for settings and controls
    with st.sidebar:
        st.subheader("Debate Configuration")
        
        if state["next"] == "setup" or not state["messages"]:
            topic_option = st.selectbox(
                "Select a debate topic:",
                DEBATE_TOPICS + ["Custom Topic"]
            )
            
            if topic_option == "Custom Topic":
                topic = st.text_input("Enter custom topic (X vs Y format):")
            else:
                topic = topic_option
            
            max_rounds = st.slider("Number of debate rounds:", min_value=1, max_value=10, value=3)
            
            model_option = st.selectbox(
                "Select LLM model:",
                ["llama3.2", "mistral", "gemma"]
            )
            
            temperature = st.slider("Model temperature:", 0.0, 1.0, 0.7, 0.1)
            
            if st.button("Begin Debate", use_container_width=True):
                if topic and " vs " in topic:
                    sides = {
                        "science": topic.split(" vs ")[0],
                        "religion": topic.split(" vs ")[1]
                    }
                    
                    state["topic"] = topic
                    state["sides"] = sides
                    state["max_rounds"] = max_rounds
                    state["next"] = workflow.entry_point
                    state["messages"] = [f"Debate Topic: {topic}"]
                    
                    # Reset state for a new debate
                    state["round"] = 1
                    state["scores"] = {"science": 0, "religion": 0}
                    state["highlights"] = {"science": "", "religion": ""}
                    state["history"] = []
                    state["debate_finished"] = False
                    state["evaluator_notes"] = []
                    
                    st.rerun()
                else:
                    st.error("Please enter a valid topic in 'X vs Y' format")
        else:
            st.info(f"Debate Topic: **{state['topic']}**")
            st.metric("Current Round", f"{state['round']}/{state['max_rounds']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Science Score", state["scores"]["science"])
            with col2:
                st.metric("Religion Score", state["scores"]["religion"])
            
            progress = (state["round"] - 1 + (0 if state["next"] == "science" else 0.5)) / state["max_rounds"]
            st.progress(min(progress, 1.0))
            
            if not state["debate_finished"]:
                if st.button("Next Turn", use_container_width=True):
                    agent_func = workflow.get_node(state["next"])
                    state = agent_func(state)
                    st.session_state.debate_state = state
                    st.rerun()
            else:
                if st.button("Start New Debate", use_container_width=True):
                    state["next"] = "setup"
                    state["messages"] = []
                    st.session_state.debate_state = state
                    st.rerun()
            
            # Show debater info
            st.subheader("Debater Information")
            prof_col1, prof_col2 = st.columns(2)
            
            with prof_col1:
                st.markdown(f"**Science: {state['sides']['science']}**")
                st.caption("Uses scientific reasoning and evidence")
            
            with prof_col2:
                st.markdown(f"**Religion: {state['sides']['religion']}**")
                st.caption("Uses moral and spiritual reasoning")
    
    # Main content area
    if state["next"] == "setup" or not state["messages"]:
        # Welcome screen
        st.markdown("## Welcome to the AI Debate Arena!")
        
        # Display sample topics
        st.markdown("### Popular Debate Topics")
        
        cols = st.columns(2)
        for i, topic in enumerate(DEBATE_TOPICS[:6]):
            with cols[i % 2]:
                st.info(topic)
        
        # How it works
        st.markdown("### How It Works")
        st.markdown("""
        1. Select a debate topic or create your own
        2. Set the number of debate rounds
        3. Watch as the AI agents debate your topic
        4. The evaluator will score each round
        5. See who wins at the end!
        """)
        
        # Features overview
        st.markdown("### Features")
        feature_cols = st.columns(3)
        with feature_cols[0]:
            st.success("🤖 AI Agents with different perspectives")
        with feature_cols[1]:
            st.success("📊 Real-time scoring and visualization")
        with feature_cols[2]:
            st.success("📝 Debate transcript export")
        
    else:
        # Debate interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"## Debate: {state['topic']}")
            
            message_container = st.container()
            with message_container:
                for msg in state["messages"]:
                    display_message(msg)
            
            if not state["debate_finished"] and state["next"] != "setup":
                next_agent = "Science" if state["next"] == "science" else "Religion" if state["next"] == "religion" else "Evaluator"
                st.markdown(f"*{next_agent} is preparing response...*")
        
        with col2:
            # Visualization area
            if state["history"]:
                st.subheader("Debate Statistics")
                
                tab1, tab2, tab3 = st.tabs(["Scores", "Per Round", "Performance"])
                
                with tab1:
                    score_chart = create_score_chart(state["history"])
                    if score_chart:
                        st.altair_chart(score_chart, use_container_width=True)
                
                with tab2:
                    round_chart = create_round_chart(state["history"])
                    if round_chart:
                        st.altair_chart(round_chart, use_container_width=True)
                
                with tab3:
                    radar_chart = create_radar_chart(state)
                    if radar_chart:
                        st.plotly_chart(radar_chart, use_container_width=True)
                
                # Evaluator notes
                if state["evaluator_notes"]:
                    st.subheader("Judge's Notes")
                    for note in state["evaluator_notes"]:
                        with st.expander(f"Round {note['round']} Evaluation"):
                            st.markdown(f"**Science ({note['science_score']}/10):** {note['science_feedback']}")
                            st.markdown(f"**Religion ({note['religion_score']}/10):** {note['religion_feedback']}")
    
    # Final results
    if state["debate_finished"]:
        st.markdown("---")
        st.header("🏆 Final Results")
        
        winner_col1, winner_col2, winner_col3 = st.columns([1, 2, 1])
        
        with winner_col2:
            if state["scores"]["science"] > state["scores"]["religion"]:
                st.success(f"🎉 WINNER: Science side ({state['sides']['science']})")
            elif state["scores"]["science"] < state["scores"]["religion"]:
                st.success(f"🎉 WINNER: Religion side ({state['sides']['religion']})")
            else:
                st.info("🤝 DRAW! Both sides presented equally compelling arguments.")
        
        # Display highlights
        st.subheader("Best Arguments")
        highlight_col1, highlight_col2 = st.columns(2)
        
        with highlight_col1:
            st.markdown("##### Science Highlight")
            if state["highlights"]["science"]:
                st.info(state["highlights"]["science"].replace("Science: ", ""))
        
        with highlight_col2:
            st.markdown("##### Religion Highlight")
            if state["highlights"]["religion"]:
                st.info(state["highlights"]["religion"].replace("Religion: ", ""))
        
        # Summary stats
        st.subheader("Debate Summary")
        
        summary_data = {
            "Metric": ["Final Score", "Best Round Score", "Total Arguments"],
            "Science": [
                state["scores"]["science"], 
                max([h["science"] for h in state["history"]]) if state["history"] else 0,
                len([m for m in state["messages"] if m.startswith("Science:")])
            ],
            "Religion": [
                state["scores"]["religion"],
                max([h["religion"] for h in state["history"]]) if state["history"] else 0,
                len([m for m in state["messages"] if m.startswith("Religion:")])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Export options
        st.download_button(
            label="Export Debate Transcript",
            data="\n\n".join(state["messages"]),
            file_name=f"debate_{state['topic'].replace(' ', '_')}.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.caption("AI Debate Arena | Created with Streamlit and Langchain")

# Run the app
if __name__ == "__main__":
    run_app()
