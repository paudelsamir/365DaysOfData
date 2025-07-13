import streamlit as st
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from torchvision import models
from torch import nn
from torchvision.datasets import ImageFolder

# Set page configuration
st.set_page_config(page_title="Guess the footballer by eyes", layout="wide")

# Load your trained model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 25)
    model.load_state_dict(torch.load('eye_classifier_resnet18.pth', map_location='cpu'))
    model.eval()
    return model

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Helper function to convert tensor to numpy for display
def image_to_numpy(img):
    img = img.permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Load test dataset
@st.cache_resource
def load_test_data():
    test_data = ImageFolder('./eye_dataset/final/test', transform=transform)
    return test_data

# Game logic
def play_game(model, test_images, class_names):
    # Setup page header with custom styling
    st.markdown("""
    <style>
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: left;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    </style>
    <div class="big-title">Guess the Footballer by their Eyes</div>
    """, unsafe_allow_html=True)
    
    st.write("Can you recognize footballers better than AI by looking only at their eyes?")
    
    # Initialize game state
    if 'game_state' not in st.session_state:
        st.session_state.game_state = {
            'round': 1,
            'player_score': 0,
            'ai_score': 0,
            'current_images': [],
            'current_labels': [],
            'streak': 0,
            'best_streak': 0,
            'game_over': False,
            'result_shown': False
        }
    
    gs = st.session_state.game_state
    
    # Game settings sidebar
    with st.sidebar:
        st.subheader("Game Settings")
        
        rounds_to_play = st.slider("Rounds to play", 3, 10, 5)
        
        # Start/restart game
        st.markdown("**Difficulty:** Beginner (more coming soon!)")
        if st.button("Start New Game", type="primary"):
            selected_indices = random.sample(range(len(test_images)), rounds_to_play)
            gs['current_images'] = [test_images[i][0] for i in selected_indices]
            gs['current_labels'] = [test_images[i][1] for i in selected_indices]
            gs['round'] = 1
            gs['player_score'] = 0
            gs['ai_score'] = 0
            gs['streak'] = 0
            gs['best_streak'] = 0
            gs['game_over'] = False
            gs['result_shown'] = False
            st.rerun()
        
        st.markdown("---")
        with st.expander("Game Breakdown"):
            st.markdown("""
            ### Footballer Eye Recognition Game - Limited Edition
            
            This game tests your ability to recognize footballers by their eyes alone! Can you beat the AI?
            
            **How to Play:**
            1. Look at the footballer's eyes
            2. Select who you think it is from the dropdown
            3. Submit your guess
            4. Try to achieve a higher score than the AI!
            
            **Model Training Info:**
            - Used ResNet18 architecture fine-tuned on footballer eye images
            - Trained on dataset of 25 famous footballers
            - Test accuracy: ~70%
            
            Challenge yourself and see if human recognition can outperform machine learning!
            
            
            Author - [**@samireey**](https://twitter.com/samireey)  
            """)
    
    # Initialize game if needed
    if len(gs.get('current_images', [])) == 0:
        st.info("Press 'Start New Game' in the sidebar to begin!")
        return

    # First check if game is over - show ONLY game over screen when game ends
    if gs['game_over']:
        st.markdown(
            "<hr style='border-top: 3px dashed #1E88E5; margin: 2rem 0;'>",
            unsafe_allow_html=True
        )
        st.header("🏆 Game Over!")
        
        # Simple score display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Score", gs['player_score'])
        with col2:
            st.metric("Best Streak", gs['best_streak'])
        with col3:
            st.metric("AI Score", gs['ai_score'])
        
        # Simple results message
        if gs['player_score'] > gs['ai_score']:
            st.balloons()
            st.success(f"You win! {gs['player_score']} - {gs['ai_score']}")
        elif gs['ai_score'] > gs['player_score']:
            st.error(f"AI wins! {gs['ai_score']} - {gs['player_score']}")
        else:
            st.warning(f"It's a tie! {gs['player_score']} - {gs['ai_score']}")
        
        # To play again, instruct user to use sidebar button
        st.markdown(
            "<br><b>To play again, please click <span style='color:#1E88E5;'>Start New Game</span> in the sidebar.</b>",
            unsafe_allow_html=True
        )
        
        return  # Exit the function to prevent showing game interface
    
    # Display scoreboard - only shown when game is not over
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Player Score", gs['player_score'])
    with col2:
        st.metric("Round", f"{gs['round']}/{len(gs['current_images'])}")
    with col3:
        st.metric("AI Score", gs['ai_score'])
    with col4:
        st.metric("Current Streak", gs['streak'], delta=f"Best: {gs['best_streak']}")
    
    # Main game area - only shown when game is not over
    if gs['round'] <= len(gs['current_images']):
        # Get current round data
        current_idx = gs['round'] - 1
        image = gs['current_images'][current_idx]
        true_label = gs['current_labels'][current_idx]
        
        # Display eye image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image_to_numpy(image), caption="Whose eyes are these?", width=350)
        
            # Player makes prediction through dropdown with a search option
            player_prediction = st.selectbox(
                "Who do you think this is?", 
                options=class_names,
                index=0,
                placeholder="Type to search..."
            )
            player_prediction_idx = class_names.index(player_prediction)
            
            # Submit button
            if st.button("Submit Guess", type="primary"):
                # Get AI prediction now (after user guesses)
                with torch.no_grad():
                    output = model(image.unsqueeze(0))
                    probs = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()
                    ai_prediction = np.argmax(probs)
                
                # Update scores
                player_correct = player_prediction_idx == true_label
                ai_correct = ai_prediction == true_label
                
                if player_correct:
                    gs['player_score'] += 1
                    gs['streak'] += 1
                    gs['best_streak'] = max(gs['streak'], gs['best_streak'])
                    st.success(f"Correct! +1 point (Streak: {gs['streak']}) ✅")
                else:
                    gs['streak'] = 0
                    st.error(f"Wrong! This was {class_names[true_label]} ❌")
                
                if ai_correct:
                    gs['ai_score'] += 1
                    st.info(f"AI guessed correctly! It said {class_names[ai_prediction]} ✅")
                else:
                    st.info(f"AI was wrong! It guessed {class_names[ai_prediction]} ❌")
                
                # Add a prominent "Continue to Next Round" button at the top of results
                next_round_col1, next_round_col2 = st.columns([1, 2])
                with next_round_col1:
                    if st.button("NEXT ROUND➡️", type="primary"):
                        gs['result_shown'] = False
                        st.rerun()
                
                # Show AI thought process in an expander to save space
                with st.expander("See AI's Thinking Process", expanded=False):
                    st.subheader("AI's Thinking Process:")
                    top3_idx = probs.argsort()[-3:][::-1]
                    top_probs = {class_names[i]: probs[i]*100 for i in top3_idx}
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.bar_chart(pd.Series(top_probs))
                    with col2:
                        st.write("AI top predictions:")
                        for i, idx in enumerate(top3_idx):
                            st.write(f"{i+1}. {class_names[idx]}: {probs[idx]*100:.1f}%")
                
                gs['round'] += 1
                gs['result_shown'] = True
                
                # Check if game over after updating round
                if gs['round'] > len(gs['current_images']):
                    gs['game_over'] = True
                    st.rerun()  # Force rerun to show game over screen

# Main
if __name__ == "__main__":
    model = load_model()
    test_data = load_test_data()
    play_game(model, test_data, test_data.classes)