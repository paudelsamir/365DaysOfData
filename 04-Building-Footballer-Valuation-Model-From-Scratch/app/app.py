import streamlit as st
import pickle
import pandas as pd
import json
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
foot_encoder = pickle.load(open('foot_encoder.pkl', 'rb'))
position_encoder = pickle.load(open('position_encoder.pkl', 'rb'))
position_category_encoder = pickle.load(open('position_category_encoder.pkl', 'rb'))

with open('team_target_encoding.json', 'r') as f:
    team_target_encoding = json.load(f)

# get team names from encoding
team_names = list(team_target_encoding.keys())

def encode_position_category(category):
    categories = ['Defender', 'Forward', 'Goalkeeper', 'Midfielder']
    return [1 if cat == category else 0 for cat in categories]

st.title('Football Players Market Value Prediction')
st.write("Enter the player's details to predict their market value.")

#  Basic Information
st.header('Basic Information')
col1, col2, col3 = st.columns(3)
with col1:
    position_category = st.selectbox('Position Category', ['Attacker', 'Midfielder', 'Defender', 'Goalkeeper'], index=1)  # Midfielder as default
with col2:
    age = st.number_input('Age', 18, 40, 30) 
with col3:
    team = st.selectbox('Team', team_names, index=team_names.index('Manchester City'))

#  Ratings
st.header('Ratings')
col1, col2, col3 = st.columns(3)
with col1:
    best_overall = st.number_input('Best Overall', 50, 100, 92) 
with col2:
    overall_rating = st.number_input('Overall Rating', 50, 100, 91)  
with col3:
    potential = st.number_input('Potential', 50, 100, 90) 

# Physical Attributes
st.header('Physical Attributes')
col1, col2, col3 = st.columns(3)
with col1:
    height = st.number_input('Height (cm)', 150, 220, 180)  
    weight = st.number_input('Weight (kg)', 50, 120, 72) 
    foot = st.selectbox('Foot', ['Right', 'Left'], index=0)  
with col2:
    acceleration = st.number_input('Acceleration', 1, 100, 75) 
    sprint_speed = st.number_input('Sprint Speed', 1, 100, 80) 
    agility = st.number_input('Agility', 1, 100, 85) 
    balance = st.number_input('Balance', 1, 100, 85) 
    stamina = st.number_input('Stamina', 1, 100, 85)  
    strength = st.number_input('Strength', 1, 100, 75)  
st.header('Position-Specific Skills')
input_features = {
    'crossing': 0, 'finishing': 0, 'heading_accuracy': 0,
    'short_passing': 0, 'volleys': 0, 'dribbling': 0,
    'curve': 0, 'fk_accuracy': 0, 'long_passing': 0,
    'ball_control': 0, 'defensive_awareness': 0,
    'standing_tackle': 0, 'sliding_tackle': 0,
    'interceptions': 0, 'aggression': 0,
    'vision': 0, 'composure': 0
}

if position_category == 'Attacker':
    input_features['crossing'] = st.number_input('Crossing', 1, 100, 60)
    input_features['finishing'] = st.number_input('Finishing', 1, 100, 75)
    input_features['heading_accuracy'] = st.number_input('Heading Accuracy', 1, 100, 70)
    input_features['short_passing'] = st.number_input('Short Passing', 1, 100, 75)
    input_features['volleys'] = st.number_input('Volleys', 1, 100, 65)
    input_features['dribbling'] = st.number_input('Dribbling', 1, 100, 80) 
    input_features['curve'] = st.number_input('Curve', 1, 100, 80) 
    input_features['fk_accuracy'] = st.number_input('FK Accuracy', 1, 100, 70)
    input_features['ball_control'] = st.number_input('Ball Control', 1, 100, 85) 
elif position_category == 'Midfielder':
    input_features['crossing'] = st.number_input('Crossing', 1, 100, 75)
    input_features['short_passing'] = st.number_input('Short Passing', 1, 100, 90) 
    input_features['long_passing'] = st.number_input('Long Passing', 1, 100, 85) 
    input_features['vision'] = st.number_input('Vision', 1, 100, 90)  
    input_features['composure'] = st.number_input('Composure', 1, 100, 85)
    input_features['curve'] = st.number_input('Curve', 1, 100, 85)
    input_features['fk_accuracy'] = st.number_input('FK Accuracy', 1, 100, 75) 
    input_features['ball_control'] = st.number_input('Ball Control', 1, 100, 90)

elif position_category == 'Defender':
    input_features['defensive_awareness'] = st.number_input('Defensive Awareness', 1, 100, 75)
    input_features['standing_tackle'] = st.number_input('Standing Tackle', 1, 100, 70)
    input_features['sliding_tackle'] = st.number_input('Sliding Tackle', 1, 100, 70)
    input_features['interceptions'] = st.number_input('Interceptions', 1, 100, 70)
    input_features['aggression'] = st.number_input('Aggression', 1, 100, 65)
    input_features['long_passing'] = st.number_input('Long Passing', 1, 100, 75)

else:  # Goalkeeper
    gk_diving = st.number_input('GK Diving', 1, 100, 70)
    gk_handling = st.number_input('GK Handling', 1, 100, 70)
    gk_kicking = st.number_input('GK Kicking', 1, 100, 75)
    gk_positioning = st.number_input('GK Positioning', 1, 100, 80)
    gk_reflexes = st.number_input('GK Reflexes', 1, 100, 85)


total_attacking = (input_features['crossing'] + input_features['finishing'] + 
                  input_features['heading_accuracy'] + input_features['short_passing'] + 
                  input_features['volleys'])

total_skill = (input_features['dribbling'] + input_features['curve'] + 
              input_features['fk_accuracy'] + input_features['ball_control'])

total_defending = (input_features['defensive_awareness'] + input_features['standing_tackle'] + 
                  input_features['sliding_tackle'] + input_features['interceptions'])

total_mentality = (input_features['vision'] + input_features['composure'] + 
                  input_features['aggression'])

input_data = {
    'Age': age,
    'Best overall': best_overall,
    'Overall rating': overall_rating,
    'Potential': potential,
    'foot': 1 if foot == "Right" else 0,  # manually encode
    # 'Wage': np.log1p(float(wage) + 1) if wage > 0 else 0,  # safe log transformation with float conversion
    'Height_cm': height,
    'Weight_kg': weight,
    'Acceleration': acceleration,
    'Sprint speed': sprint_speed,
    'Agility': agility,
    # 'Reactions': reactions,
    'Balance': balance,
    'Stamina': stamina,
    'Strength': strength,
    # 'Jumping': jumping,
    'Total attacking': total_attacking,
    'Total skill': total_skill,
    'Total defending': total_defending,
    'Total mentality': total_mentality,
    'Team_encoded': team_target_encoding.get(team, 0),
    **dict(zip(['Position Category_Defender', 'Position Category_Forward', 
               'Position Category_Goalkeeper', 'Position Category_Midfielder'],
              encode_position_category(position_category)))
}


input_df = pd.DataFrame([input_data])

missing_features = set(model.feature_names_in_) - set(input_df.columns)
for feature in missing_features:
    input_df[feature] = 0

input_df = input_df[model.feature_names_in_]

if st.button('Predict Market Value'):
    prediction = model.predict(input_df)
    predicted_value = np.exp(prediction[0])
    st.success(f'Predicted Market Value: €{predicted_value:,.2f}')
    