import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Streamlit UI
st.set_page_config(page_title='Cat or Dog?', page_icon='🔍')


# Load the model
model = tf.keras.models.load_model('trained_model.h5')
class_labels = ['Cat', 'Dog']


st.markdown("### **Is it a cat🐱? Is it a dog🐶?**")


# Upload the image
uploaded_file = st.file_uploader("*just upload a picture of cat or dog and let ai predict !!!*", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test_img = cv2.imdecode(file_bytes, 1)
    
    # Convert BGR to RGB
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # st.markdown("#### **🔍 Just a sec..**")
    
    # Add a funny effect (replacing the spinner with a wacky text)
    with st.spinner('🔍 Just a sec..'):
        st.markdown(
            """
            <audio id="funny-audio" autoplay>
                <source src="https://www.myinstants.com/media/sounds/vine-boom.mp3" type="audio/mp3">
            </audio>
            <script>
                document.getElementById("funny-audio").play();
            </script>
            """,
            unsafe_allow_html=True
)

        
        # Resize and predict
        test_img_resized = cv2.resize(test_img_rgb, (150, 150))
        test_input = test_img_resized.reshape((1, 150, 150, 3)) / 255.0
        prediction = model.predict(test_input)
    
    # Get result
    predicted_class_idx = int(prediction[0][0] > 0.5)
    predicted_class = class_labels[predicted_class_idx]
    
    # Display result
    st.image(test_img_rgb, caption="who are you ?", use_column_width=True)
    
    if predicted_class == 'Cat':
        st.balloons()
        st.markdown(
            f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #ffcccb; text-align: center; font-size: 24px; font-weight: bold; color: black;">
                AI says it’s a <span style="color:#28a745;">{predicted_class}!</span> 🐱
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.balloons()
        st.markdown(
            f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #d4edda; text-align: center; font-size: 24px; font-weight: bold; color: black;">
                AI says it’s a <span style="color:#28a745;">{predicted_class}!</span> 🐶
            </div>
            """, 
            unsafe_allow_html=True
        )

else:
    st.info("No image? Upload something random- your ex's pic, your last screenshot, or even your friend's face. Let's confuse the AI together! 😂")


st.markdown("-------------------")
st.markdown("*Crafted by: [Samir](https://www.twitter.com/samireey), expert in building pointless projects*")
