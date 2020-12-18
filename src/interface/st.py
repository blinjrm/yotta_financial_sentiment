import streamlit as st

# Text/Tutorial
st.title("Streamlit tutorial")

# Header/Subheader
st.header("Voici mon titre...")
st.subheader("...et mon sous-titre")

# Text
st.text("salut les potos")

# Markdown
st.markdown("##### test markdown")

# Error/Colorful text
st.success("Successful")
st.info("information!")
st.warning("This is a warning")
st.error("this is an error")
st.exception('NameError("namethree not defined")')

# Get help info about python
st.help(range)

# Writing text
st.write("text with write")
st.write(range(10))

# Images
from PIL import Image

img = Image.open("screen-5.jpeg")
st.image(img, width=600, caption="Simple Image")

# Videos
vid_file = open("automatic-machine-learning.mp4", "rb").read()
# vid_bytes = vid_file.read()
st.video(vid_file)

# Audio
# audio_file = open('examplemusic.mp3','rb').read()
# st.audio(audio_file,format='audio/mp3')

# Widget
# Checkbox
if st.checkbox("Show/Hide"):
    st.text("Showing or Hiding Widget")

# Radio Buttons
status = st.radio("What is your status", ("Active", "Inactive"))

if status == "Active":
    st.success("You are Active")
else:
    st.warning("Inactive", "Activate")

# SelectBox
occupation = st.selectbox("Your Occupation", ["Data Scientist", "Footballer", "Chomeur"])
st.write("You selected this option ", occupation)

# MultiSelect
location = st.multiselect("Where do you work?", ("London", "Paris", "Canardville"))
st.write("You selected", len(location), "locations")

# Slider
age = st.slider("What is your age", 18, 107)

# Buttons
st.button("Simple Buttons")

if st.button("About"):
    st.text("Streamlite is coooool")

# Text Input
firstname = st.text_input("Enter your name", "Type here...")
if st.button("Submit"):
    result = firstname.title()
    st.success(result)
