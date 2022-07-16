import streamlit as st
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
from utils import load_and_prep, get_classes
import base64


st.set_page_config(page_title="Food Vision 101", page_icon="🍉")

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()}); 
            -webkit-background-size: cover;
            -moz-background-size: cover;
            -o-background-size: cover;
            background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack("./images/bck.jpg")

                   
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        }
    </style>
    ''' % bin_str
    
    return st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg("./images/bck.jpg")

@st.cache(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

class_names = get_classes()


#### SideBar ####

st.sidebar.title("What's Food Vision 101 ?")
st.sidebar.write("""
Food Vision 101 is an end-to-end **multi-class image classification system** which identifies the kind of food in an image.

It is implemented using the popular **CNN network** while utilizing the full power of **feature extraction**,**fine tuning** and **transfer learning** to extract features and fine-tune layer. 

It can identify over 101 different food classes

It is based upon a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.

Deployed on Streanlit app

**Accuracy :** **`82%`**

**Model :** **`EfficientNetB1`**

**Dataset :** **`Food101`**
""")


#### Main Body ####

st.title("Food Vision 101 🍔 🥘 🍲 📷 🔭")
st.header("Curious!! to know what kind of food is in an image? Let food vision 101 help you out.")
st.write("To know more about this app, visit [**GitHub**](https://github.com/princewilling)")
file = st.file_uploader(label="Upload an image of food.",
                        type=["jpg", "jpeg", "png"])


model = tf.keras.models.load_model("./models/FinalModel.hdf5")


st.sidebar.markdown("Created by **Princewill Inyang**")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://linkedin.com/in/princewill-inyang-6b07021b0" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="gauravreddy08" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/princewilling" target="blank"><img align="center" src="https://cdn4.iconfinder.com/data/icons/miu-black-social-2/60/github-256.png" alt="16034820" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://twitter.com/princewillingoo" target="blank"><img align="center" src="https://bit.ly/3wK17I6" alt="gaurxvreddy" height="40" width="40" /></a></th>

""", unsafe_allow_html=True)
#st.write()
if not file:
    st.selectbox("The diffrent 101 classes of food you should try uploading", (['Click to see upload options', 'apple_pie',
                'baby_back_ribs',
                'baklava',
                'beef_carpaccio',
                'beef_tartare',
                'beet_salad',
                'beignets',
                'bibimbap',
                'bread_pudding',
                'breakfast_burrito',
                'bruschetta',
                'caesar_salad',
                'cannoli',
                'caprese_salad',
                'carrot_cake',
                'ceviche',
                'cheesecake',
                'cheese_plate',
                'chicken_curry',
                'chicken_quesadilla',
                'chicken_wings',
                'chocolate_cake',
                'chocolate_mousse',
                'churros',
                'clam_chowder',
                'club_sandwich',
                'crab_cakes',
                'creme_brulee',
                'croque_madame',
                'cup_cakes',
                'deviled_eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs_benedict',
                'escargots',
                'falafel',
                'filet_mignon',
                'fish_and_chips',
                'foie_gras',
                'french_fries',
                'french_onion_soup',
                'french_toast',
                'fried_calamari',
                'fried_rice',
                'frozen_yogurt',
                'garlic_bread',
                'gnocchi',
                'greek_salad',
                'grilled_cheese_sandwich',
                'grilled_salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot_and_sour_soup',
                'hot_dog',
                'huevos_rancheros',
                'hummus',
                'ice_cream',
                'lasagna',
                'lobster_bisque',
                'lobster_roll_sandwich',
                'macaroni_and_cheese',
                'macarons',
                'miso_soup',
                'mussels',
                'nachos',
                'omelette',
                'onion_rings',
                'oysters',
                'pad_thai',
                'paella',
                'pancakes',
                'panna_cotta',
                'peking_duck',
                'pho',
                'pizza',
                'pork_chop',
                'poutine',
                'prime_rib',
                'pulled_pork_sandwich',
                'ramen',
                'ravioli',
                'red_velvet_cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed_salad',
                'shrimp_and_grits',
                'spaghetti_bolognese',
                'spaghetti_carbonara',
                'spring_rolls',
                'steak',
                'strawberry_shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna_tartare',
                'waffles']))
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    image_show = Image.open(file)
    image_show = image_show.resize((500, 450))
    #plt.imshow(image)
    st.write("Here is the image you've selected")
    st.image(image_show)#, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    st.write("Please wait while your image is being processed.....")
    pred_class, pred_conf, df = predicting(image, model)
    st.write(df)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))
