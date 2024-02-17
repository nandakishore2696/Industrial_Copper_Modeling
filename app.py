import streamlit as st 
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np 

# set page layout
st.set_page_config(layout='wide')

# header
header_text = "Industrial Copper Selling Price and Status Modeling"

st.markdown(
    f"""
    <h1 style='text-align: center; color: white; font-style: italic;'>{header_text}</h1>
    """,
    unsafe_allow_html=True
)

# Using option menu for accessing 2 windows
selected = option_menu(menu_title=None, options= ['Regression', 'Classification'], orientation = 'horizontal' )

# loading the pickled files 
with open('transformer.pkl', 'rb') as f:
    transformer = pickle.load(f)

with open('reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('class_transformer.pkl', 'rb') as f:
    class_transformer = pickle.load(f)

with open('class_model.pkl', 'rb') as f:
    class_model = pickle.load(f)


if selected == 'Regression':

    # using streamlit forms to get input data from user
    with st.form('User Input'):
        quantity_tonnes_inp_1 = st.number_input("Enter the quantity in tonnes", min_value=0.00001, max_value=1.000000e+09)
        customer_inp_1 = st.number_input("Enter the customer ID")  
        country_inp_1 = st.number_input("Enter the country ID")   
        status_inp_1 = st.text_input("Enter the status")  
        item_label_inp_1 = st.text_input("Enter the item type")  
        application_inp_1 = st.number_input("Enter the application ID")  
        thickness_inp_1 = st.number_input("Enter the thickness", min_value=0.18000 , max_value=2.500000e+03)
        width_inp_1 = st.number_input("Enter the width", min_value=1.00000, max_value=2.990000e+03)
        product_ref_inp_1 = st.number_input("Enter the product reference ID")  
        duration_inp_1 = st.number_input("Enter the duration", min_value=0.00000, max_value=4.480000e+02)

        button = st.form_submit_button("submit")

    # after the user clicks submit button, the output displays
    if button:
        input = pd.DataFrame([[quantity_tonnes_inp_1, customer_inp_1, country_inp_1, status_inp_1,
                            item_label_inp_1, application_inp_1, thickness_inp_1,
                            width_inp_1,product_ref_inp_1,duration_inp_1]], 
                        columns=['quantity tons', 'customer', 'country', 'status', 'item type',
                                'application', 'thickness', 'width', 'product_ref', 'duration'])
        input['customer'] = input['customer'].astype('category')
        input['country'] = input['country'].astype('category')
        input['status'] = input['status'].astype('category')
        input['item type'] = input['item type'].astype('category')
        input['application'] = input['application'].astype('category')#
        input['product_ref'] = input['product_ref'].astype('category')

        input_ct = transformer.transform(input)

        sell_price = reg_model.predict(input_ct)[0]
        sell_price_out = np.round(sell_price,2)
        

        # Display the highlighted value with increased font size

        font_size = 40
        st.header('Sell Price')
        st.markdown(f'<span style="color: green; font-size: {font_size}px;">{sell_price_out}</span>', unsafe_allow_html=True)



if selected == 'Classification':
    
    # using streamlit forms to get input data from user
    with st.form('User Input'):
        quantity_tonnes_inp_2 = st.number_input("Enter the quantity in tonnes", min_value=0.00001, max_value=1.000000e+09)
        customer_inp_2 = st.number_input("Enter the customer ID")  
        country_inp_2 = st.number_input("Enter the country ID")   
        status_inp_2 = 'Won'  
        item_label_inp_2 = st.text_input("Enter the item type")  
        application_inp_2 = st.number_input("Enter the application ID")  
        thickness_inp_2 = st.number_input("Enter the thickness", min_value=0.18000 , max_value=2.500000e+03)
        width_inp_2 = st.number_input("Enter the width", min_value=1.00000, max_value=2.990000e+03)
        product_ref_inp_2 = st.number_input("Enter the product reference ID")  
        selling_price_2 = st.number_input("Enter the selling price", min_value=	0.10000, max_value=1.000010e+08) 
        duration_inp_2 = st.number_input("Enter the duration", min_value=0.00000, max_value=4.480000e+02)

        button = st.form_submit_button("submit")

   
    try:
        input = pd.DataFrame([[quantity_tonnes_inp_2, customer_inp_2, country_inp_2, status_inp_2, item_label_inp_2,
                                application_inp_2, thickness_inp_2 ,width_inp_2 ,
                                product_ref_inp_2 ,selling_price_2 ,duration_inp_2]], 
                            columns=['quantity tons', 'customer', 'country', 'status', 'item type',
                                    'application', 'thickness', 'width', 'product_ref','selling_price', 'duration'])
        input['customer'] = input['customer'].astype('category')
        input['country'] = input['country'].astype('category')
        input['status'] = input['status'].astype('category')
        input['item type'] = input['item type'].astype('category')
        input['application'] = input['application'].astype('category')#
        input['product_ref'] = input['product_ref'].astype('category')

        input_ct = class_transformer.transform(input)
        status_encoded = class_model.predict(input_ct.drop("ordinalencoder__status", axis=1))[0]
        status_encoded = int(status_encoded)
        status_out = class_transformer.named_transformers_['ordinalencoder'].categories_[5][status_encoded]
        font_size = 40

        # Display the highlighted value with increased font size
        st.header('Status')
        st.markdown(f'<span style="color: green; font-size: {font_size}px;">{status_out}</span>', unsafe_allow_html=True)


    except Exception as e: 
       print(e)





        




        


