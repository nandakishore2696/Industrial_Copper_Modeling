import streamlit as st 
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
import numpy as np 

st.set_page_config(layout='wide')

header_text = "Industrial Copper Selling Price and Status Modeling"

st.markdown(
    f"""
    <h1 style='text-align: center; color: black; font-style: italic;'>{header_text}</h1>
    """,
    unsafe_allow_html=True
)


selected = option_menu(menu_title=None, options= ['Regression', 'Classification'], orientation = 'horizontal' )


with open('label_mapping.pkl', 'rb') as f:
    label_mapping = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

if selected == 'Regression':
    col1, col2 = st.columns(2)
        # 'customer' - label, 'country' - label, 
        # 'status' - label, 'item type' - label, 
        # 'application' - label, 'width',
        # 'product_ref'- label, 'quantity tons_log', 
        #'thickness_log','selling_price_log', 'duration_log'
    with col1:
        with st.form('User Input'):
            customer_inp_1 = st.number_input("Enter the customer ID")  
            country_inp_1 = st.number_input("Enter the country ID")   
            status_inp_1 = st.text_input("Enter the status")  
            item_label_inp_1 = st.text_input("Enter the item type")  
            application_inp_1 = st.number_input("Enter the application ID")  
            product_ref_inp_1 = st.number_input("Enter the product reference ID")  
            width_inp_1 = st.number_input("Enter yhe width", min_value=1.00000, max_value=2.990000e+03)
            thickness_inp_1 = st.number_input("Enter the thickness", min_value=0.18000 , max_value=2.500000e+03)
            quantity_tonnes_inp_1 = st.number_input("Enter the quantity in tonnes", min_value=0.00001, max_value=1.000000e+09)
            duration_inp_1 = st.number_input("Enter the duration", min_value=0.00000, max_value=4.480000e+02)

            button = st.form_submit_button("submit")
  

    with col2:
        colm1,colm2,colm3= st.columns(3,gap='small')
        with colm1:
            st.write('customer_labels')
            st.dataframe(label_mapping['customer_labels'])

            st.write('item type_labels')
            st.dataframe(label_mapping['item type_labels'])            
        with colm2:
            st.write('country_labels')
            st.dataframe(label_mapping['country_labels'])

            st.write('application_labels')
            st.dataframe(label_mapping['application_labels'])
        with colm3:
            st.write('status_labels')
            st.dataframe(label_mapping['status_labels'])

            st.write('product_ref_labels')
            st.dataframe(label_mapping['product_ref_labels'])
        # 'customer', 'country', 'status', 'item type', 'application', 'width','product_ref', 
        # 'quantity tons_log', 'thickness_log', 'duration_log'
   
    try:
        customer_inp = label_mapping['customer_labels'][customer_inp_1]
        country_inp = label_mapping['country_labels'][country_inp_1]
        status_inp = label_mapping['status_labels'][status_inp_1]
        item_label_inp =  label_mapping['item type_labels'][item_label_inp_1]
        application_inp = label_mapping['application_labels'][application_inp_1]
        width_inp = width_inp_1
        product_ref_inp = label_mapping['product_ref_labels'][product_ref_inp_1]
        quantity_tonnes_inp = np.log1p(quantity_tonnes_inp_1)
        thickness_inp = np.log1p(thickness_inp_1)
        duration_inp = np.log1p(duration_inp_1)
        inp = np.array([customer_inp, country_inp, status_inp, item_label_inp, application_inp, width_inp, product_ref_inp, quantity_tonnes_inp,thickness_inp,duration_inp])


        sell_price_1 = model.predict(inp.reshape(1,-1))
        sell_price = np.exp(sell_price_1[0])-1
        font_size = 40

        # Display the highlighted value with increased font size
        st.write('Sell Price')
        st.markdown(f'<span style="color: green; font-size: {font_size}px;">{sell_price}</span>', unsafe_allow_html=True)

    except Exception as e:
        st.write(e)


if selected == 'Classification':
    col1, col2 = st.columns(2)

    with open('class_model.pkl','rb')as f:
        class_modell = pickle.load(f)

    with col1:
        with st.form('User Input'):
            customer_inp_1 = st.number_input("Enter the customer ID")  
            country_inp_1 = st.number_input("Enter the country ID")   
            selling_price_1 = st.number_input("Enter the selling price", min_value=	0.10000, max_value=1.000010e+08) 
            item_label_inp_1 = st.text_input("Enter the item type")  
            application_inp_1 = st.number_input("Enter the application ID")  
            product_ref_inp_1 = st.number_input("Enter the product reference ID")  
            width_inp_1 = st.number_input("Enter yhe width", min_value=1.00000, max_value=2.990000e+03)
            thickness_inp_1 = st.number_input("Enter the thickness", min_value=0.18000 , max_value=2.500000e+03)
            quantity_tonnes_inp_1 = st.number_input("Enter the quantity in tonnes", min_value=0.00001, max_value=1.000000e+09)
            duration_inp_1 = st.number_input("Enter the duration", min_value=0.00000, max_value=4.480000e+02)

            button = st.form_submit_button("submit")
  

    with col2:
        colm1,colm2,colm3= st.columns(3,gap='small')
        with colm1:
            st.write('customer_labels')
            st.dataframe(label_mapping['customer_labels'])

            st.write('item type_labels')
            st.dataframe(label_mapping['item type_labels'])            
        with colm2:
            st.write('country_labels')
            st.dataframe(label_mapping['country_labels'])

            st.write('application_labels')
            st.dataframe(label_mapping['application_labels'])
        with colm3:
            st.write('product_ref_labels')
            st.dataframe(label_mapping['product_ref_labels'])
        #'customer', 'country', 'item type', 'application', 'width','product_ref', 'quantity tons_log', 'thickness_log',
        # 'selling_price_log', 'duration_log'
   
    try:
        customer_inp = label_mapping['customer_labels'][customer_inp_1]
        country_inp = label_mapping['country_labels'][country_inp_1]
        selling_price = np.log1p(selling_price_1)
        item_label_inp =  label_mapping['item type_labels'][item_label_inp_1]
        application_inp = label_mapping['application_labels'][application_inp_1]
        width_inp = width_inp_1
        product_ref_inp = label_mapping['product_ref_labels'][product_ref_inp_1]
        quantity_tonnes_inp = np.log1p(quantity_tonnes_inp_1)
        thickness_inp = np.log1p(thickness_inp_1)
        duration_inp = np.log1p(duration_inp_1)
        inp = np.array([customer_inp, country_inp, item_label_inp, application_inp, width_inp, product_ref_inp, quantity_tonnes_inp,thickness_inp,selling_price ,duration_inp])

        status_output_1 = class_modell.predict(inp.reshape(1,-1))
        def get_key(dictionary, val):
            for key, value in dictionary.items():
                if value == val:
                    return key
        status_output = get_key(label_mapping['status_labels'], status_output_1)
       
        font_size = 40

        # Display the highlighted value with increased font size
        st.write('Status')
        st.markdown(f'<span style="color: green; font-size: {font_size}px;">{status_output}</span>', unsafe_allow_html=True)

    except Exception as e:
        st.write(e)
     





        




        


