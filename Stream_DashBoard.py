import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
import pandas as pd
import io
import streamlit as st
import tempfile 
warnings.filterwarnings('ignore')

# Function to split Timestamp into Date and Time columns and sort by both
def split_and_sort_timestamp(df, timestamp_col):
    # Split the timestamp into Date and Time
    df['Date'] = df[timestamp_col].str.split('T').str[0]
    df['Time'] = df[timestamp_col].str.split('T').str[1].str.split('.').str[0]

    # Drop the original Timestamp column
    df = df.drop(columns=[timestamp_col])

    # Sort by Date and Time
    df = df.sort_values(by=['Date', 'Time']).reset_index(drop=True)

    return df

def create_overheight_sheet(temp_file,threshold_right,threshold_left,threshold_horizontal):
    overheight_rows = []
    timestamps = []

    # Reopen the input file to retrieve timestamps
    with open(temp_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Skip irrelevant lines
            if (
                "Exception while processing environment." in line or
                "ProcessingTime;TimeStamp;Iteration;R10_CartNumber;R20_Height_Left;R21_Height_Right;R30_Distance;ExtractedResults" in line
            ):
                continue

            parts = line.strip().split(';')
            if len(parts) < 6:  # Make sure the line has at least 4 elements
                continue
            carrier = int(parts[3].split('.')[0])  # Extract carrier
            Left_Wheel_Measurements = float(parts[4]) if parts[4] else 0
            Right_Wheel_Measurements = float(parts[5]) if parts[5] else 0
            Horizontal_Measurements = float(parts[6]) if parts[6] else 0
            timestamp = parts[1]  # Extract timestamp

            # Check if any measurement exceeds the limits
            if Left_Wheel_Measurements > threshold_left or Right_Wheel_Measurements > threshold_right or Horizontal_Measurements > threshold_horizontal:
                overheight_rows.append([carrier, Left_Wheel_Measurements, Right_Wheel_Measurements, Horizontal_Measurements, timestamp])

    # Create a DataFrame for OverHeight data
    overheight_df = pd.DataFrame(overheight_rows, columns=['Carrier', 'Left_Wheel_Measurements', 'Right_Wheel_Measurements', 'Horizontal_Measurements', 'Timestamp'])
    overheight_df = overheight_df.fillna(0)
    # Filter carriers greater than 667
    overheight_df= overheight_df[overheight_df['Carrier'] <= 667]
    overheight_df = overheight_df[overheight_df['Carrier'] != 0]
    # Split and sort by Date and Time (assuming split_and_sort_timestamp is a function you have)
    overheight_df = split_and_sort_timestamp(overheight_df, 'Timestamp')

    # Calculate the summary counts
    left_wheel_count = (overheight_df['Left_Wheel_Measurements'] > 2.000).sum()
    right_wheel_count = (overheight_df['Right_Wheel_Measurements'] > 2.000).sum()
    horizontal_count = (overheight_df['Horizontal_Measurements'] > 1.000).sum()

    summary_data = {
        'Measurement': ['Left Wheel > 2.0', 'Right Wheel > 2.0', 'Horizontal > 1.0'],
        'Count': [left_wheel_count, right_wheel_count, horizontal_count]
    }
    summary_df = pd.DataFrame(summary_data)
    
    return summary_df,overheight_df

# Create the Error_Carriers sheet with Date and Time split
def create_error_carriers_sheet(temp_file_path):
    error_rows = []
    timestamps = []

    # Reopen the input file to retrieve timestamps
    with open(temp_file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Skip irrelevant lines
            if (
                "Exception while processing environment." in line or
                "ProcessingTime;TimeStamp;Iteration;R10_CartNumber;R20_Height_Left;R21_Height_Right;R30_Distance;ExtractedResults" in line
            ):
                continue

            parts = line.strip().split(';')
            if len(parts) < 6:  # Make sure the line has at least 4 elements
                continue
            carrier = int(parts[3].split('.')[0]) if parts[3] and parts[3].split('.')[0].isdigit() else 0 # Extract carrier
            Left_Wheel_Measurements = float(parts[4]) if parts[4] else 0
            Right_Wheel_Measurements = float(parts[5]) if parts[5] else 0
            Horizontal_Measurements = float(parts[6]) if parts[6] else 0
            timestamp = parts[1]  # Extract timestamp

            # Check if the carrier is greater than 667
            if carrier > 667:
                error_rows.append([carrier, Left_Wheel_Measurements, Right_Wheel_Measurements, Horizontal_Measurements, timestamp])

    # Create a DataFrame for Error_Carriers data
    error_df = pd.DataFrame(error_rows, columns=['Carrier', 'Left_Wheel_Measurements', 'Right_Wheel_Measurements', 'Horizontal_Measurements', 'Timestamp'])

    # Split and sort by Date and Time
    error_df = split_and_sort_timestamp(error_df, 'Timestamp')
    error_df =pd.DataFrame(error_df)
    return error_df

# Function to create and save result DataFrame as a new sheet in the same Excel file
def create_result_df(measurement_column):
    result_dict = {target: df[df['Carrier'] == target][measurement_column].tolist() for target in unique_carriers}
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')
    result_df = result_df.reset_index().rename(columns={'index': 'Carrier Code'})

    # Use numeric column names (1, 2, 3, etc.)
    result_df.columns = ['Carrier Code'] + [i + 1 for i in range(result_df.shape[1] - 1)]

    # Ensure all columns are numeric, except the 'Carrier Code' column
    result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    result_df = result_df.sort_values(by='Carrier Code', ascending=True).reset_index(drop=True)
    return result_df

def create_result_df_clean(measurement_column,threshold):
    # Filter values based on the threshold
    filtered_dict = {
        target: [val for val in df[df['Carrier'] == target][measurement_column] if val <= threshold]
        for target in unique_carriers
    }
    
    # Create a DataFrame from the filtered dictionary
    result_df = pd.DataFrame.from_dict(filtered_dict, orient='index')
    result_df = result_df.reset_index().rename(columns={'index': 'Carrier Code'})

    # Use numeric column names (1, 2, 3, etc.)
    result_df.columns = ['Carrier Code'] + [i + 1 for i in range(result_df.shape[1] - 1)]

    # Ensure all columns are numeric, except the 'Carrier Code' column
    result_df.iloc[:, 1:] = result_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    result_df = result_df.sort_values(by='Carrier Code', ascending=True).reset_index(drop=True)
    return result_df

def perform_analysis(df):
    analysis_results = []

    # Iterate over each row and perform analysis
    for index, row in df.iterrows():
        if row.dropna().empty:  # Skip the row if it's all empty
            continue

        carrier_code = row['Carrier Code']
        values = row.iloc[1:].dropna().values  # Get all values for this carrier
        
        # Skip this row if values array is empty
        if len(values) == 0:
            continue

        # Perform calculations
        min_val = values.min()
        max_val = values.max()
        value_range = round(max_val - min_val, 3)
        avg_val = round(values.mean(), 3)
        
        # Append results to the list
        analysis_results.append([carrier_code, min_val, max_val, value_range, avg_val])

    # Define columns for the output DataFrame
    columns = ['Carrier Code', 'Min', 'Max', 'Range', 'Average Value']

    # Create a DataFrame with the results
    analysis_df = pd.DataFrame(analysis_results, columns=columns)
    
    return analysis_df

# Create the "Analytical" sheet
def create_analytical_data():
    analytical_data = []
    for idx, (left_row, right_row, Horizontal_row) in enumerate(zip(result_df_1.itertuples(), result_df_2.itertuples(), result_df_3.itertuples())):
        # Calculate Average, Max, Min for each measurement
        left_values = left_row[2:]
        right_values = right_row[2:]
        Horizontal_values = Horizontal_row[2:]

        left_avg, left_max, left_min = round(pd.Series(left_values).mean(), 3), pd.Series(left_values).max(), pd.Series(left_values).min()
        right_avg, right_max, right_min = round(pd.Series(right_values).mean(), 3), pd.Series(right_values).max(), pd.Series(right_values).min()
        Horizontal_avg, Horizontal_max, Horizontal_min = round(pd.Series(Horizontal_values).mean(), 3), pd.Series(Horizontal_values).max(), pd.Series(Horizontal_values).min()

        analytical_data.append([
            left_avg, left_max, left_min,
            right_avg, right_max, right_min,
            Horizontal_avg, Horizontal_max, Horizontal_min
        ])

    analytical_df = pd.DataFrame(analytical_data, columns=[
        'Left Wheel Average', 'Left Wheel Max', 'Left Wheel Min',
        'Right Wheel Average', 'Right Wheel Max', 'Right Wheel Min',
        'Horizontal_Average', 'Horizontal_Max', 'Horizontal_Min'
    ])
    return analytical_df   

# Create the "Analytical" sheet
def clean_create_analytical_data():
    analytical_data = []
    for idx, (left_row, right_row, Horizontal_row) in enumerate(zip(clean_result_df_1.itertuples(), clean_result_df_2.itertuples(), clean_result_df_3.itertuples())):
        # Calculate Average, Max, Min for each measurement
        left_values = left_row[2:]
        right_values = right_row[2:]
        Horizontal_values = Horizontal_row[2:]

        left_avg, left_max, left_min = round(pd.Series(left_values).mean(), 3), pd.Series(left_values).max(), pd.Series(left_values).min()
        right_avg, right_max, right_min = round(pd.Series(right_values).mean(), 3), pd.Series(right_values).max(), pd.Series(right_values).min()
        Horizontal_avg, Horizontal_max, Horizontal_min = round(pd.Series(Horizontal_values).mean(), 3), pd.Series(Horizontal_values).max(), pd.Series(Horizontal_values).min()

        analytical_data.append([
            left_avg, left_max, left_min,
            right_avg, right_max, right_min,
            Horizontal_avg, Horizontal_max, Horizontal_min
        ])

    analytical_df = pd.DataFrame(analytical_data, columns=[
        'Left Wheel Average', 'Left Wheel Max', 'Left Wheel Min',
        'Right Wheel Average', 'Right Wheel Max', 'Right Wheel Min',
        'Horizontal_Average', 'Horizontal_Max', 'Horizontal_Min'
    ])
    return analytical_df

def create_carrier_data():
    # Extract the required data for the dashboard
    dashboard_data = pd.DataFrame({
        'Carrier Code': analysis_df_1['Carrier Code'],
        'Left Wheel Average': analysis_df_1['Average Value'],
        'Left Wheel Range': analysis_df_1['Range'],
        'Left Wheel Max' : analysis_df_1['Max'],
        'Left Wheel Min' : analysis_df_1['Min'],
        'Right Wheel Average': analysis_df_2['Average Value'],
        'Right Wheel Range': analysis_df_2['Range'],
        'Right Wheel Max' : analysis_df_2['Max'],
        'Right Wheel Min' : analysis_df_2['Min'],
        'Horizontal Average': analysis_df_3['Average Value'],
        'Horizontal Range': analysis_df_3['Range'],
        'RHorizontal Wheel Max' : analysis_df_3['Max'],
        'Horizontal Min' : analysis_df_3['Min'],
    })
    return dashboard_data

def create_carrier_data_clean():
    # Extract the required data for the dashboard
    dashboard_data = pd.DataFrame({
        'Carrier Code': clean_analysis_df_1['Carrier Code'],
        'Left Wheel Average': clean_analysis_df_1['Average Value'],
        'Left Wheel Range': clean_analysis_df_1['Range'],
        'Left Wheel Max' : clean_analysis_df_1['Max'],
        'Left Wheel Min' : clean_analysis_df_1['Min'],
        'Right Wheel Average': clean_analysis_df_2['Average Value'],
        'Right Wheel Range': clean_analysis_df_2['Range'],
        'Right Wheel Max' : clean_analysis_df_2['Max'],
        'Right Wheel Min' : clean_analysis_df_2['Min'],
        'Horizontal Average': clean_analysis_df_3['Average Value'],
        'Horizontal Range': clean_analysis_df_3['Range'],
        'RHorizontal Wheel Max' : clean_analysis_df_3['Max'],
        'Horizontal Min' : clean_analysis_df_3['Min'],
    })
    return dashboard_data
                      
def create_dashboard():
    # Extract the required data for the dashboard
    dashboard_data = pd.DataFrame({
        'Carrier Code': analysis_df_1['Carrier Code'],
        'Left Wheel Average': analysis_df_1['Average Value'],
        'Left Wheel Range': analysis_df_1['Range'],
        'Left Wheel Max' : analysis_df_1['Max'],
        'Left Wheel Min' : analysis_df_1['Min'],
        'Right Wheel Average': analysis_df_2['Average Value'],
        'Right Wheel Range': analysis_df_2['Range'],
        'Right Wheel Max' : analysis_df_2['Max'],
        'Right Wheel Min' : analysis_df_2['Min'],
        'Horizontal Average': analysis_df_3['Average Value'],
        'Horizontal Range': analysis_df_3['Range'],
        'RHorizontal Wheel Max' : analysis_df_3['Max'],
        'Horizontal Min' : analysis_df_3['Min'],
    })
    return dashboard_data

def clean_create_dashboard():
    # Extract the required data for the dashboard
    dashboard_data = pd.DataFrame({
        'Carrier Code': clean_analysis_df_1['Carrier Code'],
        'Left Wheel Average': clean_analysis_df_1['Average Value'],
        'Left Wheel Range': clean_analysis_df_1['Range'],
        'Left Wheel Max' : clean_analysis_df_1['Max'],
        'Left Wheel Min' : clean_analysis_df_1['Min'],
        'Right Wheel Average': clean_analysis_df_2['Average Value'],
        'Right Wheel Range': clean_analysis_df_2['Range'],
        'Right Wheel Max' : clean_analysis_df_2['Max'],
        'Right Wheel Min' : clean_analysis_df_2['Min'],
        'Horizontal Average': clean_analysis_df_3['Average Value'],
        'Horizontal Range': clean_analysis_df_3['Range'],
        'RHorizontal Wheel Max' : clean_analysis_df_3['Max'],
        'Horizontal Min' : clean_analysis_df_3['Min'],
    })
    return dashboard_data
# Load your logo image
logo_url = "Images/LCY3 Logo.png"
st.set_page_config(page_title='Amazon RME Wheel Alignment Analysis ', page_icon=logo_url, layout = "wide")
st.title(f":bar_chart: AMAZON RME WHEEL ALIGNMENT ANALYSIS DASHBOARD")
st.markdown('<style>div.block-container{padding-top:2.5%;}</style>',unsafe_allow_html=True)


# Display the logo in the sidebar
st.sidebar.image(logo_url, width=300)
# Input fields for thresholds
st.sidebar.header("Set Thresholds")
threshold_left = st.sidebar.number_input("Threshold for Left", value=0.0, step=0.1, format="%.2f")
threshold_right = st.sidebar.number_input("Threshold for Right", value=0.0, step=0.1, format="%.2f")
threshold_horizontal = st.sidebar.number_input("Threshold for Horizontal", value=0.0, step=0.1, format="%.2f")

file_upload = st.file_uploader(":file_folder: Upload a file", type=(["txt"]))
if st.button('Process Data'):
    if file_upload is not None:
        with st.spinner('Please wait processing your data...'):
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_upload.read())
                    temp_file.flush()
                    temp_file_path = temp_file.name
                    # Read the input file into a DataFrame
                    data = []
                    timestamps = []
                    with open(temp_file_path, 'r') as file:
                        for line in file:
                            if "Exception while processing environment." in line or "ProcessingTime;TimeStamp;Iteration;R10_CartNumber;R20_Height_Left;R21_Height_Right;R30_Distance;ExtractedResults" in line:
                                continue  # Skip lines containing the specified pattern

                            parts = line.strip().split(';')
                            if len(parts) < 6:  # Make sure the line has at least 4 elements
                                continue
                            carrier = int(parts[3].split('.')[0])  # Remove .000 from the carrier
                            Left_Wheel_Measurements = float(parts[4]) if parts[4] else 0
                            Right_Wheel_Measurements = float(parts[5]) if parts[5] else 0
                            Horizontal_Measurements = float(parts[6]) if parts[6] else 0
                            timestamp = parts[1]  # Extract timestamp
                            data.append([carrier, Left_Wheel_Measurements, Right_Wheel_Measurements, Horizontal_Measurements,timestamp])
                            
                    df = pd.DataFrame(data, columns=['Carrier', 'Left_Wheel_Measurements', 'Right_Wheel_Measurements', 'Horizontal_Measurements','Timestamp'])
                    # Split and sort by Date and Time
                    df = split_and_sort_timestamp(df, 'Timestamp')
                    df = df.fillna(0)
                    data_original = df.copy()
                    # Filter carriers greater than 667
                    df = df[df['Carrier'] <= 667]
                    df = df[df['Carrier'] != 0]
                    unique_carriers = df['Carrier'].unique()
                    summarys_df, overheight_measurements = create_overheight_sheet(temp_file_path,threshold_right,threshold_left,threshold_horizontal)
                    errors_df= create_error_carriers_sheet(temp_file_path)
                    
                    # Create result DataFrames for each measurement column and save them as separate sheets
                    result_df_1 = create_result_df('Left_Wheel_Measurements')
                    result_df_2 = create_result_df('Right_Wheel_Measurements')
                    result_df_3 = create_result_df('Horizontal_Measurements')
                    
                    # Create result DataFrames for each measurement column and save them as separate sheets
                    clean_result_df_1 = create_result_df_clean('Left_Wheel_Measurements', threshold=threshold_left)
                    clean_result_df_2 = create_result_df_clean('Right_Wheel_Measurements', threshold=threshold_right)
                    clean_result_df_3 = create_result_df_clean('Horizontal_Measurements', threshold=threshold_horizontal)
                    
                    
                    # Generate the analysis sheets
                    analysis_df_1 = perform_analysis(result_df_1)
                    analysis_df_2 = perform_analysis(result_df_2)
                    analysis_df_3 = perform_analysis(result_df_3)
                    
                    
                    # Generate the analysis sheets
                    clean_analysis_df_1 = perform_analysis(clean_result_df_1)
                    clean_analysis_df_2 = perform_analysis(clean_result_df_2)
                    clean_analysis_df_3 = perform_analysis(clean_result_df_3)
                    
                    analytical_data = create_analytical_data()
                    clean_analytical_data = clean_create_analytical_data()
                    
                    graph_data = create_dashboard()
                    clean_graph_data = clean_create_dashboard()
                    
                    card_1,card_2,card_3,card_4 = st.columns(4,gap='large')
                    with card_1:
                        st.info('Total Row Analysed')
                        st.metric(label = 'Total Row Analysed',value=f"{len(data_original):,.0f}")
                        
                    with card_2:
                        st.info('Rows With Valid Carriers')
                        st.metric(label = 'Rows With Valid Carriers',value=f"{len(df):,.0f}")
                        
                    with card_3:
                        st.info('Error % Rows')
                        st.metric(label = 'Rows With Error Carries Percenatge',value=f"{(len(errors_df)/len(data_original))*100:,.0f}%")
                        
                    with card_4:
                        st.info('Higher than Threshold Count')
                        st.metric(label = 'Rows With Error Carries Percenatge',value=f"{len(overheight_measurements):,.0f}")
                    
                    # Create tabs
                    tab1, tab2, tab3 = st.tabs(["Datasets", "Visuals for Original Data","Visuals for Threshold Data"])
                    with tab1:
                        col1,col2 = st.columns((2))
                        with col1:
                            # st.subheader('Invalid Data Count by Wheels')
                            # st.write(summarys_df)
                            st.subheader('Errors Dataset')
                            st.dataframe(errors_df,height=400)

                        
                        with col2:
                            st.subheader('Overheight measurments')
                            st.write(overheight_measurements)
                            # st.subheader('Valid Dataset')
                            # st.dataframe(df)
    
                        st.subheader('Carrier Data')
                        st.write(graph_data)
                        st.subheader('Cleaned Carrier Data')
                        st.dataframe(clean_graph_data)   
                    
                        
                    # Display views in each tab
                    with tab2:
                        st.write("View for Dataset 1")
                        st.subheader('Original Data Left Wheel Average')
                        linechart_1 = pd.DataFrame(graph_data[['Carrier Code','Left Wheel Average']])
                        fig1 = px.line(linechart_1,x = 'Carrier Code',y = 'Left Wheel Average')
                        st.plotly_chart(fig1)
                        st.subheader('Original Data Left Wheel Range')
                        linechart_2 = pd.DataFrame(graph_data[['Carrier Code','Left Wheel Range']])
                        fig2 = px.line(linechart_2,x = 'Carrier Code',y = 'Left Wheel Range')
                        st.plotly_chart(fig2)
                        
                        st.subheader('Original Data Right Wheel Average')
                        linechart_3 = pd.DataFrame(graph_data[['Carrier Code','Right Wheel Average']])
                        fig3 = px.line(linechart_3,x = 'Carrier Code',y = 'Right Wheel Average')
                        st.plotly_chart(fig3)
                        
                        st.subheader('Original Data Right Wheel Range')
                        linechart_4 = pd.DataFrame(graph_data[['Carrier Code','Right Wheel Range']])
                        fig4 = px.line(linechart_4,x = 'Carrier Code',y = 'Right Wheel Range')
                        st.plotly_chart(fig4)
                        
                        st.subheader('Original Data Horizontal Average')
                        linechart_5 = pd.DataFrame(graph_data[['Carrier Code','Horizontal Average']])
                        fig5 = px.line(linechart_5,x = 'Carrier Code',y = 'Horizontal Average')
                        st.plotly_chart(fig5)
                        
                        st.subheader('Original Data Horizontal Range')
                        linechart_6 = pd.DataFrame(graph_data[['Carrier Code','Horizontal Range']])
                        fig6 = px.line(linechart_6,x = 'Carrier Code',y = 'Horizontal Range')
                        st.plotly_chart(fig6)

                    with tab3:
                        st.write("View for Dataset 2")
                        st.subheader('Threshold Data Left Wheel Average')
                        linechart_7 = pd.DataFrame(clean_graph_data[['Carrier Code','Left Wheel Average']])
                        fig7 = px.line(linechart_7,x = 'Carrier Code',y = 'Left Wheel Average')
                        st.plotly_chart(fig7)
                        st.subheader('Threshold Data Left Wheel Range')
                        linechart_8 = pd.DataFrame(clean_graph_data[['Carrier Code','Left Wheel Range']])
                        fig8 = px.line(linechart_8,x = 'Carrier Code',y = 'Left Wheel Range')
                        st.plotly_chart(fig8)

                        st.subheader('Threshold Data Right Wheel Average')
                        linechart_9 = pd.DataFrame(clean_graph_data[['Carrier Code','Right Wheel Average']])
                        fig9 = px.line(linechart_9,x = 'Carrier Code',y = 'Right Wheel Average')
                        st.plotly_chart(fig9)

                        st.subheader('Threshold Data Right Wheel Range')
                        linechart_10 = pd.DataFrame(clean_graph_data[['Carrier Code','Right Wheel Range']])
                        fig10 = px.line(linechart_10,x = 'Carrier Code',y = 'Right Wheel Range')
                        st.plotly_chart(fig10)

                        st.subheader('Threshold Data Horizontal Average')
                        linechart_11 = pd.DataFrame(clean_graph_data[['Carrier Code','Horizontal Average']])
                        fig11 = px.line(linechart_11,x = 'Carrier Code',y = 'Horizontal Average')
                        st.plotly_chart(fig11)

                        st.subheader('Threshold Data Horizontal Range')
                        linechart_12 = pd.DataFrame(clean_graph_data[['Carrier Code','Horizontal Range']])
                        fig12 = px.line(linechart_12,x = 'Carrier Code',y = 'Horizontal Range')
                        st.plotly_chart(fig12)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            
    else:
        st.warning('Please select a file!')