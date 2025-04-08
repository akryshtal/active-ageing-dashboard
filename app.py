import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Active Ageing Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set theme configuration
st.markdown("""
    <style>
    :root {
        --primary-color: #e5004d;
        --secondary-background-color: #e0e4d7;
        --text-color: #1d1d1d;
    }
    </style>
""", unsafe_allow_html=True)

# Add custom CSS for better dropdown styling
st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        min-height: 38px;
    }
    div[data-baseweb="select"] {
        margin-bottom: 15px;
    }
    .selectbox-container {
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load and preprocess data
def load_data():
    # Load the dataset
    df = pd.read_csv('Active_Ageing_Evaluation.csv', encoding='utf-8')
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Define column groups
    # Chronic illnesses group
    chronic_columns = [
        'chronic cardiovascular', 'Diabetes', 'Respiratory diseases', 
        'Musculoskeletal disorders', 'Neurological disorders', 'Dementia',
        'Oncological diseases', 'Vision or hearing impairments', 
        'Physical limitations due to injuries or disability',
        'Other (please specify)', 'No chronic diseases or physical limitations'
    ]
    
    # Activities group
    activity_columns = [
        'activity volunteers social', 'Social events', 'Interest-based clubs', 
        'Home care', 'Psychological support (group)', 'Psychological support (individual)',
        'Warm ome / Welcoming space', 'Physical rehabilitation', 'Physical activities'
    ]
    
    # Services group
    service_columns = [
        'None of the above', 'Purchase of tablets or phones', 
        'Mobility or exercise equipment', 'Medical treatment', 'Medical devices',
        'Delivery of food packages/hot meals', 'Medical support',
        'Delivery of blood pressure/oximeters', 'Winter assistance',
        'Other material assistance'
    ]
    
    # All binary columns to convert
    binary_columns = chronic_columns + activity_columns + service_columns
    
    for col in binary_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate additional metrics
    if all(col in df.columns for col in chronic_columns):
        df['chronic_count'] = df[chronic_columns].sum(axis=1)
    
    if all(col in df.columns for col in activity_columns):
        df['activity_count'] = df[activity_columns].sum(axis=1)
    
    if all(col in df.columns for col in service_columns):
        df['service_count'] = df[service_columns].sum(axis=1)
    
    return df

# Function to filter the dataframe based on selected criteria
def filter_dataframe(df, country, region, partner, age_range, gender, new_client, duration, victim_status):
    filtered_df = df.copy()
    
    # Apply each filter if not empty list
    if country and 'country' in df.columns:
        filtered_df = filtered_df[filtered_df['country'].isin(country)]
    
    if region and 'region' in df.columns:
        filtered_df = filtered_df[filtered_df['region'].isin(region)]
    
    if partner and 'Partner' in df.columns:
        filtered_df = filtered_df[filtered_df['Partner'].isin(partner)]
    
    if age_range and 'client_age_range' in df.columns:
        filtered_df = filtered_df[filtered_df['client_age_range'].isin(age_range)]
    
    if gender and 'client_gender' in df.columns:
        filtered_df = filtered_df[filtered_df['client_gender'].isin(gender)]
    
    if new_client and 'client_new' in df.columns:
        filtered_df = filtered_df[filtered_df['client_new'].isin(new_client)]
    
    if duration and 'How long does a client receive services funded by World Jewish Relief?' in df.columns:
        duration_column = 'How long does a client receive services funded by World Jewish Relief?'
        filtered_df = filtered_df[filtered_df[duration_column].isin(duration)]

    if victim_status and 'client_victim_of_nazism_status' in df.columns:
        filtered_df = filtered_df[filtered_df['client_victim_of_nazism_status'].isin(victim_status)]
    
    return filtered_df

def generate_chart_summary(data, chart_type, metric_name=None):
    """Generate automatic text summaries for charts."""
    if chart_type == 'gender_distribution':
        total = data['Count'].sum()
        if total == 0:
            return "No data available for gender distribution."
        main_gender = data.loc[data['Count'].idxmax(), 'Gender']
        main_percentage = (data.loc[data['Count'].idxmax(), 'Count'] / total * 100).round(1)
        return f"The majority of respondents ({main_percentage}%) identify as {main_gender}."
    
    elif chart_type == 'age_distribution':
        total = data['Count'].sum()
        if total == 0:
            return "No data available for age distribution."
        main_age = data.loc[data['Count'].idxmax(), 'Age Range']
        main_percentage = (data.loc[data['Count'].idxmax(), 'Count'] / total * 100).round(1)
        return f"The largest age group is {main_age}, representing {main_percentage}% of respondents."
    
    elif chart_type == 'chronic_diseases':
        total = len(data)
        most_common = data.loc[data['Count'].idxmax(), 'Disease']
        most_common_percentage = data.loc[data['Count'].idxmax(), 'Percentage']
        return f"The most common chronic condition is {most_common}, affecting {most_common_percentage:.1f}% of respondents."
    
    elif chart_type == 'activities':
        total = len(data)
        most_popular = data.loc[data['Count'].idxmax(), 'Activity']
        most_popular_percentage = data.loc[data['Count'].idxmax(), 'Percentage']
        return f"The most popular activity is {most_popular}, with {most_popular_percentage:.1f}% participation rate."
    
    elif chart_type == 'services':
        total = len(data)
        most_used = data.loc[data['Count'].idxmax(), 'Service']
        most_used_percentage = data.loc[data['Count'].idxmax(), 'Percentage']
        return f"The most utilized service is {most_used}, used by {most_used_percentage:.1f}% of respondents."
    
    elif chart_type == 'rating':
        avg_rating = data['Rating'].mean()
        return f"The average rating is {avg_rating:.1f} out of {data['Rating'].max()}, indicating {metric_name}."
    
    return ""

# Main function to run the Streamlit app
def main():
    # Language selection at the top
    language = st.sidebar.radio("Language / Мова / Язык", ["English", "Ukrainian", "Russian"])
    
    # Translations for UI elements
    translations = {
        "English": {
            "title": "Active Ageing Evaluation Dashboard",
            "filters": "Filters",
            "demographics": "Demographics",
            "program_impact": "Program Impact",
            "psychological": "Psychological State",
            "physical": "Physical State",
            "hypothesis": "Hypothesis Testing",
            "export_data": "Export Data",
            "download_button": "Download Filtered Data as CSV",
            "click_download": "Click to Download",
            "dataset_overview": "Dataset Overview",
            "total_records": "Total records after filtering",
            "gender_distribution": "Gender Distribution",
            "age_distribution": "Age Distribution",
            "chronic_diseases": "Chronic Diseases Distribution",
            "count_diseases": "Count of Chronic Diseases",
            "percent_diseases": "Percentage of Clients with Each Disease (%)",
            "comorbidity": "Chronic Disease Comorbidity",
            "correlation": "Correlation Between Different Chronic Conditions",
            "activities_participation": "Activities Participation",
            "activities_count": "Participation in Different Activities",
            "activities_percent": "Percentage Distribution of Activity Participation",
            "services_received": "Services Received",
            "services_count": "Distribution of Services Received",
            "services_percent": "Percentage Distribution of Services Received",
            "service_activity": "Service-Activity Intersection Analysis",
            "heatmap_desc": "This heatmap shows the number of clients participating in each activity who also receive each service:",
            "intersection": "Intersection Between Top Services and Activities",
            "activity": "Activity",
            "service": "Service",
            "count": "Number of Participants",
            "physical_freq": "Physical Activity Frequency",
            "freq_by_age": "Physical Activity Frequency by Age Group:",
            "social_impact": "Impact on Social Connections",
            "social_count": "Impact on Social Connections (Count)",
            "social_percent": "Impact on Social Connections (%)",
            "social_by_gender": "Impact on Social Connections by Gender:",
            "gender": "Gender",
            "physical_support": "Impact of Phycal Support on Health",
            "avg_rating": "Average Rating",
            "phys_support_title": "Impact of Physical Support on Health and Quality of Life",
            "phys_by_activity": "Impact of Physical Support by Type of Activities:",
            "avg_by_participation": "Average Physical Support Rating by Activity Participation",
            "rating": "Rating",
            "activity_status": "Activity Status",
            "distribution_rating": "Distribution of Physical Support Ratings by Activity Participation",
            "hyp1_title": "Hypothesis #1: Younger vs Older Activity Preferences",
            "hyp1_desc": "Hypothesis: 'Younger clients tend to choose physical activities more often; older clients prefer social and psychological support.'",
            "pref_by_age": "Activity Preferences by Age Group (%)",
            "participation_rate": "Participation Rate (%)",
            "activity_type": "Activity Type",
            "age_range": "Age Range",
            "physical_activities": "Physical Activities",
            "psychosocial_activities": "Psychosocial Activities",
            "statistical_analysis": "Statistical Analysis",
            "younger_older_comp": "Comparison between younger and older client groups:",
            "younger_vs_older": "Activity Preferences: Younger vs. Older Clients",
            "age_group": "Age Group",
            "conclusion": "Conclusion:"
        },
        "Ukrainian": {
            "title": "Панель оцінки активного старіння",
            "filters": "Фільтри",
            "demographics": "Демографія",
            "program_impact": "Вплив програми",
            "psychological": "Психологічний стан",
            "physical": "Фізичний стан",
            "hypothesis": "Перевірка гіпотез",
            "export_data": "Експорт даних",
            "download_button": "Завантажити відфільтровані дані як CSV",
            "click_download": "Натисніть для завантаження",
            "dataset_overview": "Огляд набору даних",
            "total_records": "Загальна кількість записів після фільтрації",
            "gender_distribution": "Розподіл за статтю",
            "age_distribution": "Розподіл за віком",
            "chronic_diseases": "Розподіл хронічних захворювань",
            "count_diseases": "Кількість хронічних захворювань",
            "percent_diseases": "Відсоток клієнтів з кожним захворюванням (%)",
            "comorbidity": "Коморбідність хронічних захворювань",
            "correlation": "Кореляція між різними хронічними станами",
            "activities_participation": "Участь в активностях",
            "activities_count": "Участь у різних активностях",
            "activities_percent": "Відсотковий розподіл участі в активностях",
            "services_received": "Отримані послуги",
            "services_count": "Розподіл отриманих послуг",
            "services_percent": "Відсотковий розподіл отриманих послуг",
            "service_activity": "Аналіз перетину послуг та активностей",
            "heatmap_desc": "Ця теплова карта показує кількість клієнтів, які беруть участь у кожній активності та одночасно отримують кожну послугу:",
            "intersection": "Перетин між найпопулярнішими послугами та активностями",
            "activity": "Активність",
            "service": "Послуга",
            "count": "Кількість учасників",
            "physical_freq": "Частота фізичної активності",
            "freq_by_age": "Частота фізичної активності за віковими групами:",
            "social_impact": "Вплив на соціальні зв'язки",
            "social_count": "Вплив на соціальні зв'язки (Кількість)",
            "social_percent": "Вплив на соціальні зв'язки (%)",
            "social_by_gender": "Вплив на соціальні зв'язки за статтю:",
            "gender": "Стать",
            "physical_support": "Вплив фізичної підтримки на здоров'я",
            "avg_rating": "Середня оцінка",
            "phys_support_title": "Вплив фізичної підтримки на здоров'я та якість життя",
            "phys_by_activity": "Вплив фізичної підтримки за типом активностей:",
            "avg_by_participation": "Середня оцінка фізичної підтримки за участю в активностях",
            "rating": "Оцінка",
            "activity_status": "Статус активності",
            "distribution_rating": "Розподіл оцінок фізичної підтримки за участю в активностях",
            "hyp1_title": "Гіпотеза №1: Вподобання активностей молодших і старших клієнтів",
            "hyp1_desc": "Гіпотеза: 'Молодші клієнти частіше обирають фізичні активності; старші надають перевагу соціальній і психологічній підтримці.'",
            "pref_by_age": "Вподобання активностей за віковими групами (%)",
            "participation_rate": "Рівень участі (%)",
            "activity_type": "Тип активності",
            "age_range": "Віковий діапазон",
            "physical_activities": "Фізичні активності",
            "psychosocial_activities": "Психосоціальні активності",
            "statistical_analysis": "Статистичний аналіз",
            "younger_older_comp": "Порівняння між групами молодших та старших клієнтів:",
            "younger_vs_older": "Вподобання активностей: молодші vs. старші клієнти",
            "age_group": "Вікова група",
            "conclusion": "Висновок:"
        },
        "Russian": {
            "title": "Панель оценки активного старения",
            "filters": "Фильтры",
            "demographics": "Демография",
            "program_impact": "Влияние программы",
            "psychological": "Психологическое состояние",
            "physical": "Физическое состояние",
            "hypothesis": "Проверка гипотез",
            "export_data": "Экспорт данных",
            "download_button": "Загрузить отфильтрованные данные как CSV",
            "click_download": "Нажмите для загрузки",
            "dataset_overview": "Обзор набора данных",
            "total_records": "Общее количество записей после фильтрации",
            "gender_distribution": "Распределение по полу",
            "age_distribution": "Распределение по возрасту",
            "chronic_diseases": "Распределение хронических заболеваний",
            "count_diseases": "Количество хронических заболеваний",
            "percent_diseases": "Процент клиентов с каждым заболеванием (%)",
            "comorbidity": "Коморбидность хронических заболеваний",
            "correlation": "Корреляция между различными хроническими состояниями",
            "activities_participation": "Участие в активностях",
            "activities_count": "Участие в различных активностях",
            "activities_percent": "Процентное распределение участия в активностях",
            "services_received": "Полученные услуги",
            "services_count": "Распределение полученных услуг",
            "services_percent": "Процентное распределение полученных услуг",
            "service_activity": "Анализ пересечения услуг и активностей",
            "heatmap_desc": "Эта тепловая карта показывает количество клиентов, участвующих в каждой активности и одновременно получающих каждую услугу:",
            "intersection": "Пересечение между самыми популярными услугами и активностями",
            "activity": "Активность",
            "service": "Услуга",
            "count": "Количество участников",
            "physical_freq": "Частота физической активности",
            "freq_by_age": "Частота физической активности по возрастным группам:",
            "social_impact": "Влияние на социальные связи",
            "social_count": "Влияние на социальные связи (Количество)",
            "social_percent": "Влияние на социальные связи (%)",
            "social_by_gender": "Влияние на социальные связи по полу:",
            "gender": "Пол",
            "physical_support": "Влияние физической поддержки на здоровье",
            "avg_rating": "Средняя оценка",
            "phys_support_title": "Влияние физической поддержки на здоровье и качество жизни",
            "phys_by_activity": "Влияние физической поддержки по типу активностей:",
            "avg_by_participation": "Средняя оценка физической поддержки по участию в активностях",
            "rating": "Оценка",
            "activity_status": "Статус активности",
            "distribution_rating": "Распределение оценок физической поддержки по участию в активностях",
            "hyp1_title": "Гипотеза №1: Предпочтения активностей младших и старших клиентов",
            "hyp1_desc": "Гипотеза: 'Младшие клиенты чаще выбирают физические активности; старшие предпочитают социальную и психологическую поддержку.'",
            "pref_by_age": "Предпочтения активностей по возрастным группам (%)",
            "participation_rate": "Уровень участия (%)",
            "activity_type": "Тип активности",
            "age_range": "Возрастной диапазон",
            "physical_activities": "Физические активности",
            "psychosocial_activities": "Психосоциальные активности",
            "statistical_analysis": "Статистический анализ",
            "younger_older_comp": "Сравнение между группами младших и старших клиентов:",
            "younger_vs_older": "Предпочтения активностей: младшие vs. старшие клиенты",
            "age_group": "Возрастная группа",
            "conclusion": "Заключение:"
        }
    }
    
    # Set title based on language
    st.title(translations[language]["title"])
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header(translations[language]["filters"])

    with st.sidebar:
        # Country filter
        if 'country' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            country_options = sorted(df['country'].dropna().unique().tolist())
            selected_country = st.multiselect('Country 🌍', 
                                            options=country_options,
                                            default=country_options,
                                            key='country_select',
                                            placeholder='Select countries...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_country = []
        
        # Region filter
        if 'region' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            if selected_country and 'country' in df.columns:
                available_regions = df[df['country'].isin(selected_country)]['region'].dropna().unique()
                region_options = sorted(available_regions.tolist())
            else:
                region_options = sorted(df['region'].dropna().unique().tolist())
            
            selected_region = st.multiselect('Region 📍', 
                                           options=region_options,
                                           default=region_options,
                                           key='region_select',
                                           placeholder='Select regions...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_region = []
        
        # Partner filter
        if 'Partner' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            filtered_for_partners = df
            if selected_country and 'country' in df.columns:
                filtered_for_partners = filtered_for_partners[filtered_for_partners['country'].isin(selected_country)]
            if selected_region and 'region' in df.columns:
                filtered_for_partners = filtered_for_partners[filtered_for_partners['region'].isin(selected_region)]
            
            partner_options = sorted(filtered_for_partners['Partner'].dropna().unique().tolist())
            selected_partner = st.multiselect('Partner 🤝', 
                                            options=partner_options,
                                            default=partner_options,
                                            key='partner_select',
                                            placeholder='Select partners...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_partner = []
        
        # Age range filter
        if 'client_age_range' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            age_range_options = sorted(df['client_age_range'].dropna().unique().tolist())
            selected_age_range = st.multiselect('Client Age Range 👥', 
                                              options=age_range_options,
                                              default=age_range_options,
                                              key='age_range_select',
                                              placeholder='Select age ranges...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_age_range = []
        
        # Gender filter
        if 'client_gender' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            gender_options = sorted(df['client_gender'].dropna().unique().tolist())
            selected_gender = st.multiselect('Client Gender ⚧', 
                                           options=gender_options,
                                           default=gender_options,
                                           key='gender_select',
                                           placeholder='Select genders...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_gender = []
        
        # New client filter
        if 'client_new' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            new_client_options = sorted(df['client_new'].dropna().unique().tolist())
            selected_new_client = st.multiselect('New Client 🆕', 
                                               options=new_client_options,
                                               default=new_client_options,
                                               key='new_client_select',
                                               placeholder='Select client status...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_new_client = []
        
        # Program duration filter
        if 'How long does a client receive services funded by World Jewish Relief?' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            duration_column = 'How long does a client receive services funded by World Jewish Relief?'
            duration_options = sorted(df[duration_column].dropna().unique().tolist())
            selected_duration = st.multiselect('Program Duration ⏳', 
                                             options=duration_options,
                                             default=duration_options,
                                             key='duration_select',
                                             placeholder='Select durations...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_duration = []

        # Victim of Nazism status filter
        if 'client_victim_of_nazism_status' in df.columns:
            st.markdown('<div class="selectbox-container">', unsafe_allow_html=True)
            victim_status_options = sorted(df['client_victim_of_nazism_status'].dropna().unique().tolist())
            selected_victim_status = st.multiselect('Victim of Nazism Status ⚠️', 
                                                  options=victim_status_options,
                                                  default=victim_status_options,
                                                  key='victim_status_select',
                                                  placeholder='Select status...')
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            selected_victim_status = []
    
    # Apply filters to create filtered_df
    filtered_df = filter_dataframe(df, selected_country, selected_region, selected_partner, 
                                  selected_age_range, selected_gender, 
                                  selected_new_client, selected_duration,
                                  selected_victim_status)
    
    # Add data export functionality
    st.sidebar.header(translations[language]["export_data"])
    
    if st.sidebar.button(translations[language]["download_button"]):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label=translations[language]["click_download"],
            data=csv,
            file_name="filtered_active_ageing_data.csv",
            mime="text/csv",
        )
    
    # Display basic metrics
    st.subheader(translations[language]["dataset_overview"])
    st.write(f"{translations[language]['total_records']}: {len(filtered_df)}")
    
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        translations[language]["demographics"], 
        translations[language]["program_impact"], 
        translations[language]["psychological"], 
        translations[language]["physical"],
        translations[language]["hypothesis"]
    ])
    
    # Tab 1: Demographic Analysis
    with tab1:
        st.header(translations[language]["demographics"])
        
        # Gender distribution
        if 'client_gender' in filtered_df.columns:
            st.subheader(translations[language]["gender_distribution"])
            gender_counts = filtered_df['client_gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            # Calculate percentages with safeguard
            total = gender_counts['Count'].sum()
            if total > 0:
                gender_counts['Percentage'] = (gender_counts['Count'] / total * 100).round(1)
            else:
                gender_counts['Percentage'] = 0
            
            # Add chart summary
            st.write(generate_chart_summary(gender_counts, 'gender_distribution'))
            
            fig = px.pie(gender_counts, values='Count', names='Gender', 
                         title=translations[language]["gender_distribution"],
                         labels={'Count': 'Count', 'Gender': 'Gender'},
                         custom_data=['Percentage'])
            
            # Add percentage labels
            fig.update_traces(textposition='inside', 
                            textinfo='percent+label',
                            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{customdata[0]}%<extra></extra>')
            
            st.plotly_chart(fig)
        
        # Age distribution
        if 'client_age_range' in filtered_df.columns:
            st.subheader(translations[language]["age_distribution"])
            age_counts = filtered_df['client_age_range'].value_counts().reset_index()
            age_counts.columns = ['Age Range', 'Count']
            
            # Calculate percentages with safeguard
            total = age_counts['Count'].sum()
            if total > 0:
                age_counts['Percentage'] = (age_counts['Count'] / total * 100).round(1)
            else:
                age_counts['Percentage'] = 0
            
            # Add chart summary
            st.write(generate_chart_summary(age_counts, 'age_distribution'))
            
            # Define the desired order
            age_order = ['18-54', '55-69', '70-85', '85+', 'unknown']
            
            # Create a mapping for sorting
            age_mapping = {age: i for i, age in enumerate(age_order)}
            
            # Add a temporary column for sorting
            age_counts['sort_order'] = age_counts['Age Range'].map(age_mapping)
            
            # Sort the dataframe
            age_counts = age_counts.sort_values('sort_order')
            
            # Create the bar chart with custom colors
            fig = px.bar(age_counts, x='Age Range', y='Count', 
                         title=translations[language]["age_distribution"],
                         color='Age Range',
                         color_discrete_map={
                             '18-54': '#1f77b4',
                             '55-69': '#1f77b4',
                             '70-85': '#1f77b4',
                             '85+': '#1f77b4',
                             'unknown': '#d3d3d3'
                         },
                         text=age_counts['Percentage'].astype(str) + '%')
            
            # Update layout to remove the legend and adjust text position
            fig.update_layout(showlegend=False)
            fig.update_traces(textposition='outside')
            
            st.plotly_chart(fig)
        
        # Chronic diseases visualization - enhanced for multiple-choice
        st.subheader(translations[language]["chronic_diseases"])
        chronic_columns = [
            'chronic cardiovascular', 'Diabetes', 'Respiratory diseases', 
            'Musculoskeletal disorders', 'Neurological disorders', 'Dementia',
            'Oncological diseases', 'Vision or hearing impairments', 
            'Physical limitations due to injuries or disability',
            'Other (please specify)', 'No chronic diseases or physical limitations'
        ]
        
        # Check if these columns exist in the dataset
        available_chronic_columns = [col for col in chronic_columns if col in filtered_df.columns]
        
        if available_chronic_columns:
            # Sum the values for each chronic disease type
            chronic_data = []
            total_respondents = len(filtered_df)
            if total_respondents > 0:
                for col in available_chronic_columns:
                    count = filtered_df[col].sum()
                    percentage = (count / total_respondents) * 100
                    chronic_data.append({'Disease': col, 'Count': count, 'Percentage': percentage})
            else:
                for col in available_chronic_columns:
                    chronic_data.append({'Disease': col, 'Count': 0, 'Percentage': 0})
            
            chronic_df = pd.DataFrame(chronic_data)
            
            # Sort by count in descending order
            chronic_df = chronic_df.sort_values('Count', ascending=False)
            
            # Add chart summary
            st.write(generate_chart_summary(chronic_df, 'chronic_diseases'))
            
            # Create two charts side by side - counts and percentages
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(chronic_df, x='Disease', y='Count', 
                             title=translations[language]["count_diseases"])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
            
            with col2:
                fig = px.bar(chronic_df, x='Disease', y='Percentage', 
                             title=translations[language]["percent_diseases"])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
            
            # Add a treemap for comorbidity visualization if at least 2 disease types exist
            if len(available_chronic_columns) >= 2:
                st.subheader(translations[language]["comorbidity"])
                
                # Create a correlation matrix between diseases
                disease_corr = filtered_df[available_chronic_columns].corr()
                
                fig = px.imshow(disease_corr, text_auto=True, aspect="auto",
                                title=translations[language]["correlation"])
                st.plotly_chart(fig)
    
    # Tab 2: Program Impact
    with tab2:
        st.header(translations[language]["program_impact"])
        
        # Activities participation - enhanced for multiple-choice
        st.subheader(translations[language]["activities_participation"])
        activity_columns = [
            'activity volunteers social', 'Social events', 'Interest-based clubs', 
            'Home care', 'Psychological support (group)', 'Psychological support (individual)',
            'Warm ome / Welcoming space', 'Physical rehabilitation', 'Physical activities'
        ]
        
        available_activity_columns = [col for col in activity_columns if col in filtered_df.columns]
        
        if available_activity_columns:
            activity_data = []
            total_respondents = len(filtered_df)
            if total_respondents > 0:
                for col in available_activity_columns:
                    count = filtered_df[col].sum()
                    percentage = (count / total_respondents) * 100
                    activity_data.append({'Activity': col, 'Count': count, 'Percentage': percentage})
            else:
                for col in available_activity_columns:
                    activity_data.append({'Activity': col, 'Count': 0, 'Percentage': 0})
            
            activity_df = pd.DataFrame(activity_data)
            activity_df = activity_df.sort_values('Count', ascending=False)
            
            # Add chart summary
            st.write(generate_chart_summary(activity_df, 'activities'))
            
            # Count visualization for activities
            fig = px.bar(activity_df, x='Activity', y='Count', 
                         title=translations[language]["activities_count"],
                         text=activity_df['Percentage'].round(1).astype(str) + '%',
                         labels={'Activity': translations[language]["activity"], 
                                'Count': translations[language]["count"]})
            
            # Customize the appearance
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(t=50, b=100),  # Increase top and bottom margins
                height=500  # Increase overall height of the chart
            )
            st.plotly_chart(fig)
        
        # Services received - enhanced for multiple-choice
        st.subheader(translations[language]["services_received"])
        service_columns = [
            'None of the above', 'Purchase of tablets or phones', 
            'Mobility or exercise equipment', 'Medical treatment', 'Medical devices',
            'Delivery of food packages/hot meals', 'Medical support',
            'Delivery of blood pressure/oximeters', 'Winter assistance',
            'Other material assistance'
        ]
        
        available_service_columns = [col for col in service_columns if col in filtered_df.columns]
        
        if available_service_columns:
            service_data = []
            total_respondents = len(filtered_df)
            if total_respondents > 0:
                for col in available_service_columns:
                    count = filtered_df[col].sum()
                    percentage = (count / total_respondents) * 100
                    service_data.append({'Service': col, 'Count': count, 'Percentage': percentage})
            else:
                for col in available_service_columns:
                    service_data.append({'Service': col, 'Count': 0, 'Percentage': 0})
            
            service_df = pd.DataFrame(service_data)
            service_df = service_df.sort_values('Count', ascending=False)
            
            # Add chart summary
            st.write(generate_chart_summary(service_df, 'services'))
            
            # Count visualization for services
            fig = px.bar(service_df, x='Service', y='Count', 
                         title=translations[language]["services_count"],
                         text=service_df['Percentage'].round(1).astype(str) + '%',
                         labels={'Service': translations[language]["service"], 
                                'Count': translations[language]["count"]})
            
            # Customize the appearance
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(t=50, b=100),  # Increase top and bottom margins
                height=500  # Increase overall height of the chart
            )
            st.plotly_chart(fig)
        
        # Overall help rating
        if 'How would you rate the help you receive (1 - it\'s not enough, 5 - everything is perfect)?' in filtered_df.columns:
            rating_col = 'How would you rate the help you receive (1 - it\'s not enough, 5 - everything is perfect)?'
            st.subheader("Overall Help Rating")
            
            # Show average rating
            avg_rating = filtered_df[rating_col].mean()
            st.metric("Average Rating", f"{avg_rating:.2f} / 5.0")
            
            # Distribution of ratings
            rating_counts = filtered_df[rating_col].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            
            # Calculate percentage with safeguard
            total = rating_counts['Count'].sum()
            if total > 0:
                rating_counts['Percentage'] = (rating_counts['Count'] / total * 100)
            else:
                rating_counts['Percentage'] = 0
            
            # Add chart summary
            st.write(generate_chart_summary(rating_counts, 'rating', 
                                          "generally positive feedback" if avg_rating >= 3.5 else "mixed feedback"))
            
            # Create a more visually appealing rating distribution
            fig = px.bar(rating_counts, x='Rating', y='Count', 
                         title='Distribution of Overall Help Ratings',
                         text=rating_counts['Percentage'].round(1).astype(str) + '%',
                         color='Rating',
                         color_continuous_scale='RdYlGn')
            
            # Customize the appearance
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title='Rating (1-5)',
                yaxis_title='Number of Responses',
                showlegend=False,
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['1 - Not Enough', '2', '3', '4', '5 - Perfect']
                )
            )
            
            st.plotly_chart(fig)
    
    # Tab 3: Psychological State
    with tab3:
        st.header(translations[language]["psychological"])
        
        # Physical activity frequency
        if 'Как часто вы участвуете в мероприятиях, которые касаються улучшения физического здоровья, организуемых нашими партнерами?' in filtered_df.columns:
            activity_freq_col = 'Как часто вы участвуете в мероприятиях, которые касаються улучшения физического здоровья, организуемых нашими партнерами?'
            st.subheader(translations[language]["physical_freq"])
            
            freq_counts = filtered_df[activity_freq_col].value_counts().reset_index()
            freq_counts.columns = ['Frequency', 'Count']
            
            fig = px.pie(freq_counts, values='Count', names='Frequency', 
                         title=translations[language]["physical_freq"])
            st.plotly_chart(fig)
            
            # Add analysis by age group if available
            if 'client_age_range' in filtered_df.columns:
                # Create cross-tabulation
                activity_by_age = pd.crosstab(
                    filtered_df['client_age_range'], 
                    filtered_df[activity_freq_col],
                    normalize='index'
                ) * 100  # Convert to percentages
                
                # Plot as stacked bar chart
                fig = px.bar(activity_by_age, 
                             title=translations[language]["freq_by_age"],
                             labels={'value': '%', 'client_age_range': translations[language]["age_range"]})
                
                # Add chart summary
                most_active_age = activity_by_age['Several times a week'].idxmax()
                highest_percentage = activity_by_age['Several times a week'].max()
                st.write(f"The {most_active_age} age group shows the highest frequency of multiple weekly activities ({highest_percentage:.1f}%).")
                
                st.plotly_chart(fig)
        
        # Social connections impact with deeper analysis
        if 'Как программа повлияла на ваши социальные связи или сеть контактов? ' in filtered_df.columns:
            social_col = 'Как программа повлияла на ваши социальные связи или сеть контактов? '
            st.subheader(translations[language]["social_impact"])
            
            social_counts = filtered_df[social_col].value_counts().reset_index()
            social_counts.columns = ['Impact', 'Count']
            
            # Calculate percentages with safeguard
            total = social_counts['Count'].sum()
            if total > 0:
                social_counts['Percentage'] = (social_counts['Count'] / total * 100)
            else:
                social_counts['Percentage'] = 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(social_counts, x='Impact', y='Count', 
                             title=translations[language]["social_count"])
                st.plotly_chart(fig)
            
            with col2:
                fig = px.pie(social_counts, values='Count', names='Impact', 
                             title=translations[language]["social_percent"])
                st.plotly_chart(fig)
            
            # Analysis by demographics if available
            if 'client_gender' in filtered_df.columns:
                st.write(translations[language]["social_by_gender"])
                
                # Create cross-tabulation
                social_by_gender = pd.crosstab(
                    filtered_df['client_gender'], 
                    filtered_df[social_col],
                    normalize='index'
                ) * 100  # Convert to percentages
                
                # Remove 'unknown' category if it exists
                if 'unknown' in social_by_gender.index:
                    social_by_gender = social_by_gender.drop('unknown')
                
                # Plot as grouped bar chart
                fig = px.bar(social_by_gender, 
                             barmode='group',
                             title=translations[language]["social_by_gender"],
                             labels={'value': '%', 'client_gender': translations[language]["gender"]})
                
                st.plotly_chart(fig)
        
        # Made friends in program
        if 'Вы завели друзей или контакты, посещая занятия? ' in filtered_df.columns:
            friends_col = 'Вы завели друзей или контакты, посещая занятия? '
            st.subheader("Made Friends in Program")
            
            friends_counts = filtered_df[friends_col].value_counts().reset_index()
            friends_counts.columns = ['Response', 'Count']
            
            fig = px.pie(friends_counts, values='Count', names='Response', 
                         title='Made Friends Through Program')
            st.plotly_chart(fig)
    
    # Tab 4: Physical State
    with tab4:
        st.header(translations[language]["physical"])
        
        # Physical support rating with enhanced analysis
        if 'Если вы получали оборудование, лекарства или другую медицинскую помощь, как вы думаете, это повлияло на ваше физическое состояние и уровень жизни по шкале от 1 до 10 (1 — не повлияло вообще, 10 — максимальный эффект)? ' in filtered_df.columns:
            physical_rating_col = 'Если вы получали оборудование, лекарства или другую медицинскую помощь, как вы думаете, это повлияло на ваше физическое состояние и уровень жизни по шкале от 1 до 10 (1 — не повлияло вообще, 10 — максимальный эффект)? '
            st.subheader(translations[language]["physical_support"])
            
            # Show average rating
            avg_physical_rating = filtered_df[physical_rating_col].mean()
            st.metric(translations[language]["avg_rating"], f"{avg_physical_rating:.2f} / 10.0")
            
            # Distribution of ratings
            rating_counts = filtered_df[physical_rating_col].value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            
            # Calculate percentage with safeguard
            total = rating_counts['Count'].sum()
            if total > 0:
                rating_counts['Percentage'] = (rating_counts['Count'] / total * 100)
            else:
                rating_counts['Percentage'] = 0
            
            # Create a line graph with markers
            fig = go.Figure()
            
            # Add line with markers
            fig.add_trace(go.Scatter(
                x=rating_counts['Rating'],
                y=rating_counts['Count'],
                mode='lines+markers+text',
                name='Count',
                text=rating_counts['Percentage'].round(1).astype(str) + '%',
                textposition='top center',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=10)
            ))
            
            # Update layout
            fig.update_layout(
                title=translations[language]["phys_support_title"],
                xaxis_title=translations[language]["rating"] + ' (1-10)',
                yaxis_title=translations[language]["count"],
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(1, 11)),
                    ticktext=['1 - No Effect'] + list(range(2, 10)) + ['10 - Maximum']
                ),
                showlegend=False,
                height=500,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig)
            
            # Health control level
            if 'Чувствуете ли вы, что контролируете свое физическое здоровье благодаря нашей программе?' in filtered_df.columns:
                health_control_col = 'Чувствуете ли вы, что контролируете свое физическое здоровье благодаря нашей программе?'
                st.subheader("Health Control Level")
                
                health_control_counts = filtered_df[health_control_col].value_counts().reset_index()
                health_control_counts.columns = ['Response', 'Count']
                
                fig = px.pie(health_control_counts, values='Count', names='Response', 
                             title='Control Over Physical Health')
                st.plotly_chart(fig)
            
            # General health description
            if 'В целом, как бы Вы охарактеризовали своё здоровье?' in filtered_df.columns:
                general_health_col = 'В целом, как бы Вы охарактеризовали своё здоровье?'
                st.subheader("General Health Self-Assessment")
                
                # Create a mapping for correct order and translations
                health_mapping = {
                    '1. Отличное': {'order': 1, 'en': 'Excellent', 'uk': 'Відмінне', 'ru': 'Отличное'},
                    '2. Очень хорошее': {'order': 2, 'en': 'Very good', 'uk': 'Дуже добре', 'ru': 'Очень хорошее'},
                    '3. Хорошее': {'order': 3, 'en': 'Good', 'uk': 'Добре', 'ru': 'Хорошее'},
                    '4. Удовлетворительное': {'order': 4, 'en': 'Satisfactory', 'uk': 'Задовільне', 'ru': 'Удовлетворительное'},
                    '5. Плохое': {'order': 5, 'en': 'Bad', 'uk': 'Погане', 'ru': 'Плохое'},
                    '4. Satisfactory': {'order': 4, 'en': 'Satisfactory', 'uk': 'Задовільне', 'ru': 'Удовлетворительное'},
                    '3. Good': {'order': 3, 'en': 'Good', 'uk': 'Добре', 'ru': 'Хорошее'},
                    '2. Very good': {'order': 2, 'en': 'Very good', 'uk': 'Дуже добре', 'ru': 'Очень хорошее'},
                    '1. Excellent': {'order': 1, 'en': 'Excellent', 'uk': 'Відмінне', 'ru': 'Отличное'},
                    '5. Bad': {'order': 5, 'en': 'Bad', 'uk': 'Погане', 'ru': 'Плохое'}
                }
                
                # Create a mapping for language codes
                language_code_map = {
                    'English': 'en',
                    'Ukrainian': 'uk',
                    'Russian': 'ru'
                }
                
                # Get counts and add sorting column
                general_health_counts = filtered_df[general_health_col].value_counts().reset_index()
                general_health_counts.columns = ['Assessment', 'Count']
                
                # Add sorting order and translate labels
                general_health_counts['sort_order'] = general_health_counts['Assessment'].map(lambda x: health_mapping[x]['order'])
                general_health_counts['Assessment'] = general_health_counts['Assessment'].map(lambda x: health_mapping[x][language_code_map[language]])
                
                # Sort by the defined order
                general_health_counts = general_health_counts.sort_values('sort_order')
                
                # Calculate percentages with safeguard
                total = general_health_counts['Count'].sum()
                if total > 0:
                    general_health_counts['Percentage'] = (general_health_counts['Count'] / total * 100).round(1)
                else:
                    general_health_counts['Percentage'] = 0
                
                # Create the bar chart
                fig = px.bar(general_health_counts, x='Assessment', y='Count', 
                             title='General Health Self-Assessment',
                             text=general_health_counts['Percentage'].astype(str) + '%')
                
                # Update layout
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    xaxis_title='Assessment',
                    yaxis_title='Number of Responses',
                    xaxis={'categoryorder': 'array', 
                          'categoryarray': [health_mapping[x][language_code_map[language]] for x in 
                                          ['1. Отличное', '2. Очень хорошее', '3. Хорошее', 
                                           '4. Удовлетворительное', '5. Плохое']]}
                )
                
                st.plotly_chart(fig)
    
    # Tab 5: Hypothesis Testing
    with tab5:
        st.header(translations[language]["hypothesis"])
        
        # Hypothesis 1: Younger vs Older clients' activity preferences
        st.subheader(translations[language]["hyp1_title"])
        st.write(translations[language]["hyp1_desc"])
        
        if 'client_age_range' in filtered_df.columns:
            # Check for physical activity columns
            physical_cols = ['Physical rehabilitation', 'Physical activities']
            available_physical = [col for col in physical_cols if col in filtered_df.columns]
            
            # Check for psychosocial activity columns
            psycho_cols = ['Social events', 'Psychological support (group)', 'Psychological support (individual)']
            available_psycho = [col for col in psycho_cols if col in filtered_df.columns]
            
            if available_physical and available_psycho:
                # Add composite columns with safeguards
                filtered_df['physical_activities_sum'] = filtered_df[available_physical].sum(axis=1)
                filtered_df['psychosocial_activities_sum'] = filtered_df[available_psycho].sum(axis=1)
                
                # Normalize by creating percentage columns with safeguards
                if len(available_physical) > 0:
                    filtered_df['physical_percent'] = filtered_df['physical_activities_sum'] / len(available_physical)
                else:
                    filtered_df['physical_percent'] = 0
                
                if len(available_psycho) > 0:
                    filtered_df['psychosocial_percent'] = filtered_df['psychosocial_activities_sum'] / len(available_psycho)
                else:
                    filtered_df['psychosocial_percent'] = 0
                
                # Remove unknown age range
                filtered_df_no_unknown = filtered_df[filtered_df['client_age_range'] != 'unknown']
                
                # Group by age range
                age_activity_pref = filtered_df_no_unknown.groupby('client_age_range').agg({
                    'physical_percent': 'mean',
                    'psychosocial_percent': 'mean',
                    'ID': 'count'  # Count of clients in each age group
                }).reset_index()
                
                age_activity_pref.rename(columns={'ID': 'client_count'}, inplace=True)
                
                # Convert to percentages for display
                age_activity_pref['physical_percent'] = age_activity_pref['physical_percent'] * 100
                age_activity_pref['psychosocial_percent'] = age_activity_pref['psychosocial_percent'] * 100
                
                # Create a grouped bar chart
                fig = px.bar(age_activity_pref, x='client_age_range', 
                             y=['physical_percent', 'psychosocial_percent'],
                             title=translations[language]["pref_by_age"],
                             labels={
                                 'value': translations[language]["participation_rate"], 
                                 'variable': translations[language]["activity_type"], 
                                 'client_age_range': translations[language]["age_range"]
                             },
                             barmode='group')
                
                # Update names in legend
                fig.update_traces(name=translations[language]["physical_activities"], 
                                 selector=dict(name='physical_percent'))
                fig.update_traces(name=translations[language]["psychosocial_activities"], 
                                 selector=dict(name='psychosocial_percent'))
                
                st.plotly_chart(fig)
                
                # Analysis and conclusion
                younger_physical = age_activity_pref.loc[age_activity_pref['client_age_range'].isin(['18-54', '55-69']), 'physical_percent'].mean()
                older_physical = age_activity_pref.loc[age_activity_pref['client_age_range'].isin(['70-85', '85+']), 'physical_percent'].mean()
                
                younger_psycho = age_activity_pref.loc[age_activity_pref['client_age_range'].isin(['18-54', '55-69']), 'psychosocial_percent'].mean()
                older_psycho = age_activity_pref.loc[age_activity_pref['client_age_range'].isin(['70-85', '85+']), 'psychosocial_percent'].mean()
                
                if younger_physical > older_physical and older_psycho > younger_psycho:
                    conclusion = "The data supports the hypothesis. Younger clients tend to participate more in physical activities, while older clients prefer psychosocial activities."
                elif younger_physical > older_physical:
                    conclusion = "The data partially supports the hypothesis. Younger clients do participate more in physical activities, but there's no clear preference for psychosocial activities among older clients."
                elif older_psycho > younger_psycho:
                    conclusion = "The data partially supports the hypothesis. Older clients do prefer psychosocial activities, but there's no clear preference for physical activities among younger clients."
                else:
                    conclusion = "The data does not support the hypothesis. The observed preferences don't align with the expected pattern."
                
                st.write(f"**{translations[language]['conclusion']}** {conclusion}")
        
        # Hypothesis 2: New clients satisfaction
        st.subheader("Hypothesis #2: New Clients Satisfaction")
        st.write("""
        **Hypothesis:** "New clients (less than 3 months in the program) have a lower level of overall satisfaction."
        """)
        
        duration_col = 'How long does a client receive services funded by World Jewish Relief?'
        rating_col = 'How would you rate the help you receive (1 - it\'s not enough, 5 - everything is perfect)?'
        
        if duration_col in filtered_df.columns and rating_col in filtered_df.columns:
            # Create new client flag (less than 3 months)
            filtered_df['is_new_client'] = filtered_df[duration_col].str.contains('1-3', na=False)
            
            # Calculate average ratings for new vs existing clients
            satisfaction_by_duration = filtered_df.groupby('is_new_client')[rating_col].agg(['mean', 'count']).round(2)
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                new_clients = satisfaction_by_duration.loc[True] if True in satisfaction_by_duration.index else pd.Series({'mean': 0, 'count': 0})
                st.metric(
                    "New Clients (1-3 months)",
                    f"{new_clients['mean']:.2f}",
                    help=f"Based on {int(new_clients['count'])} responses"
                )
            
            with col2:
                existing_clients = satisfaction_by_duration.loc[False] if False in satisfaction_by_duration.index else pd.Series({'mean': 0, 'count': 0})
                st.metric(
                    "Existing Clients (>3 months)",
                    f"{existing_clients['mean']:.2f}",
                    help=f"Based on {int(existing_clients['count'])} responses"
                )
            
            # Add conclusion
            if new_clients['mean'] < existing_clients['mean']:
                st.write("**Conclusion:** The hypothesis is supported. New clients (1-3 months) show lower satisfaction levels compared to existing clients.")
            else:
                st.write("**Conclusion:** The hypothesis is not supported. New clients (1-3 months) do not show lower satisfaction levels compared to existing clients.")
        
        # Hypothesis 3: Longer programme better health
        st.subheader("Hypothesis #3: Longer Programme Better Health")
        st.write("""
        **Hypothesis:** "The longer a client participates in the program, the better their self-assessed health status."
        """)
        
        health_col = 'В целом, как бы Вы охарактеризовали своё здоровье?'
        
        if duration_col in filtered_df.columns and health_col in filtered_df.columns:
            # Create health score mapping (5 is best, 1 is worst)
            health_score_mapping = {
                '1. Отличное': 5,
                '2. Очень хорошее': 4,
                '3. Хорошее': 3,
                '4. Удовлетворительное': 2,
                '5. Плохое': 1,
                '1. Excellent': 5,
                '2. Very good': 4,
                '3. Good': 3,
                '4. Satisfactory': 2,
                '5. Bad': 1
            }
            
            # Create duration order mapping
            duration_order = {
                '1-3 months': 1,
                '3-6 months': 2,
                '6-12 months': 3,
                '1-2 years': 4,
                'More than 2 years': 5,
                'more than 2 years': 5  # Add lowercase variant
            }
            
            # Create numeric health score
            filtered_df['health_score'] = filtered_df[health_col].map(health_score_mapping)
            
            # Calculate average health score by duration
            health_by_duration = filtered_df.groupby(duration_col).agg({
                'health_score': ['mean', 'count']
            }).round(2)
            
            health_by_duration.columns = ['Average Health Score', 'Count']
            health_by_duration = health_by_duration.reset_index()
            
            # Sort by duration order
            health_by_duration['sort_order'] = health_by_duration[duration_col].map(duration_order)
            health_by_duration = health_by_duration.sort_values('sort_order')
            
            # Create visualization
            fig = px.bar(health_by_duration, 
                        x=duration_col, 
                        y='Average Health Score',
                        text='Average Health Score',
                        title='Average Health Score by Program Duration')
            
            # Update layout to fix the y-axis scale
            fig.update_layout(
                yaxis=dict(
                    range=[1, 5],  # Set fixed range from 1 to 5
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                xaxis_tickangle=-45
            )
            
            fig.update_traces(
                textposition='outside',
                texttemplate='%{text:.2f}'
            )
            
            st.plotly_chart(fig)
            
            # Add sample size information
            st.write("Sample size for each duration group:")
            for _, row in health_by_duration.iterrows():
                st.write(f"{row[duration_col]}: {int(row['Count'])} respondents")
            
            # Calculate correlation
            filtered_df['duration_score'] = filtered_df[duration_col].map(duration_order)
            correlation = filtered_df['duration_score'].corr(filtered_df['health_score'])
            
            st.write(f"Correlation coefficient between program duration and health score: {correlation:.3f}")
            
            # Provide conclusion based on correlation and visual trend
            if abs(correlation) < 0.1:
                st.write("**Conclusion:** The hypothesis is not supported. There is no significant correlation between program duration and self-assessed health status.")
            elif correlation > 0.1:
                st.write("**Conclusion:** The hypothesis is supported. There is a positive correlation between program duration and self-assessed health status.")
            else:
                st.write("**Conclusion:** The hypothesis is not supported. There is a slight negative correlation between program duration and self-assessed health status.")

# Run the app
if __name__ == '__main__':
    main()