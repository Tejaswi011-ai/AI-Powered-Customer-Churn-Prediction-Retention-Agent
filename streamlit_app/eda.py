import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CLEAN_DATA_PATH = "D:/customer_churn_ai_agent/data/processed_churn_data.csv"

def show_eda():
    st.markdown(
        "<div style='margin-bottom: 25px;'>"
        "<h1>📊 Exploratory Data Analysis</h1>"
        "<p style='color: #94a3b8; font-size: 16px; margin-top: 5px;'>"
        "Unlock key operational insights and customer patterns behind churn dynamics using interactive filtering and rich Plotly visualizations."
        "</p>"
        "</div>", 
        unsafe_allow_html=True
    )
    
    # 1. Verify that preprocessed data exists
    if not os.path.exists(CLEAN_DATA_PATH):
        st.warning("⚠️ Cleaned dataset is not available yet.")
        st.info("💡 To generate this page's interactive dashboards, go to the ⚙️ 'Model Training & Comparison' page and trigger the preprocessing and training sequence. This will programmatically cache and clean the raw dataset.")
        return
        
    # Load dataset
    @st.cache_data
    def load_eda_data():
        df = pd.read_csv(CLEAN_DATA_PATH)
        # Create user-friendly Churn label for charts
        df['Churn_Label'] = df['Churn'].map({1: "Churned (Yes)", 0: "Stayed (No)"})
        return df
        
    df = load_eda_data()
    
    # 2. Interactive Filtering sidebar/panel
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>🔍 Interactive Cohort Filter</h3>", unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        gender_opt = ["All"] + list(df['gender'].unique())
        selected_gender = st.selectbox("Filter by Gender", gender_opt)
    with col_f2:
        contract_opt = ["All"] + list(df['Contract'].unique())
        selected_contract = st.selectbox("Filter by Contract Type", contract_opt)
    with col_f3:
        internet_opt = ["All"] + list(df['InternetService'].unique())
        selected_internet = st.selectbox("Filter by Internet Infrastructure", internet_opt)
        
    # Apply filters
    filtered_df = df.copy()
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
    if selected_contract != "All":
        filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
    if selected_internet != "All":
        filtered_df = filtered_df[filtered_df['InternetService'] == selected_internet]
        
    # Display record counts
    st.markdown(
        f"<div style='font-size: 14px; color: #a78bfa; font-weight: 600; margin-top: 5px;'>"
        f"Showing {len(filtered_df):,} out of {len(df):,} customer records ({len(filtered_df)/len(df)*100:.1f}%)"
        f"</div>", 
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Plotly Charts Rows
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Chart 1: Donut Churn Distribution
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Churn Distribution Ratio</h4>", unsafe_allow_html=True)
        
        churn_counts = filtered_df['Churn_Label'].value_counts().reset_index()
        churn_counts.columns = ['Status', 'Count']
        
        fig1 = px.pie(
            churn_counts, 
            values='Count', 
            names='Status',
            hole=0.45,
            color='Status',
            color_discrete_map={"Stayed (No)": "#3b82f6", "Churned (Yes)": "#ef4444"}
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart2:
        # Chart 2: Stacked Bar Chart Contract vs Churn
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Attrition by Billing Contract Type</h4>", unsafe_allow_html=True)
        
        contract_churn = filtered_df.groupby(['Contract', 'Churn_Label']).size().reset_index(name='Count')
        
        fig2 = px.bar(
            contract_churn, 
            x='Contract', 
            y='Count', 
            color='Churn_Label',
            barmode='group',
            color_discrete_map={"Stayed (No)": "#3b82f6", "Churned (Yes)": "#ef4444"}
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(title='Contract Term', showgrid=False),
            yaxis=dict(title='Number of Customers', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Second chart row
    col_chart3, col_chart4 = st.columns(2)
    
    with col_chart3:
        # Chart 3: Monthly Charges Box Plot
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Monthly Charges Distribution by Churn Status</h4>", unsafe_allow_html=True)
        
        fig3 = px.box(
            filtered_df,
            x='Churn_Label',
            y='MonthlyCharges',
            color='Churn_Label',
            color_discrete_map={"Stayed (No)": "#3b82f6", "Churned (Yes)": "#ef4444"}
        )
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(title='Churn Status', showgrid=False),
            yaxis=dict(title='Monthly Charges ($)', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart4:
        # Chart 4: Tenure Distribution Histogram
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Customer Tenure Distribution (Months)</h4>", unsafe_allow_html=True)
        
        fig4 = px.histogram(
            filtered_df,
            x='tenure',
            color='Churn_Label',
            nbins=30,
            color_discrete_map={"Stayed (No)": "#3b82f6", "Churned (Yes)": "#ef4444"},
            marginal="rug" # Adds a density representation at bottom
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(title='Tenure (Months)', showgrid=False),
            yaxis=dict(title='Customer Count', showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            margin=dict(t=20, b=20, l=10, r=10),
            height=300
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. Interactive Data Table Row
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>📋 Cohort Records Explorer</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-bottom: 15px;'>Select features and browse raw client details for the filtered cohort.</p>", unsafe_allow_html=True)
    
    view_cols = st.multiselect(
        "Select columns to display", 
        list(df.columns), 
        default=['customerID', 'gender', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'InternetService', 'Churn_Label']
    )
    
    if len(view_cols) > 0:
        st.dataframe(
            filtered_df[view_cols].head(100), # Cap at 100 rows for smooth UI rendering
            use_container_width=True,
            column_config={
                "MonthlyCharges": st.column_config.NumberColumn(format="$%.2f"),
                "TotalCharges": st.column_config.NumberColumn(format="$%.2f"),
                "Churn_Label": st.column_config.TextColumn(label="Churn Status")
            }
        )
        st.markdown("<span style='font-size: 11px; color: #94a3b8;'>*Displaying top 100 records in filtered dataset for fluid UI response times.</span>", unsafe_allow_html=True)
    else:
        st.info("Select at least one feature column to render the table.")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    show_eda()
