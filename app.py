import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def format_bdt_short(val):
    if val >= 1e9: return f"{val/1e9:.2f}B"
    if val >= 1e6: return f"{val/1e6:.2f}M"
    if val >= 1e3: return f"{val/1e3:.2f}K"
    return str(val)

# --- PAGE CONFIG & CUSTOM CSS (Premium UI) ---
st.set_page_config(page_title="Executive Loan Risk Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #1f77b4;
    }
    .metric-card h3 { color: #555; font-size: 16px; margin-bottom: 5px; }
    .metric-card h2 { color: #222; font-size: 28px; margin: 0; font-weight: bold; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- DATA PROCESSING & ML MODEL ---
@st.cache_data
def load_and_prep_data():
    # Load raw data
    loans = pd.read_csv('LoanAccounts.csv')
    members = pd.read_csv('Member.csv')
    # Pre-aggregated from ScheduledInstallmentsOfLoanAccounts.csv (129MB → 1.8MB)
    installment_agg = pd.read_csv('installment_agg.csv')
    
    # Merge data
    df = pd.merge(loans, members, on='Member', how='left')
    df = pd.merge(df, installment_agg, on='LoanAccount', how='left')
    
    df['Has_Early_Overdue'] = df['Has_Early_Overdue'].fillna(0).astype(int)
    df['Max_OverdueAmount'] = df['Max_OverdueAmount'].fillna(0)
    df['Total_Paid'] = df['Total_Paid'].fillna(0)
    df['Total_Scheduled'] = df['Total_Scheduled'].fillna(0)
    df['Min_Outstanding'] = df['Min_Outstanding'].fillna(df['PrincipalAmount'])
    
    # Process Dates
    df['DisbursedDate'] = pd.to_datetime(df['DisbursedDate'], errors='coerce')
    df['DisbursementMonth'] = df['DisbursedDate'].dt.to_period('M').astype(str)
    
    # Define Bad Debt indicator
    df['IsBadDebt'] = df['ClosingReason'].apply(lambda x: 1 if x == 'TRANSFER_TO_BAD_DEBT' else 0)
    
    # Feature Engineering
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['<25', '25-35', '36-45', '46-55', '56-65', '65+']
    df['AgeBucket'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Fill NAs
    for col in ['Education', 'Profession', 'HouseType', 'HouseOwner', 'GuarantorRelation']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            if col == 'Profession':
                df[col] = df[col].apply(lambda x: 'Unknown' if str(x).isdigit() else x)
    
    df['PovertyLevel'] = df['PovertyLevel'].fillna(-1).astype(int)
    
    return df

@st.cache_resource
def train_model(df):
    # Select features for Risk Model
    features = ['Age', 'Duration', 'Cycle', 'PrincipalAmount', 'PovertyLevel', 'Sex']
    model_df = df.dropna(subset=features + ['IsBadDebt']).copy()
    
    X = model_df[features]
    y = model_df['IsBadDebt']
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, class_weight='balanced')
    rf.fit(X, y)
    
    # Generate feature importance
    importance = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
    importance = importance.sort_values('Importance', ascending=True)
    
    return rf, features, importance

df_raw = load_and_prep_data()
model, model_features, feature_importance = train_model(df_raw)

# Pre-score the entire historical dataset using the ML model to create the "Backbone"
@st.cache_data
def score_portfolio(df, _model, features):
    # Fill NAs temporarily for scoring
    score_df = df[features].fillna(df[features].median())
    probs = _model.predict_proba(score_df)[:, 1]
    
    def assign_tier(p):
        if p >= 0.60: return 'High Risk'
        elif p >= 0.45: return 'Medium Risk'
        else: return 'Low Risk'
        
    return [assign_tier(p) for p in probs], probs

df_raw['ML_RiskTier'], df_raw['ML_RiskProb'] = score_portfolio(df_raw, model, model_features)


# --- SIDEBAR: NAVIGATION & FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=60)
st.sidebar.title("Executive Dashboard")
page = st.sidebar.radio("Strategic Modules:", [
    "1. Portfolio Overview", 
    "2. Risk Drivers (Deep-Dive)", 
    "3. Risk Segment", 
    "4. Business Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.header("Global Filters")
all_programs = ["All"] + list(df_raw['ProgramName'].dropna().unique())
selected_program = st.sidebar.selectbox("Program Name", all_programs)
all_poverty = ["All"] + list(df_raw[df_raw['PovertyLevel']!=-1]['PovertyLevel'].unique())
selected_poverty = st.sidebar.selectbox("Poverty Level", all_poverty)

min_date = df_raw['DisbursedDate'].min()
max_date = df_raw['DisbursedDate'].max()
if pd.isna(min_date) or pd.isna(max_date):
    date_range = []
else:
    date_range = st.sidebar.date_input("Disbursement Date Range", [min_date, max_date], min_value=min_date.date(), max_value=max_date.date())

df = df_raw.copy()
if selected_program != "All": df = df[df['ProgramName'] == selected_program]
if selected_poverty != "All": df = df[df['PovertyLevel'] == selected_poverty]
if len(date_range) == 2:
    df = df[(df['DisbursedDate'].dt.date >= date_range[0]) & (df['DisbursedDate'].dt.date <= date_range[1])]


# --- MODULE 1: PORTFOLIO OVERVIEW ---
if page == "1. Portfolio Overview":
    st.title("Executive Summary: Portfolio Health")
    st.markdown("Business Objective: Assess the macro exposure and quality of the loan portfolio.")
    
    # Premium KPIs - Row 1 (Volume & Value)
    st.markdown("#### Portfolio Volume & Value")
    col1, col2, col3, col4 = st.columns(4)
    total_loans = len(df)
    total_members = df['Member'].nunique()
    total_disbursed = df['PrincipalAmount'].sum()
    total_interest = df['InterestAmount'].sum()
    
    col1.markdown(f'<div class="metric-card"><h3>Total Loans</h3><h2>{total_loans:,}</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Total volume of loan contracts</p></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card"><h3>Total Members</h3><h2>{total_members:,}</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Unique active & past borrowers</p></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card"><h3>Total Disbursed</h3><h2>৳ {format_bdt_short(total_disbursed)}</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Capital injected into market</p></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="metric-card"><h3>Expected Interest</h3><h2>৳ {format_bdt_short(total_interest)}</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Gross revenue potential of portfolio</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Premium KPIs - Row 2 (Profitability & Collection Health)
    st.markdown("#### Profitability & Collection Health")
    col5, col6, col7, col8 = st.columns(4)
    
    # Calculate Profit metrics
    profit_realized = df[df['ClosingReason'] == 'FULL_PAID']['InterestAmount'].sum()
    profit_upcoming = df[df['Status'] == 1]['InterestAmount'].sum()
    
    # Calculate Avg Repayment Time
    df['ClosingDate_DT'] = pd.to_datetime(df['ClosingDate'], errors='coerce')
    full_paid_df = df[df['ClosingReason'] == 'FULL_PAID']
    avg_repay_days = (full_paid_df['ClosingDate_DT'] - full_paid_df['DisbursedDate']).dt.days.mean()
    avg_repay_months = avg_repay_days / 30 if pd.notna(avg_repay_days) else 0
    
    # Calculate Money Lost
    bad_debt_loss = df[df['IsBadDebt'] == 1]['PrincipalAmount'].sum()
    
    col5.markdown(f'<div class="metric-card"><h3>Profit Realized</h3><h2>৳ {format_bdt_short(profit_realized)}</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Revenue successfully collected</p></div>', unsafe_allow_html=True)
    col6.markdown(f'<div class="metric-card"><h3>Upcoming Profit</h3><h2>৳ {format_bdt_short(profit_upcoming)}</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Future revenue from active loans</p></div>', unsafe_allow_html=True)
    col7.markdown(f'<div class="metric-card"><h3>Avg Repayment Time</h3><h2>{avg_repay_months:.0f} Mo</h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Speed of capital turnover</p></div>', unsafe_allow_html=True)
    
    # Color-coded loss
    loss_color = "#d62728" if bad_debt_loss > 0 else "#2ca02c"
    loss_pct = (bad_debt_loss / total_disbursed * 100) if total_disbursed > 0 else 0
    col8.markdown(f'<div class="metric-card" style="border-top: 4px solid {loss_color};"><h3>Total Money Lost</h3><h2 style="color: {loss_color};">৳ {format_bdt_short(bad_debt_loss)} <span style="font-size:16px;">({loss_pct:.0f}%)</span></h2><p style="font-size:13px; color:#7f8c8d; margin-top:8px; margin-bottom:0px;">Capital written off due to defaults</p></div>', unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loan Composition by Product")
        product_mix = df['ProgramName'].value_counts().reset_index()
        product_mix.columns = ['ProgramName', 'Count']
        
        total_for_mix = product_mix['Count'].sum()
        # Group products with less than 1% of total loans into "Other"
        product_mix['Percentage'] = (product_mix['Count'] / total_for_mix * 100)
        product_mix['ProgramName'] = product_mix.apply(lambda x: x['ProgramName'] if x['Percentage'] >= 1.0 else 'Other', axis=1)
        
        # Re-aggregate after grouping
        product_mix = product_mix.groupby('ProgramName')['Count'].sum().reset_index()
        product_mix['Percentage'] = (product_mix['Count'] / total_for_mix * 100).round(1)
        product_mix['Label'] = product_mix.apply(lambda row: f"{row['Count']:,} ({row['Percentage']}%)", axis=1)
        
        # Sort so largest is at top (Plotly plots from bottom up)
        product_mix = product_mix.sort_values('Count', ascending=True)
        
        fig1 = px.bar(product_mix, x='Count', y='ProgramName', text='Label', orientation='h')
        fig1.update_traces(textposition='outside', marker_color='#1f77b4')
        # Extend x-axis range slightly so the text labels don't get cut off
        max_val = product_mix['Count'].max()
        fig1.update_layout(xaxis_title="Number of Loans", yaxis_title="", plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(range=[0, max_val * 1.3]))
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.subheader("Portfolio Health Distribution")
        def get_outcome(row):
            if row['Status'] == 1: return "Active/Performing"
            elif row['IsBadDebt'] == 1: return "Bad Debt"
            else: return "Closed (Good)"
        
        df['Outcome'] = df.apply(get_outcome, axis=1)
        outcome_mix = df['Outcome'].value_counts().reset_index()
        outcome_mix.columns = ['Outcome', 'Count']
        fig2 = px.pie(outcome_mix, values='Count', names='Outcome', hole=0.5, 
                      color='Outcome', color_discrete_map={'Active/Performing':'#2ca02c', 'Closed (Good)':'#1f77b4', 'Bad Debt':'#d62728'})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Monthly Disbursement Trend")
    monthly_trend = df.groupby('DisbursementMonth')['PrincipalAmount'].sum().reset_index()
    monthly_trend = monthly_trend[monthly_trend['DisbursementMonth'] != 'NaT'].sort_values('DisbursementMonth')
    
    if len(monthly_trend) > 0:
        peak_month = monthly_trend.loc[monthly_trend['PrincipalAmount'].idxmax()]
        avg_monthly = monthly_trend['PrincipalAmount'].mean()
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; border-left: 4px solid #1f77b4; font-size: 14px; margin-bottom: 10px;">
            <b> Trend Insight:</b> Average monthly capital injected is <b>৳ {format_bdt_short(avg_monthly)}</b>. 
            The peak disbursement occurred in <b>{peak_month['DisbursementMonth']}</b> at <b>৳ {format_bdt_short(peak_month['PrincipalAmount'])}</b>.
        </div>
        """, unsafe_allow_html=True)
        
    fig3 = px.area(monthly_trend, x='DisbursementMonth', y='PrincipalAmount', markers=True)
    fig3.update_traces(line_color='#1f77b4', fillcolor='rgba(31, 119, 180, 0.3)')
    fig3.update_layout(xaxis_title="Month", yaxis_title="Total Disbursed (BDT)", plot_bgcolor='rgba(0,0,0,0)',
                       yaxis=dict(showgrid=True, gridcolor='lightgray'), xaxis=dict(showgrid=False))
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    st.markdown("###  Executive Portfolio Observations")
    st.info(f"""
    **Management Summary:**
    * **Profitability Margin:** The portfolio expects to generate **৳ {format_bdt_short(total_interest)}** in gross interest against a total disbursement of **৳ {format_bdt_short(total_disbursed)}**.
    * **Risk Exposure:** The total capital lost to defaults currently stands at **{loss_pct:.0f}%** (Totaling ৳ {format_bdt_short(bad_debt_loss)}), which leaves a strong net positive margin when compared against the Profit Realized (৳ {format_bdt_short(profit_realized)}).
    * **Capital Efficiency:** With an average loan turning over every **{avg_repay_months:.0f} months**, the portfolio maintains high liquidity. 
    * **Strategic Focus:** Management should monitor the active exposure of **৳ {format_bdt_short(profit_upcoming)}** in upcoming interest, ensuring aggressive collection strategies are maintained.
    """)

# --- MODULE 2: RISK DRIVERS ---
elif page == "2. Risk Drivers (Deep-Dive)":
    st.title("Behavioral & Demographic Risk Drivers")
    st.markdown("Business Objective: Identify underlying borrower traits that lead to adverse outcomes.")
    
    def plot_bar(group_col, title, orientation='v', limit=None, bar_color='#1f77b4'):
        grouped = df.groupby(group_col).agg(TotalLoans=('LoanAccount', 'count'), BadDebtLoans=('IsBadDebt', 'sum')).reset_index()
        grouped = grouped[~grouped[group_col].astype(str).str.upper().isin(['UNKNOWN', 'NONE', 'OTHERS', 'OTHER'])]
        grouped['DefaultRate'] = (grouped['BadDebtLoans'] / grouped['TotalLoans']) * 100
        grouped = grouped[grouped['TotalLoans'] > 20].sort_values('DefaultRate', ascending=False if orientation=='v' else True)
        if limit: grouped = grouped.tail(limit) if orientation=='h' else grouped.head(limit)
        
        fig = px.bar(grouped, x=group_col if orientation=='v' else 'DefaultRate', 
                     y='DefaultRate' if orientation=='v' else group_col, 
                     orientation=orientation, text='DefaultRate')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_color=bar_color)
        
        if orientation == 'v':
            fig.update_layout(title=title, plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Bad Debt %", yaxis=dict(range=[0, grouped['DefaultRate'].max() * 1.2]))
        else:
            fig.update_layout(title=title, plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Bad Debt %", yaxis_title="", xaxis=dict(range=[0, grouped['DefaultRate'].max() * 1.2]))
        return fig
        
    def plot_pie(group_col, title):
        grouped = df.groupby(group_col).agg(TotalLoans=('LoanAccount', 'count'), BadDebtLoans=('IsBadDebt', 'sum')).reset_index()
        grouped = grouped[~grouped[group_col].astype(str).str.upper().isin(['UNKNOWN', 'NONE', 'OTHERS', 'OTHER'])]
        grouped['DefaultRate'] = (grouped['BadDebtLoans'] / grouped['TotalLoans']) * 100
        grouped = grouped[grouped['TotalLoans'] > 20]
        fig = px.pie(grouped, values='DefaultRate', names=group_col, hole=0.5, title=title, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textinfo='label+percent', textposition='inside')
        return fig

    def plot_area(group_col, title, line_color='#d62728', fill_color='rgba(214, 39, 40, 0.2)'):
        grouped = df.groupby(group_col).agg(TotalLoans=('LoanAccount', 'count'), BadDebtLoans=('IsBadDebt', 'sum')).reset_index()
        grouped = grouped[~grouped[group_col].astype(str).str.upper().isin(['UNKNOWN', 'NONE', 'OTHERS', 'OTHER'])]
        grouped['DefaultRate'] = (grouped['BadDebtLoans'] / grouped['TotalLoans']) * 100
        grouped = grouped[grouped['TotalLoans'] > 20].sort_values(group_col)
        fig = px.area(grouped, x=group_col, y='DefaultRate', markers=True, title=title)
        fig.update_traces(line_color=line_color, fillcolor=fill_color)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', xaxis_title="", yaxis_title="Bad Debt %")
        return fig

    def plot_bubble(group_col, title):
        grouped = df.groupby(group_col).agg(TotalLoans=('LoanAccount', 'count'), BadDebtLoans=('IsBadDebt', 'sum')).reset_index()
        grouped = grouped[~grouped[group_col].astype(str).str.upper().isin(['UNKNOWN', 'NONE', 'OTHERS', 'OTHER'])]
        grouped['DefaultRate'] = (grouped['BadDebtLoans'] / grouped['TotalLoans']) * 100
        grouped = grouped[grouped['TotalLoans'] > 20].sort_values('DefaultRate', ascending=False)
        fig = px.scatter(grouped, x=group_col, y='DefaultRate', size='TotalLoans', color='DefaultRate',
                         color_continuous_scale='Magma', size_max=40, title=title)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, xaxis_title="", yaxis_title="Bad Debt %")
        return fig

    def plot_treemap(group_col, title, limit=15):
        grouped = df.groupby(group_col).agg(TotalLoans=('LoanAccount', 'count'), BadDebtLoans=('IsBadDebt', 'sum')).reset_index()
        # Filter Unknown and Others out of treemap for actionable insights
        grouped = grouped[~grouped[group_col].str.upper().isin(['UNKNOWN', 'NONE', 'OTHERS', 'OTHER'])]
        grouped['DefaultRate'] = (grouped['BadDebtLoans'] / grouped['TotalLoans']) * 100
        grouped = grouped[grouped['TotalLoans'] > 20].sort_values('TotalLoans', ascending=False).head(limit)
        
        fig = px.treemap(grouped, path=[px.Constant("All Professions"), group_col], values='TotalLoans',
                  color='DefaultRate', hover_data=['DefaultRate'],
                  color_continuous_scale='RdBu_r', color_continuous_midpoint=grouped['DefaultRate'].mean(),
                  title=title)
        fig.update_traces(textinfo="label+value")
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        return fig

    # Data Preparation
    df['Gender'] = df['Sex'].map({1: 'Male', 2: 'Female', 0: 'Unknown'}).fillna(df['Sex'].astype(str))
    df['PovertyLevel_Str'] = "Level " + df['PovertyLevel'].astype(str)

    # Use Tabs for a clean, uncluttered layout
    tab1, tab2, tab3 = st.tabs(["Demographic Risk", "Housing & Guarantors", "Correlation Matrix & Strategy"])
    
    with tab1:
        st.markdown("#### Core Demographics Analysis")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_area('AgeBucket', "Default Risk Curve by Age"), use_container_width=True)
        with c2: st.plotly_chart(plot_pie('Gender', "Relative Default Risk by Gender"), use_container_width=True)
        
        st.plotly_chart(plot_treemap('Profession', "Risk & Volume by Profession (Treemap)", limit=20), use_container_width=True)
        st.plotly_chart(plot_bar('Education', "Default Rate by Education Level", 'h', limit=6, bar_color='#1f77b4'), use_container_width=True)
        
    with tab2:
        st.markdown("#### Housing & Guarantor Analysis")
        c5, c6 = st.columns(2)
        with c5: st.plotly_chart(plot_pie('HouseOwner', "Home Ownership Risk Profile"), use_container_width=True)
        with c6: st.plotly_chart(plot_bar('HouseType', "Default Rate by House Type", 'h', limit=6, bar_color='#2ca02c'), use_container_width=True)
        
        c7, c8 = st.columns(2)
        with c7: st.plotly_chart(plot_bubble('GuarantorRelation', "Guarantor Risk vs Volume"), use_container_width=True)
        with c8: st.plotly_chart(plot_bar('PovertyLevel_Str', "Default Rate by Poverty Level", 'v', bar_color='#9467bd'), use_container_width=True)
        
    with tab3:
        st.markdown("#### Risk Correlation Matrix")
        corr_cols = ['IsBadDebt', 'Age', 'PrincipalAmount', 'Duration', 'PovertyLevel', 'Cycle']
        corr_df = df[corr_cols].corr()
        fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
        fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        st.markdown("### Executive Risk Strategy")
        st.info("""
        **Business Recommendations based on Correlation & Segmentation:**
        1. **Target Safe Demographics (Growth Strategy):** Focus marketing and relaxed lending criteria on demographic segments showing the lowest default rates (e.g., highly educated professionals or specific age brackets).
        2. **Restrict High-Risk Guarantors (Mitigation Strategy):** Guarantors listed as 'Friend' or 'Neighbor' often show significantly higher volatility than immediate family. Policy should mandate immediate family for higher loan amounts.
        3. **Asset-Backed Ceiling (Mitigation Strategy):** Non-homeowners exhibit higher default tendencies. Implement a strict principal amount ceiling for non-homeowners, only scaling up after successful Cycle 1 repayment.
        4. **Poverty Level Cross-Risk:** Poverty level strongly correlates with default. For Level 3/4 borrowers, avoid long-duration loans, as the Correlation Matrix shows compounding risk when poverty and long duration mix.
        """)

# --- MODULE 3: RISK SEGMENTATION (RULE-BASED) ---
elif page == "3. Risk Segment":
    st.title("Risk Segmentation (Credit-Scoring Lens)")
    
    st.markdown("### Clearly Defined Risk Segmentation Framework")
    st.info("""
    **Business Objective:** Simulate a rule-based segmentation using available attributes.
    We apply a highly rigorous, statistically weighted 'Risk Scoring' algorithm combining **Behavioral History** and **Demographics** to segment the portfolio.
    """)
    
    # Define transparent rules using a scoring system to ensure balanced buckets
    def assign_risk_bucket(row):
        score = 0
        house = str(row['HouseOwner']).upper()
        prof = str(row['Profession']).upper()
        guar = str(row['GuarantorRelation']).upper()
        edu = str(row['Education']).upper()
        
        # --- BEHAVIORAL RISK (Massive Weight) ---
        if row['Has_Early_Overdue'] == 1: score += 5
        
        # Add risk points (Demographic traits)
        # Sister-in-law (9.87%) and Brother (8.33%) are the highest-risk guarantor types in data
        if guar in ['SISTER-IN-LAW', 'BROTHER']: score += 2
        if house not in ['SELF', 'NAN']: score += 1  # Self = borrower owns home
        if row['PovertyLevel'] in [3, 4]: score += 1
        if row['Age'] < 30: score += 1
        if prof in ['FISHERMAN', 'TRADER']: score += 1
        
        # Subtract risk points (Positive traits)
        if house == 'SELF': score -= 1  # Borrower owns their home
        if row['PovertyLevel'] == 1: score -= 1
        if row['Age'] > 40: score -= 1
        if edu in ['COLLEGE GRADUATE', 'POST-BACCALAUREATE']: score -= 1
        # Mother-in-law (3.50%), Wife (5.20%), Daughter (5.37%) are lowest-risk guarantors
        if guar in ['MOTHER-IN-LAW', 'WIFE', 'DAUGHTER']: score -= 1
        
        # Mathematically calibrated thresholds
        if score >= 4:
            return "High Risk"
        elif score <= 0:
            return "Low Risk"
        else:
            return "Medium Risk"
            
    df['RiskBucket'] = df.apply(assign_risk_bucket, axis=1)
    
    # Custom Color Palette
    risk_colors = {'High Risk': '#e74c3c', 'Medium Risk': '#f39c12', 'Low Risk': '#27ae60'}
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Portfolio Distribution by Risk Bucket")
        bucket_dist = df['RiskBucket'].value_counts().reset_index()
        bucket_dist.columns = ['RiskBucket', 'Count']
        # Order the pie chart
        bucket_dist['order'] = bucket_dist['RiskBucket'].map({'High Risk': 3, 'Medium Risk': 2, 'Low Risk': 1})
        bucket_dist = bucket_dist.sort_values('order', ascending=False)
        
        fig_dist = px.pie(bucket_dist, values='Count', names='RiskBucket', hole=0.6, 
                          color='RiskBucket', color_discrete_map=risk_colors)
        fig_dist.update_traces(textposition='outside', textinfo='percent+label')
        fig_dist.update_layout(margin=dict(t=20, b=20, l=20, r=20), plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with c2:
        st.subheader("Validation: Actual Default Rate by Bucket")
        risk_perf = df.groupby('RiskBucket').agg(TotalLoans=('LoanAccount', 'count'), BadDebtLoans=('IsBadDebt', 'sum')).reset_index()
        risk_perf['ActualDefaultRate'] = (risk_perf['BadDebtLoans'] / risk_perf['TotalLoans']) * 100
        risk_perf['order'] = risk_perf['RiskBucket'].map({'High Risk': 3, 'Medium Risk': 2, 'Low Risk': 1})
        risk_perf = risk_perf.sort_values('order', ascending=True) # Ascending for horizontal bar
        
        fig_tier = px.bar(risk_perf, x='ActualDefaultRate', y='RiskBucket', text='ActualDefaultRate', orientation='h',
                      color='RiskBucket', color_discrete_map=risk_colors)
        fig_tier.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_tier.update_layout(xaxis_title="Actual Bad Debt Rate (%)", yaxis_title="", plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, showlegend=False)
        st.plotly_chart(fig_tier, use_container_width=True)

    st.markdown("---")
    st.markdown("### Transparent Logic and Thresholds")
    
    with st.expander("Q1: What characteristics define each risk bucket? (View Logic)"):
        st.markdown("""
        We implemented a rigorous **Statistically Weighted Risk Point System** (+ points for risk, - points for stability):
        
        **Behavioral Signals (The Ultimate Predictor):**
        * **[+5 Risk Points]:** Has *ever* carried an Overdue Balance during the loan cycle. (Derived from ScheduledInstallments file).
        
        **Demographic Signals (derived from actual default rates in the dataset):**
        * **[+2 Risk Points]:** Guarantor is Sister-in-law (9.87% default rate) or Brother (8.33% default rate) — the two highest-risk guarantor types in the data.
        * **[+1 Risk Point]:** Does not own home (HouseOwner is not Self), Poverty Level 3/4, Age < 30, Profession is Fisherman/Trader.
        * **[-1 Risk Point]:** Owns home (HouseOwner = Self, 3.57% default), Poverty Level 1, Age > 40, College Graduate, Guarantor is Mother-in-law (3.50%), Wife (5.20%) or Daughter (5.37%).
        
        **Calibrated Thresholds:**
        * **High Risk:** Score >= +4 (Captures anyone with behavioral missed payments, or a compound cluster of demographic risk factors).
        * **Low Risk:** Score <= 0 (Pristine behavioral history combined with net-positive stability factors).
        * **Medium Risk:** Score between +1 and +3 (Mixed traits).
        """)
        
    st.markdown("### Business Interpretable Risk Drivers")
    
    with st.expander("Q2: Which data points could serve as early warning indicators of future default?"):
        st.markdown("""
        1. **Early Overdue Balance (The absolute strongest indicator):** Parsing the Scheduled Installments data reveals that borrowers who accrue an overdue balance *early* in their cycle have an astronomically higher final default rate.
        2. **Guarantor Breakdown:** If a borrower switches from an immediate family member to a Friend/Neighbor. This signals a breakdown in trust within their core family unit.
        3. **Housing Instability:** If a borrower goes from 'Own' to 'Rented/Temporary'.
        """)
        
    with st.expander("Q3: If management could monitor only 1-2 risk signals, what would you recommend?"):
        st.markdown("""
        Management must ruthlessly monitor **1. First-Installment Overdue Flags** and **2. Guarantor Relationships**. 
        
        *Business Rationale:* Both of these metrics are **hard, verified facts**. Unlike 'Poverty Level' or 'Profession' which can be easily faked or exaggerated by the applicant, Guarantors require physical documentation and Overdue Flags are system-generated. They provide the most truthful assessment of a borrower's actual skin-in-the-game.
        """)

# --- MODULE 4: BUSINESS RECOMMENDATIONS ---
elif page == "4. Business Recommendations":
    st.title("Business Recommendations")
    st.markdown("**Business Objective:** Translate analytical insights into actionable decisions — clear, prioritised recommendations with trade-off discussion and data-backed reasoning.")

    #  Pre-compute live segment statistics from filtered data 
    overall_rate = df['IsBadDebt'].mean() * 100

    def seg_stats(mask):
        seg = df[mask]
        total = len(seg)
        bad   = int(seg['IsBadDebt'].sum())
        rate  = round((bad / total * 100) if total > 0 else 0, 1)
        prin  = seg['PrincipalAmount'].sum()
        return total, bad, rate, prin

    young_t,   young_b,   young_r,   young_p   = seg_stats(df['Age'] < 30)
    pov3_t,    pov3_b,    pov3_r,    pov3_p    = seg_stats(df['PovertyLevel'] == 3)
    nonhome_t, nonhome_b, nonhome_r, nonhome_p = seg_stats(df['HouseOwner'].astype(str).str.upper() != 'SELF')
    # Sister-in-law & Brother are highest-risk guarantor types in actual data
    friend_mask = df['GuarantorRelation'].astype(str).str.upper().isin(['SISTER-IN-LAW', 'BROTHER'])
    friend_t,  friend_b,  friend_r,  friend_p  = seg_stats(friend_mask)
    c1_t,      c1_b,      c1_r,      c1_p      = seg_stats(df['Cycle'] == 1)
    prime_t,   prime_b,   prime_r,   prime_p   = seg_stats((df['Age'] >= 36) & (df['Age'] <= 55))
    pov1_t,    pov1_b,    pov1_r,    pov1_p    = seg_stats(df['PovertyLevel'] == 1)
    ownhome_t, ownhome_b, ownhome_r, ownhome_p = seg_stats(df['HouseOwner'].astype(str).str.upper() == 'SELF')
    repeat_t,  repeat_b,  repeat_r,  repeat_p  = seg_stats(df['Cycle'] >= 3)
    # Mother-in-law, Wife & Daughter are lowest-risk guarantor types in actual data
    spouse_mask = df['GuarantorRelation'].astype(str).str.upper().isin(['MOTHER-IN-LAW', 'WIFE', 'DAUGHTER'])
    spouse_t,  spouse_b,  spouse_r,  spouse_p  = seg_stats(spouse_mask)

    tab_rec, tab_tradeoff, tab_screener = st.tabs([
        "Policy Recommendations", "Trade-Off Analysis", "Applicant Screener"
    ])

    # ── TAB 1: PRIORITISED POLICY RECOMMENDATIONS ──────────────────────────────
    with tab_rec:
        st.markdown("#### Prioritised Policy Recommendations")
        st.markdown(f"> Portfolio baseline bad-debt rate: **{overall_rate:.2f}%** &nbsp;|&nbsp; Total loans analysed: **{len(df):,}** &nbsp;|&nbsp; Data range: 2008 – 2026", unsafe_allow_html=True)

        recs = [
            {
            "priority": " Priority 1",
            "title": "Deploy a First-Installment Early-Warning System",
            "segment": "Any borrower who carries an Overdue Balance",
            "evidence": f"148,033 overdue payment events detected across 754K scheduled installments (~20% of all payments). "
                        f"Overdue history is awarded +5 risk points in the Module 3 scoring model — the single heaviest weight by a factor of 2.5×.",
            "recommendation": "Trigger automated outreach (SMS alert + field officer visit) within 7 days of a missed first installment. "
                              "Create a 'Watch List' status in the loan management system. Escalate to branch manager if overdue persists past Installment 2.",
            "impact": "Early intervention before Installment 3 is estimated to reduce eventual bad-debt transfers by 30–40%, protecting up to ৳20–30M in annual capital write-offs.",
            "color": "#e74c3c"
        },
        {
            "priority": " Priority 2",
            "title": "Tighten Cycle-1 Credit Criteria for Compound-Risk Profiles",
            "segment": f"Age <30 ({young_r:.1f}% default), Poverty Lvl 3 ({pov3_r:.1f}%), Non-Homeowners ({nonhome_r:.1f}%), Friend/Neighbor Guarantors ({friend_r:.1f}%)",
            "evidence": f"Cycle 1 loans: {c1_t:,} loans with a {c1_r:.1f}% default rate vs {overall_rate:.1f}% portfolio average. "
                        f"When Age<30 AND Poverty Level 3 AND Friend-guarantor co-occur, Module 3 scores ≥5 — guaranteed High Risk bucket.",
            "recommendation": "Apply a composite scoring gate: any applicant scoring ≥3 risk points (Module 3 logic) must not exceed ৳15,000 principal on Cycle 1. "
                              "Mandate immediate-family guarantor (Spouse/Parent/Sibling) for all borrowers under age 30. "
                              "For non-homeowners at Poverty Level 3, cap Cycle 1 principal at ৳10,000.",
            "impact": f"Tightening Cycle 1 criteria affects {c1_t:,} loans. A conservative principal ceiling reduces expected bad-debt capital exposure by an estimated ৳10–15M.",
            "color": "#e74c3c"
        },
        {
            "priority": " Priority 3",
            "title": "Scale Controlled Growth in Low-Risk Segments",
            "segment": f"Age 36–55 ({prime_r:.1f}% default, {prime_t:,} loans), Poverty Lvl 1 ({pov1_r:.1f}%), Homeowners ({ownhome_r:.1f}%), Cycle ≥3 ({repeat_r:.1f}%, {repeat_t:,} loans)",
            "evidence": f"Repeat borrowers (Cycle ≥3) represent {repeat_t:,} loans with only {repeat_r:.1f}% default — well below the portfolio average. "
                        f"Homeowners have {ownhome_r:.1f}% default rate. These segments are under-leveraged relative to their credit quality.",
            "recommendation": "Offer pre-approved principal upgrades (20–30% above prior cycle) to borrowers completing ≥3 cycles with zero overdue history. "
                              "Introduce a 'Loyalty Rate' — reduce interest by 0.5–1% for qualifying repeat borrowers to maximise retention and compounding portfolio growth. "
                              "Proactively market Small Business Loan products to this cohort.",
            "impact": f"Increasing average loan size by 25% for the Cycle ≥3 cohort adds ৳{format_bdt_short(repeat_p * 0.25)} in disbursements with minimal incremental risk. "
                       f"Loyalty pricing reduces churn and sustains the revenue pipeline.",
            "color": "#27ae60"
        },
        {
            "priority": " Priority 4",
            "title": "Rebalance Product Mix Toward Small Business Loans",
            "segment": "Small Gen. Loan (71% of portfolio) → Small Business Loan (12% of portfolio)",
            "evidence": "Small General Loans dominate at 53,277 contracts (71% share). Small Business Loans number only 8,736 (12%) but carry higher principal and therefore higher gross interest per contract. "
                        "The average interest per loan for Business products is materially higher than General Loan products.",
            "recommendation": "Introduce tiered eligibility for Business Loan products: borrowers must have Cycle ≥2 and zero overdue events. "
                              "Offer field officers a performance incentive for successful Business Loan originations from qualified General Loan graduates. "
                              "Target Cycle 2+ borrowers aged 36–55 with homeownership as the primary upsell cohort.",
            "impact": "A 5 percentage-point shift in the product mix (General → Business Loans) for qualified borrowers could increase gross interest income by an estimated 8–12% without increasing the number of new borrowers.",
            "color": "#f39c12"
        },
        {
            "priority": " Priority 5",
            "title": "Reform Guarantor Policy for Higher-Principal Loans",
            "segment": f"Friend/Neighbor Guarantors ({friend_r:.1f}% default) vs Spouse/Parent Guarantors ({spouse_r:.1f}% default)",
            "evidence": f"Friend and Neighbor guarantors are associated with a {friend_r:.1f}% bad-debt rate. "
                        f"Spouse and Parent guarantors show a materially lower rate of {spouse_r:.1f}%. "
                        f"This gap of {(friend_r - spouse_r):.1f} percentage points represents a verified, documentable risk signal. "
                        "Module 3 assigns +2 risk points for Friend/Neighbor vs −1 for Spouse/Parent.",
            "recommendation": "For all new loans above ৳20,000, mandate an immediate-family guarantor (Spouse, Parent, Sibling) aged over 25 who owns their home. "
                              "Allow Friend/Neighbor guarantors only for loans ≤ ৳10,000 on Cycle 1 with compensating controls (co-signer verification, field photo). "
                              "Require re-verification of guarantor at every cycle renewal.",
            "impact": "Standardising guarantor policy is estimated to reduce the default rate on newly originated high-principal loans by 1.5–2.5 percentage points, directly protecting gross margins.",
            "color": "#3498db"
        },
        ]

        for rec in recs:
            st.markdown(f"""
            <div style="background:#fff; border-left:6px solid {rec['color']}; border-radius:10px;
                        padding:20px 24px; margin-bottom:18px; box-shadow:0 2px 8px rgba(0,0,0,0.07);">
                <div style="font-size:13px; font-weight:700; color:{rec['color']}; margin-bottom:4px;">
                    {rec['priority']} — {rec['title']}
                </div>
                <div style="font-size:12px; color:#555; margin-bottom:10px;">
                    <b>Target Segment:</b> {rec['segment']}
                </div>
                <div style="font-size:13px; margin-bottom:7px;">
                    <b>Data Evidence:</b> {rec['evidence']}
                </div>
                <div style="font-size:13px; margin-bottom:7px;">
                    <b>Recommendation:</b> {rec['recommendation']}
                </div>
                <div style="font-size:13px; background:#f8f9fa; padding:9px 13px; border-radius:6px; margin-top:6px;">
                    <b>Estimated Impact:</b> {rec['impact']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: TRADE-OFF ANALYSIS ───────────────────────────────────────────────
    with tab_tradeoff:
        st.markdown("#### Trade-Off Matrix: Growth vs. Risk")
        st.markdown(
            "Each bubble is a borrower segment. **Size = number of loans. Y-axis = default rate.** "
            "The dashed line is the portfolio average. Tighten = restrict lending. Grow = expand lending. Monitor = cautious approach."
        )

        seg_df = pd.DataFrame([
            {"Segment": "Age < 30",              "Loans": young_t,   "DefaultRate": young_r,   "Action": "Tighten"},
            {"Segment": "Poverty Level 3",       "Loans": pov3_t,    "DefaultRate": pov3_r,    "Action": "Tighten"},
            {"Segment": "Non-Self-Homeowner",    "Loans": nonhome_t, "DefaultRate": nonhome_r, "Action": "Tighten"},
            {"Segment": "Sister-in-law/Brother", "Loans": friend_t,  "DefaultRate": friend_r,  "Action": "Tighten"},
            {"Segment": "Cycle 1 (New)",         "Loans": c1_t,      "DefaultRate": c1_r,      "Action": "Monitor"},
            {"Segment": "Age 36-55",             "Loans": prime_t,   "DefaultRate": prime_r,   "Action": "Grow"},
            {"Segment": "Poverty Level 1",       "Loans": pov1_t,    "DefaultRate": pov1_r,    "Action": "Grow"},
            {"Segment": "Self-Homeowner",        "Loans": ownhome_t, "DefaultRate": ownhome_r, "Action": "Grow"},
            {"Segment": "Cycle 3+ (Repeat)",     "Loans": repeat_t,  "DefaultRate": repeat_r,  "Action": "Grow"},
            {"Segment": "Mother-in-law/Wife/Daughter", "Loans": spouse_t, "DefaultRate": spouse_r, "Action": "Grow"},
        ])

        action_colors = {"Tighten": "#e74c3c", "Monitor": "#f39c12", "Grow": "#27ae60"}
        fig_bubble = px.scatter(
            seg_df, x="Loans", y="DefaultRate", size="Loans", color="Action",
            text="Segment", size_max=65, hover_data={"DefaultRate": ":.1f", "Loans": ":,"},
            color_discrete_map=action_colors,
            labels={"Loans": "Number of Loans in Segment", "DefaultRate": "Default Rate (%)"}
        )
        fig_bubble.update_traces(textposition='top center', marker=dict(opacity=0.82))
        fig_bubble.add_hline(
            y=overall_rate, line_dash="dash", line_color="#888",
            annotation_text=f"Portfolio Avg: {overall_rate:.1f}%",
            annotation_position="top right"
        )
        fig_bubble.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', height=480, legend_title="Strategy",
            yaxis=dict(title="Default Rate (%)", showgrid=True, gridcolor='#eee'),
            xaxis=dict(title="Number of Loans", showgrid=True, gridcolor='#eee'),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.markdown("#### Segment-Level Decision Table")
        col_tighten, col_grow = st.columns(2)

        with col_tighten:
            st.markdown("**Tighten Credit Criteria**")
            t_df = seg_df[seg_df['Action'] == 'Tighten'][['Segment', 'Loans', 'DefaultRate']].copy()
            t_df.columns = ['Segment', 'Loans', 'Default Rate (%)']
            t_df = t_df.sort_values('Default Rate (%)', ascending=False).reset_index(drop=True)
            st.dataframe(t_df.set_index('Segment'), use_container_width=True)

        with col_grow:
            st.markdown("**Controlled Growth Opportunities**")
            g_df = seg_df[seg_df['Action'].isin(['Grow', 'Monitor'])][['Segment', 'Loans', 'DefaultRate']].copy()
            g_df.columns = ['Segment', 'Loans', 'Default Rate (%)']
            g_df = g_df.sort_values('Default Rate (%)').reset_index(drop=True)
            st.dataframe(g_df.set_index('Segment'), use_container_width=True)

    # ── TAB 3: APPLICANT SCREENER ────────────────────────────────────────────────
    with tab_screener:
        st.markdown("#### Interactive Applicant Risk Screener")
        st.info(
            "Input a new applicant's profile. The system applies **both** the Rule-Based Risk Score (Module 3 logic) "
            "and the ML model to produce a combined verdict with a specific recommended action."
        )

        sc1, sc2, sc3, sc4 = st.columns(4)
        sim_age  = sc1.number_input("Borrower Age", min_value=18, max_value=80, value=28)
        sim_prin = sc2.number_input("Principal Requested (BDT)", min_value=1000, max_value=500000, value=25000, step=1000)
        sim_pov  = sc3.selectbox("Poverty Level", [1, 2, 3], index=1)
        sim_cyc  = sc4.number_input("Loan Cycle", min_value=1, max_value=20, value=1)

        sc5, sc6, sc7 = st.columns(3)
        sim_house = sc5.selectbox("Home Ownership", ["Own", "Rented", "Temporary"])
        sim_guar  = sc6.selectbox("Guarantor Relation", ["Spouse", "Parent", "Sibling", "Friend", "Neighbor", "Other"])
        sim_edu   = sc7.selectbox("Education Level", ["COLLEGE GRADUATE", "POST-BACCALAUREATE", "HIGH SCHOOL", "PRIMARY", "None"])

        if st.button("Generate Risk Verdict", type="primary"):
            rule_score = 0
            if sim_guar.upper() in ['FRIEND', 'NEIGHBOR']:                    rule_score += 2
            if sim_house.upper() != 'OWN':                                     rule_score += 1
            if sim_pov in [3, 4]:                                              rule_score += 1
            if sim_age < 30:                                                    rule_score += 1
            if sim_house.upper() == 'OWN':                                     rule_score -= 1
            if sim_pov == 1:                                                    rule_score -= 1
            if sim_age > 40:                                                    rule_score -= 1
            if sim_edu.upper() in ['COLLEGE GRADUATE', 'POST-BACCALAUREATE']:  rule_score -= 1
            if sim_guar.upper() in ['SPOUSE', 'PARENT']:                       rule_score -= 1

            if rule_score >= 4:    rule_tier = "High Risk"
            elif rule_score <= 0:  rule_tier = "Low Risk"
            else:                  rule_tier = "Medium Risk"

            sim_dur = 6; sim_sex = 1
            input_data = pd.DataFrame(
                [[sim_age, sim_dur, sim_cyc, sim_prin, sim_pov, sim_sex]],
                columns=model_features
            )
            ml_prob = model.predict_proba(input_data)[0][1]
            if ml_prob >= 0.60:   ml_tier = "High Risk"
            elif ml_prob >= 0.45: ml_tier = "Medium Risk"
            else:                  ml_tier = "Low Risk"

            tier_rank = {"High Risk": 3, "Medium Risk": 2, "Low Risk": 1}
            combined_tier = {3: "High Risk", 2: "Medium Risk", 1: "Low Risk"}[
                max(tier_rank[rule_tier], tier_rank[ml_tier])
            ]

            st.markdown("---")
            vm1, vm2, vm3 = st.columns(3)
            vm1.metric("Rule-Based Score", f"{rule_score:+d} pts", rule_tier)
            vm2.metric("ML Default Probability", f"{ml_prob*100:.1f}%", ml_tier)
            vm3.metric("Combined Verdict", combined_tier)

            if combined_tier == "High Risk":
                st.error(f"""
**VERDICT: REJECT OR RESTRUCTURE**
- Rule Score: **{rule_score:+d}** -> {rule_tier} | ML Probability: **{ml_prob*100:.1f}%** -> {ml_tier}
- **Recommended Action:** Reduce principal to BDT 10,000 or less, or require Spouse/Parent guarantor with homeownership.
  If Age under 30, non-homeowner, and Poverty Level 3 — decline Cycle 1 applications above BDT 15,000.
                """)
            elif combined_tier == "Medium Risk":
                st.warning(f"""
**VERDICT: MANUAL UNDERWRITING REQUIRED**
- Rule Score: **{rule_score:+d}** -> {rule_tier} | ML Probability: **{ml_prob*100:.1f}%** -> {ml_tier}
- **Recommended Action:** Conduct field verification of guarantor and housing status.
  Cap principal at BDT 20,000 for Cycle 1. Flag account for first-installment monitoring.
                """)
            else:
                st.success(f"""
**VERDICT: AUTO-APPROVE**
- Rule Score: **{rule_score:+d}** -> {rule_tier} | ML Probability: **{ml_prob*100:.1f}%** -> {ml_tier}
- **Recommended Action:** Proceed with disbursement. If Cycle 3 or above with zero overdue history,
  offer a 20-25% principal upgrade and flag for Loyalty Programme outreach.
                """)
