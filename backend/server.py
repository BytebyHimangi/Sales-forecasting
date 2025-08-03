from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
import tempfile
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class SalesData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    date: datetime
    product_category: str
    region: str
    units_sold: int
    revenue: float
    upload_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SalesDataUpload(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    total_records: int
    status: str = "completed"
    
class ForecastRequest(BaseModel):
    upload_id: str
    forecast_months: int = 6
    
class ForecastResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    upload_id: str
    forecast_months: int
    model_accuracy: Dict[str, float]
    forecast_data: List[Dict[str, Any]]
    insights: List[str]
    feature_importance: Dict[str, float]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DataPreview(BaseModel):
    columns: List[str]
    sample_data: List[Dict[str, Any]]
    total_rows: int
    issues: List[str]

# Utility functions
def validate_sales_data(df: pd.DataFrame) -> List[str]:
    """Validate uploaded sales data format"""
    issues = []
    required_columns = ['date', 'product_category', 'region', 'units_sold', 'revenue']
    
    # Normalize column names for comparison
    normalized_cols = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Check for required columns
    missing_columns = []
    for required_col in required_columns:
        if required_col not in normalized_cols:
            missing_columns.append(required_col)
    
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for empty data
    if df.empty:
        issues.append("File is empty")
    
    # Check date format
    date_col = None
    for i, col in enumerate(df.columns):
        if col.lower().replace(' ', '_') == 'date':
            date_col = df.iloc[:, i]
            break
    
    if date_col is not None:
        try:
            pd.to_datetime(date_col)
        except:
            issues.append("Invalid date format. Please use YYYY-MM-DD format")
    
    # Check numeric columns
    numeric_cols = ['units_sold', 'revenue']
    for required_col in numeric_cols:
        for i, col in enumerate(df.columns):
            if col.lower().replace(' ', '_') == required_col:
                if not pd.api.types.is_numeric_dtype(df.iloc[:, i]):
                    issues.append(f"Column '{col}' should contain numeric values")
                break
    
    return issues

def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess sales data"""
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values
    df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce').fillna(0)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['date', 'product_category', 'region'])
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for ML model"""
    # Create time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Encode categorical variables
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    
    df['product_category_encoded'] = le_category.fit_transform(df['product_category'])
    df['region_encoded'] = le_region.fit_transform(df['region'])
    
    # Create lag features
    df['revenue_lag1'] = df['revenue'].shift(1)
    df['revenue_lag2'] = df['revenue'].shift(2)
    
    # Create moving averages
    df['revenue_ma3'] = df['revenue'].rolling(window=3).mean()
    df['revenue_ma6'] = df['revenue'].rolling(window=6).mean()
    
    return df

def build_forecast_model(df: pd.DataFrame) -> tuple:
    """Build and train the forecasting model"""
    # Prepare features
    df_features = prepare_features(df)
    
    # Remove rows with NaN values (due to lag features)
    df_features = df_features.dropna()
    
    # Define features and target
    feature_columns = ['year', 'month', 'quarter', 'day_of_year', 
                      'product_category_encoded', 'region_encoded', 'units_sold',
                      'revenue_lag1', 'revenue_lag2', 'revenue_ma3', 'revenue_ma6']
    
    X = df_features[feature_columns]
    y = df_features['revenue']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance (coefficients)
    feature_importance = dict(zip(feature_columns, abs(model.coef_)))
    
    return model, scaler, feature_importance, {'mape': mape, 'rmse': rmse}

def generate_forecast(model, scaler, df: pd.DataFrame, months: int = 6) -> List[Dict[str, Any]]:
    """Generate future sales forecast"""
    # Get the last date and extend forecast
    last_date = df['date'].max()
    
    # Prepare base data for forecast
    df_features = prepare_features(df)
    df_features = df_features.dropna()
    
    forecast_data = []
    
    for i in range(1, months + 1):
        future_date = last_date + timedelta(days=30 * i)  # Approximate monthly forecast
        
        # Create average features for forecast
        avg_features = {
            'year': future_date.year,
            'month': future_date.month,
            'quarter': (future_date.month - 1) // 3 + 1,
            'day_of_year': future_date.timetuple().tm_yday,
            'product_category_encoded': df_features['product_category_encoded'].mean(),
            'region_encoded': df_features['region_encoded'].mean(),
            'units_sold': df_features['units_sold'].mean(),
            'revenue_lag1': df_features['revenue'].tail(1).iloc[0] if len(df_features) > 0 else 0,
            'revenue_lag2': df_features['revenue'].tail(2).iloc[0] if len(df_features) > 1 else 0,
            'revenue_ma3': df_features['revenue'].tail(3).mean() if len(df_features) > 2 else 0,
            'revenue_ma6': df_features['revenue'].tail(6).mean() if len(df_features) > 5 else 0,
        }
        
        # Prepare features for prediction
        feature_columns = ['year', 'month', 'quarter', 'day_of_year', 
                          'product_category_encoded', 'region_encoded', 'units_sold',
                          'revenue_lag1', 'revenue_lag2', 'revenue_ma3', 'revenue_ma6']
        
        X_forecast = np.array([avg_features[col] for col in feature_columns]).reshape(1, -1)
        X_forecast_scaled = scaler.transform(X_forecast)
        
        # Make prediction
        predicted_revenue = model.predict(X_forecast_scaled)[0]
        
        forecast_data.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'predicted_revenue': round(predicted_revenue, 2),
            'month': future_date.strftime('%B %Y')
        })
    
    return forecast_data

def generate_insights(df: pd.DataFrame, forecast_data: List[Dict[str, Any]]) -> List[str]:
    """Generate business insights from data and forecast"""
    insights = []
    
    # Historical trends
    monthly_revenue = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
    recent_trend = monthly_revenue.pct_change().tail(3).mean()
    
    if recent_trend > 0.05:
        insights.append("📈 Sales are showing positive growth trend over the last 3 months")
    elif recent_trend < -0.05:
        insights.append("📉 Sales are declining over the last 3 months - immediate attention needed")
    else:
        insights.append("📊 Sales are relatively stable with minor fluctuations")
    
    # Category performance
    category_performance = df.groupby('product_category')['revenue'].sum().sort_values(ascending=False)
    top_category = category_performance.index[0]
    bottom_category = category_performance.index[-1]
    
    insights.append(f"🏆 Top performing category: {top_category} (${category_performance.iloc[0]:,.0f})")
    insights.append(f"⚠️ Lowest performing category: {bottom_category} (${category_performance.iloc[-1]:,.0f})")
    
    # Regional analysis
    regional_performance = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
    if len(regional_performance) > 1:
        top_region = regional_performance.index[0]
        insights.append(f"🌍 Strongest region: {top_region} (${regional_performance.iloc[0]:,.0f})")
    
    # Forecast insights
    current_monthly_avg = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum().tail(3).mean()
    forecast_avg = np.mean([item['predicted_revenue'] for item in forecast_data])
    
    if forecast_avg > current_monthly_avg * 1.05:
        insights.append("🚀 Forecast shows expected growth - maintain current strategies")
    elif forecast_avg < current_monthly_avg * 0.95:
        insights.append("⚠️ Forecast shows potential decline - consider intervention strategies")
    
    return insights

# API Routes
@api_router.post("/upload-sales-data")
async def upload_sales_data(file: UploadFile = File(...)):
    """Upload sales data file (CSV/Excel)"""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or Excel files.")
        
        # Validate data
        issues = validate_sales_data(df)
        if issues:
            return {"status": "error", "issues": issues}
        
        # Clean data
        df_clean = clean_sales_data(df)
        
        # Create upload record
        upload_record = SalesDataUpload(
            filename=file.filename,
            total_records=len(df_clean)
        )
        
        # Store upload record
        await db.sales_uploads.insert_one(upload_record.dict())
        
        # Store individual sales records
        sales_records = []
        for _, row in df_clean.iterrows():
            sales_record = SalesData(
                date=row['date'],
                product_category=row['product_category'],
                region=row['region'],
                units_sold=row['units_sold'],
                revenue=row['revenue'],
                upload_id=upload_record.id
            )
            sales_records.append(sales_record.dict())
        
        # Bulk insert sales data
        if sales_records:
            await db.sales_data.insert_many(sales_records)
        
        return {
            "status": "success",
            "upload_id": upload_record.id,
            "total_records": len(df_clean),
            "message": "Data uploaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_router.post("/preview-sales-data")
async def preview_sales_data(file: UploadFile = File(...)):
    """Preview uploaded sales data"""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate data
        issues = validate_sales_data(df)
        
        # Get preview data (first 10 rows)
        preview_data = df.head(10).fillna("").to_dict('records')
        
        return DataPreview(
            columns=list(df.columns),
            sample_data=preview_data,
            total_rows=len(df),
            issues=issues
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error previewing file: {str(e)}")

@api_router.post("/generate-forecast")
async def generate_sales_forecast(request: ForecastRequest):
    """Generate sales forecast"""
    try:
        # Retrieve sales data
        sales_data = await db.sales_data.find({"upload_id": request.upload_id}).to_list(10000)
        
        if not sales_data:
            raise HTTPException(status_code=404, detail="No sales data found for this upload")
        
        # Convert to DataFrame
        df = pd.DataFrame(sales_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Build model and generate forecast
        model, scaler, feature_importance, accuracy = build_forecast_model(df)
        forecast_data = generate_forecast(model, scaler, df, request.forecast_months)
        insights = generate_insights(df, forecast_data)
        
        # Create forecast result
        forecast_result = ForecastResult(
            upload_id=request.upload_id,
            forecast_months=request.forecast_months,
            model_accuracy=accuracy,
            forecast_data=forecast_data,
            insights=insights,
            feature_importance=feature_importance
        )
        
        # Store forecast result
        await db.forecast_results.insert_one(forecast_result.dict())
        
        return forecast_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

@api_router.get("/uploads", response_model=List[SalesDataUpload])
async def get_uploads():
    """Get all data uploads"""
    uploads = await db.sales_uploads.find().sort("upload_date", -1).to_list(100)
    return [SalesDataUpload(**upload) for upload in uploads]

@api_router.get("/forecast/{upload_id}")
async def get_forecast(upload_id: str):
    """Get forecast results for an upload"""
    forecast = await db.forecast_results.find_one({"upload_id": upload_id})
    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")
    return ForecastResult(**forecast)

@api_router.get("/export-forecast-csv/{upload_id}")
async def export_forecast_csv(upload_id: str):
    """Export forecast data as CSV"""
    try:
        forecast = await db.forecast_results.find_one({"upload_id": upload_id})
        if not forecast:
            raise HTTPException(status_code=404, detail="Forecast not found")
        
        # Create CSV content
        df = pd.DataFrame(forecast['forecast_data'])
        csv_content = df.to_csv(index=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(csv_content)
            tmp_file_path = tmp_file.name
        
        return FileResponse(
            tmp_file_path,
            media_type='text/csv',
            filename=f"forecast_{upload_id}.csv"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()