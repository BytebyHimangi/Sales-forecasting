#!/usr/bin/env python3

import requests
import sys
import os
import json
from datetime import datetime
import time

class SalesForecastAPITester:
    def __init__(self, base_url="https://90115413-6d90-4f5f-a585-a9724fca0d0a.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.upload_id = None
        self.sample_file_path = "/app/sample_sales_data.csv"

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def test_preview_sales_data(self):
        """Test the preview sales data endpoint"""
        print(f"\n🔍 Testing Preview Sales Data API...")
        
        try:
            # Check if sample file exists
            if not os.path.exists(self.sample_file_path):
                return self.log_test("Preview Sales Data", False, "- Sample file not found")
            
            with open(self.sample_file_path, 'rb') as file:
                files = {'file': ('sample_sales_data.csv', file, 'text/csv')}
                response = requests.post(f"{self.api_url}/preview-sales-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['columns', 'sample_data', 'total_rows', 'issues']
                
                if all(field in data for field in required_fields):
                    details = f"- Status: {response.status_code}, Rows: {data['total_rows']}, Columns: {len(data['columns'])}"
                    if data['issues']:
                        details += f", Issues: {len(data['issues'])}"
                    return self.log_test("Preview Sales Data", True, details)
                else:
                    return self.log_test("Preview Sales Data", False, f"- Missing required fields in response")
            else:
                return self.log_test("Preview Sales Data", False, f"- Status: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            return self.log_test("Preview Sales Data", False, f"- Error: {str(e)}")

    def test_upload_sales_data(self):
        """Test the upload sales data endpoint"""
        print(f"\n📤 Testing Upload Sales Data API...")
        
        try:
            if not os.path.exists(self.sample_file_path):
                return self.log_test("Upload Sales Data", False, "- Sample file not found")
            
            with open(self.sample_file_path, 'rb') as file:
                files = {'file': ('sample_sales_data.csv', file, 'text/csv')}
                response = requests.post(f"{self.api_url}/upload-sales-data", files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'upload_id' in data:
                    self.upload_id = data['upload_id']
                    details = f"- Status: {response.status_code}, Upload ID: {self.upload_id}, Records: {data.get('total_records')}"
                    return self.log_test("Upload Sales Data", True, details)
                else:
                    return self.log_test("Upload Sales Data", False, f"- Invalid response format: {data}")
            else:
                return self.log_test("Upload Sales Data", False, f"- Status: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            return self.log_test("Upload Sales Data", False, f"- Error: {str(e)}")

    def test_get_uploads(self):
        """Test the get uploads endpoint"""
        print(f"\n📋 Testing Get Uploads API...")
        
        try:
            response = requests.get(f"{self.api_url}/uploads")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    details = f"- Status: {response.status_code}, Found {len(data)} uploads"
                    return self.log_test("Get Uploads", True, details)
                else:
                    return self.log_test("Get Uploads", False, f"- Expected list, got: {type(data)}")
            else:
                return self.log_test("Get Uploads", False, f"- Status: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            return self.log_test("Get Uploads", False, f"- Error: {str(e)}")

    def test_generate_forecast(self, months=6):
        """Test the generate forecast endpoint"""
        print(f"\n🔮 Testing Generate Forecast API...")
        
        if not self.upload_id:
            return self.log_test("Generate Forecast", False, "- No upload_id available")
        
        try:
            payload = {
                "upload_id": self.upload_id,
                "forecast_months": months
            }
            
            response = requests.post(f"{self.api_url}/generate-forecast", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['forecast_data', 'model_accuracy', 'insights', 'feature_importance']
                
                if all(field in data for field in required_fields):
                    forecast_count = len(data['forecast_data'])
                    mape = data['model_accuracy'].get('mape', 0)
                    rmse = data['model_accuracy'].get('rmse', 0)
                    insights_count = len(data['insights'])
                    
                    details = f"- Status: {response.status_code}, Forecast points: {forecast_count}, MAPE: {mape:.4f}, RMSE: {rmse:.2f}, Insights: {insights_count}"
                    return self.log_test("Generate Forecast", True, details)
                else:
                    missing = [f for f in required_fields if f not in data]
                    return self.log_test("Generate Forecast", False, f"- Missing fields: {missing}")
            else:
                return self.log_test("Generate Forecast", False, f"- Status: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            return self.log_test("Generate Forecast", False, f"- Error: {str(e)}")

    def test_get_forecast(self):
        """Test the get forecast endpoint"""
        print(f"\n📊 Testing Get Forecast API...")
        
        if not self.upload_id:
            return self.log_test("Get Forecast", False, "- No upload_id available")
        
        try:
            response = requests.get(f"{self.api_url}/forecast/{self.upload_id}")
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['forecast_data', 'model_accuracy', 'insights', 'feature_importance']
                
                if all(field in data for field in required_fields):
                    details = f"- Status: {response.status_code}, Forecast retrieved successfully"
                    return self.log_test("Get Forecast", True, details)
                else:
                    return self.log_test("Get Forecast", False, f"- Invalid response format")
            else:
                return self.log_test("Get Forecast", False, f"- Status: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            return self.log_test("Get Forecast", False, f"- Error: {str(e)}")

    def test_export_forecast_csv(self):
        """Test the export forecast CSV endpoint"""
        print(f"\n💾 Testing Export Forecast CSV API...")
        
        if not self.upload_id:
            return self.log_test("Export Forecast CSV", False, "- No upload_id available")
        
        try:
            response = requests.get(f"{self.api_url}/export-forecast-csv/{self.upload_id}")
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                
                if 'csv' in content_type.lower() or content_length > 0:
                    details = f"- Status: {response.status_code}, Content-Type: {content_type}, Size: {content_length} bytes"
                    return self.log_test("Export Forecast CSV", True, details)
                else:
                    return self.log_test("Export Forecast CSV", False, f"- Invalid content type or empty response")
            else:
                return self.log_test("Export Forecast CSV", False, f"- Status: {response.status_code}, Response: {response.text}")
                
        except Exception as e:
            return self.log_test("Export Forecast CSV", False, f"- Error: {str(e)}")

    def test_invalid_file_upload(self):
        """Test upload with invalid file format"""
        print(f"\n🚫 Testing Invalid File Upload...")
        
        try:
            # Create a fake text file
            fake_content = "This is not a CSV file"
            files = {'file': ('fake.txt', fake_content, 'text/plain')}
            response = requests.post(f"{self.api_url}/upload-sales-data", files=files)
            
            # Should return 400 or error status
            if response.status_code == 400 or (response.status_code == 200 and 'error' in response.json().get('status', '')):
                details = f"- Status: {response.status_code}, Correctly rejected invalid file"
                return self.log_test("Invalid File Upload", True, details)
            else:
                return self.log_test("Invalid File Upload", False, f"- Should have rejected invalid file, got: {response.status_code}")
                
        except Exception as e:
            return self.log_test("Invalid File Upload", False, f"- Error: {str(e)}")

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Sales Forecasting API Tests...")
        print(f"📍 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test sequence
        self.test_preview_sales_data()
        self.test_upload_sales_data()
        self.test_get_uploads()
        
        # Only run forecast tests if upload was successful
        if self.upload_id:
            self.test_generate_forecast(6)  # Test 6-month forecast
            time.sleep(1)  # Brief pause between tests
            self.test_get_forecast()
            self.test_export_forecast_csv()
        
        # Test error handling
        self.test_invalid_file_upload()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {self.tests_run}")
        print(f"   Passed: {self.tests_passed}")
        print(f"   Failed: {self.tests_run - self.tests_passed}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed!")
            return 1

def main():
    """Main test runner"""
    tester = SalesForecastAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())