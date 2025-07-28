import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";
import { Line, Bar, Doughnut } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from "chart.js";
import { saveAs } from "file-saver";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [activeTab, setActiveTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [uploadId, setUploadId] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploads, setUploads] = useState([]);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [forecastMonths, setForecastMonths] = useState(6);

  useEffect(() => {
    fetchUploads();
  }, []);

  const fetchUploads = async () => {
    try {
      const response = await axios.get(`${API}/uploads`);
      setUploads(response.data);
    } catch (error) {
      console.error("Error fetching uploads:", error);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    if (selectedFile) {
      previewFile(selectedFile);
    }
  };

  const previewFile = async (file) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(`${API}/preview-sales-data`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPreview(response.data);
    } catch (error) {
      console.error("Error previewing file:", error);
      alert("Error previewing file: " + error.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(`${API}/upload-sales-data`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data.status === "success") {
        setUploadId(response.data.upload_id);
        fetchUploads();
        alert("Data uploaded successfully!");
        setActiveTab("forecast");
      } else {
        alert("Upload failed: " + response.data.issues.join(", "));
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("Error uploading file: " + error.response?.data?.detail);
    } finally {
      setLoading(false);
    }
  };

  const generateForecast = async (selectedUploadId) => {
    const targetUploadId = selectedUploadId || uploadId;
    if (!targetUploadId) {
      alert("Please upload data first or select an upload from the list");
      return;
    }

    setLoading(true);
    try {
      console.log("Generating forecast for upload ID:", targetUploadId);
      const response = await axios.post(`${API}/generate-forecast`, {
        upload_id: targetUploadId,
        forecast_months: forecastMonths,
      });

      console.log("Forecast response:", response.data);
      setForecast(response.data);
      setActiveTab("results");
      alert("Forecast generated successfully!");
    } catch (error) {
      console.error("Error generating forecast:", error);
      alert("Error generating forecast: " + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const exportCSV = async () => {
    if (!forecast) return;

    try {
      const response = await axios.get(
        `${API}/export-forecast-csv/${forecast.upload_id}`,
        { responseType: "blob" }
      );

      const blob = new Blob([response.data], { type: "text/csv" });
      saveAs(blob, `forecast_${forecast.upload_id}.csv`);
    } catch (error) {
      console.error("Error exporting CSV:", error);
      alert("Error exporting CSV");
    }
  };

  const renderPreview = () => {
    if (!preview) return null;

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">Data Preview</h3>
        <div className="mb-4">
          <p className="text-sm text-gray-600">
            Total Rows: {preview.total_rows} | Columns: {preview.columns.length}
          </p>
        </div>

        {preview.issues.length > 0 && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
            <h4 className="font-semibold text-red-800 mb-2">Issues Found:</h4>
            <ul className="list-disc list-inside text-red-700">
              {preview.issues.map((issue, index) => (
                <li key={index}>{issue}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border">
            <thead>
              <tr className="bg-gray-50">
                {preview.columns.map((col, index) => (
                  <th
                    key={index}
                    className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.sample_data.slice(0, 5).map((row, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  {preview.columns.map((col, colIndex) => (
                    <td
                      key={colIndex}
                      className="px-4 py-2 whitespace-nowrap text-sm text-gray-900 border-b"
                    >
                      {row[col] || "N/A"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderForecastCharts = () => {
    if (!forecast) return null;

    // Prepare chart data
    const chartData = {
      labels: forecast.forecast_data.map((item) => item.month),
      datasets: [
        {
          label: "Predicted Revenue",
          data: forecast.forecast_data.map((item) => item.predicted_revenue),
          borderColor: "rgb(59, 130, 246)",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          fill: true,
        },
      ],
    };

    const chartOptions = {
      responsive: true,
      plugins: {
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: `${forecast.forecast_months} Month Sales Forecast`,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
          }
        }
      }
    };

    // Feature importance chart
    const featureLabels = Object.keys(forecast.feature_importance);
    const featureValues = Object.values(forecast.feature_importance);
    
    const featureImportanceData = {
      labels: featureLabels,
      datasets: [
        {
          label: "Feature Importance",
          data: featureValues,
          backgroundColor: [
            "rgba(255, 99, 132, 0.8)",
            "rgba(54, 162, 235, 0.8)",
            "rgba(255, 205, 86, 0.8)",
            "rgba(75, 192, 192, 0.8)",
            "rgba(153, 102, 255, 0.8)",
            "rgba(255, 159, 64, 0.8)",
            "rgba(199, 199, 199, 0.8)",
            "rgba(83, 102, 255, 0.8)",
            "rgba(255, 99, 255, 0.8)",
            "rgba(99, 255, 132, 0.8)",
            "rgba(255, 192, 203, 0.8)",
          ],
        },
      ],
    };

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-md p-6">
          <Line data={chartData} options={chartOptions} />
        </div>
        <div className="bg-white rounded-lg shadow-md p-6">
          <Bar
            data={featureImportanceData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  display: false,
                },
                title: {
                  display: true,
                  text: "Feature Importance",
                },
              },
            }}
          />
        </div>
      </div>
    );
  };

  const renderResults = () => {
    if (!forecast) {
      console.log("No forecast data available");
      return null;
    }

    console.log("Rendering forecast results:", forecast);
    
    return (
      <div className="space-y-6">
        {/* Model Accuracy */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Model Accuracy</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="font-semibold text-blue-800">MAPE</h4>
              <p className="text-2xl font-bold text-blue-600">
                {(forecast.model_accuracy.mape * 100).toFixed(2)}%
              </p>
              <p className="text-sm text-blue-600">Mean Absolute Percentage Error</p>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <h4 className="font-semibold text-green-800">RMSE</h4>
              <p className="text-2xl font-bold text-green-600">
                ${forecast.model_accuracy.rmse.toLocaleString()}
              </p>
              <p className="text-sm text-green-600">Root Mean Square Error</p>
            </div>
          </div>
        </div>

        {/* Charts */}
        {renderForecastCharts()}

        {/* Business Insights */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Business Insights</h3>
          <div className="space-y-3">
            {forecast.insights.map((insight, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="bg-blue-100 rounded-full p-2 flex-shrink-0">
                  <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-gray-700">{insight}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Action Recommendations */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Recommended Actions</h3>
          <div className="space-y-3">
            <div className="flex items-start space-x-3">
              <div className="bg-green-100 rounded-full p-2 flex-shrink-0">
                <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-700">Focus marketing efforts on high-performing categories and regions</p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-yellow-100 rounded-full p-2 flex-shrink-0">
                <svg className="w-4 h-4 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-700">Investigate underperforming categories for potential improvements</p>
            </div>
            <div className="flex items-start space-x-3">
              <div className="bg-purple-100 rounded-full p-2 flex-shrink-0">
                <svg className="w-4 h-4 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-700">Monitor forecast accuracy and adjust strategies based on actual performance</p>
            </div>
          </div>
        </div>

        {/* Export Options */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold mb-4">Export Options</h3>
          <div className="flex space-x-4">
            <button
              onClick={exportCSV}
              className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
              <span>Export CSV</span>
            </button>
            <button
              disabled
              className="bg-gray-400 text-white px-4 py-2 rounded-lg flex items-center space-x-2 cursor-not-allowed"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6zm5 6a1 1 0 10-2 0v3.586l-1.293-1.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V8z" clipRule="evenodd" />
              </svg>
              <span>Export PDF (Coming Soon)</span>
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            📊 Sales Forecasting Dashboard
          </h1>
          <p className="text-gray-600">
            Upload your sales data and generate AI-powered forecasts with business insights
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-md mb-6">
          <nav className="flex space-x-1 p-1">
            {[
              { key: "upload", label: "📤 Upload Data" },
              { key: "forecast", label: "🔮 Generate Forecast" },
              { key: "results", label: "📈 Results & Insights" },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.key
                    ? "bg-blue-500 text-white"
                    : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {/* Upload Tab */}
          {activeTab === "upload" && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Upload Sales Data</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Choose CSV or Excel file
                  </label>
                  <input
                    type="file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
                {renderPreview()}
                {file && preview && preview.issues.length === 0 && (
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium"
                  >
                    {loading ? "Uploading..." : "Upload Data"}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Forecast Tab */}
          {activeTab === "forecast" && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4">Generate Forecast</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Forecast Period (months)
                    </label>
                    <select
                      value={forecastMonths}
                      onChange={(e) => setForecastMonths(parseInt(e.target.value))}
                      className="block w-full md:w-48 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value={3}>3 months</option>
                      <option value={6}>6 months</option>
                      <option value={12}>12 months</option>
                    </select>
                  </div>
                  <button
                    onClick={() => generateForecast()}
                    disabled={loading || !uploadId}
                    className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium"
                  >
                    {loading ? "Generating..." : "🔮 Run Forecast"}
                  </button>
                </div>
              </div>

              {/* Previous Uploads */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Previous Uploads</h3>
                <div className="space-y-2">
                  {uploads.map((upload) => (
                    <div
                      key={upload.id}
                      className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50"
                    >
                      <div>
                        <p className="font-medium">{upload.filename}</p>
                        <p className="text-sm text-gray-600">
                          {upload.total_records} records •{" "}
                          {new Date(upload.upload_date).toLocaleDateString()}
                        </p>
                      </div>
                      <button
                        onClick={() => generateForecast(upload.id)}
                        disabled={loading}
                        className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg"
                      >
                        Generate Forecast
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === "results" && (
            <div>
              {forecast ? (
                renderResults()
              ) : (
                <div className="bg-white rounded-lg shadow-md p-6 text-center">
                  <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">No Forecast Data</h3>
                  <p className="text-gray-600 mb-4">
                    Please upload data and generate a forecast to see results here.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;