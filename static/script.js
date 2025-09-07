let uploadedFile = null;
let forecastChart = null;

document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    // File upload handling
    uploadBox.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
        
        if (file && (validTypes.includes(file.type) || file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.xls'))) {
            uploadedFile = file;
            uploadBox.querySelector('p').textContent = `Selected: ${file.name}`;
            uploadBtn.disabled = false;
        } else {
            showError('Please select a valid CSV or Excel file');
        }
    });

    // Drag and drop
    uploadBox.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', function() {
        uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        const validTypes = ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
        
        if (file && (validTypes.includes(file.type) || file.name.endsWith('.csv') || file.name.endsWith('.xlsx') || file.name.endsWith('.xls'))) {
            uploadedFile = file;
            fileInput.files = e.dataTransfer.files;
            uploadBox.querySelector('p').textContent = `Selected: ${file.name}`;
            uploadBtn.disabled = false;
        } else {
            showError('Please select a valid CSV or Excel file');
        }
    });

    // Upload and process
    uploadBtn.addEventListener('click', function() {
        if (!uploadedFile) return;

        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('unit_cost', document.getElementById('unitCost').value);
        formData.append('model_type', document.getElementById('modelType').value);

        updateWorkflowStep(2);
        showLoading();
        hideError();
        hideResults();

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.success) {
                updateWorkflowStep(3);
                setTimeout(() => updateWorkflowStep(4), 1000);
                displayResults(data);
            } else {
                showError(data.error);
                updateWorkflowStep(1);
            }
        })
        .catch(err => {
            hideLoading();
            showError('Error processing file: ' + err.message);
        });
    });

    function showLoading() {
        loading.style.display = 'block';
    }

    function hideLoading() {
        loading.style.display = 'none';
    }

    function showError(message) {
        error.style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
    }

    function hideError() {
        error.style.display = 'none';
    }

    function hideResults() {
        results.style.display = 'none';
    }

    function displayResults(data) {
        results.style.display = 'block';
        
        // Update workflow steps
        updateWorkflowStep(5);
        
        // Display accuracy
        displayAccuracy(data.accuracy);
        
        // Display alerts
        displayAlerts(data.alerts);
        
        // Display recommendations
        displayRecommendations(data.recommendations);
        
        // Display budget overview
        displayBudgetOverview(data.budget);
        
        // Create chart
        createForecastChart(data.historical, data.forecast);
        
        // Create forecast table
        createForecastTable(data.forecast, data.budget.unit_cost);
        
        // Display stock analysis
        displayStockAnalysis(data.stock_analysis);
        
        // Display material info
        displayMaterialInfo(data.material_info);
        
        // Setup report download buttons
        setupReportDownloads();
    }

    function createForecastChart(historical, forecast) {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (forecastChart) {
            forecastChart.destroy();
        }

        const allDates = [...historical.dates, ...forecast.dates];
        const consumptionData = [...historical.consumption, ...Array(forecast.dates.length).fill(null)];
        const forecastData = [...Array(historical.dates.length).fill(null), ...forecast.forecasts];
        const stockData = [...historical.stock, ...Array(forecast.dates.length).fill(null)];

        forecastChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allDates,
                datasets: [
                    {
                        label: 'Historical Consumption',
                        data: consumptionData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'Current Stock',
                        data: stockData,
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        fill: false,
                        tension: 0.4
                    },
                    {
                        label: 'FY Forecast',
                        data: forecastData,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        fill: false,
                        tension: 0.4,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'WCL Material Consumption & FY Forecast'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Quantity (Units)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Financial Year Timeline'
                        }
                    }
                }
            }
        });
    }

    function createForecastTable(forecast, unitCost) {
        const tableContainer = document.getElementById('forecastTable');
        tableContainer.className = 'forecast-table';
        
        let tableHTML = '';
        forecast.dates.forEach((date, index) => {
            const quantity = Math.round(forecast.forecasts[index]);
            const cost = quantity * unitCost;
            tableHTML += `
                <div class="forecast-item">
                    <div class="date">${new Date(date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</div>
                    <div class="value">${quantity} units</div>
                    <div class="cost">₹${cost.toLocaleString('en-IN')}</div>
                </div>
            `;
        });
        
        tableContainer.innerHTML = tableHTML;
    }
    
    function displayAlerts(alerts) {
        const alertsContainer = document.getElementById('alertsSection');
        if (alerts && alerts.length > 0) {
            let alertsHTML = '';
            alerts.forEach(alert => {
                alertsHTML += `<div class="alert ${alert.type}">${alert.message}</div>`;
            });
            alertsContainer.innerHTML = alertsHTML;
        } else {
            alertsContainer.innerHTML = '';
        }
    }
    
    function displayBudgetOverview(budget) {
        const budgetContainer = document.getElementById('budgetCards');
        budgetContainer.className = 'budget-cards';
        
        budgetContainer.innerHTML = `
            <div class="budget-card">
                <h4>Total Forecast</h4>
                <div class="value">${budget.total_forecast} units</div>
            </div>
            <div class="budget-card">
                <h4>Monthly Average</h4>
                <div class="value">${budget.monthly_average} units</div>
            </div>
            <div class="budget-card safety">
                <h4>Safety Buffer</h4>
                <div class="value">${budget.safety_buffer} units</div>
            </div>
            <div class="budget-card total">
                <h4>Total FY Budget</h4>
                <div class="value">₹${budget.total_budget.toLocaleString('en-IN')}</div>
            </div>
        `;
    }
    
    function displayStockAnalysis(stockAnalysis) {
        const stockContainer = document.getElementById('stockAnalysis');
        stockContainer.innerHTML = `
            <div class="stock-metrics">
                <div class="stock-metric">
                    <h4>Current Stock</h4>
                    <div class="value">${stockAnalysis.current_stock} units</div>
                </div>
                <div class="stock-metric">
                    <h4>Months Remaining</h4>
                    <div class="value">${stockAnalysis.months_remaining} months</div>
                </div>
                <div class="stock-metric">
                    <h4>Avg Monthly Usage</h4>
                    <div class="value">${stockAnalysis.avg_monthly_consumption} units</div>
                </div>
            </div>
        `;
    }
    
    function displayAccuracy(accuracy) {
        const accuracyContainer = document.getElementById('accuracyCard');
        const statusColor = accuracy.percentage >= 90 ? '#10b981' : 
                           accuracy.percentage >= 85 ? '#3b82f6' : '#f59e0b';
        
        accuracyContainer.innerHTML = `
            <div class="accuracy-card" style="border-left-color: ${statusColor}">
                <div class="accuracy-header">
                    <h3>Model Accuracy</h3>
                    <span class="accuracy-badge" style="background: ${statusColor}">${accuracy.status}</span>
                </div>
                <div class="accuracy-value">${accuracy.percentage}%</div>
                <div class="accuracy-details">
                    <p>Model: ${accuracy.model.charAt(0).toUpperCase() + accuracy.model.slice(1)}</p>
                    <p>Tested on historical FY data</p>
                </div>
            </div>
        `;
    }
    
    function displayRecommendations(recommendations) {
        const container = document.getElementById('recommendationsContainer');
        let html = '';
        
        recommendations.forEach(rec => {
            const priorityColor = rec.priority === 'high' ? '#ef4444' : 
                                 rec.priority === 'medium' ? '#f59e0b' : '#10b981';
            
            html += `
                <div class="recommendation-card" style="border-left-color: ${priorityColor}">
                    <div class="recommendation-header">
                        <h4>${rec.title}</h4>
                        <span class="priority-badge ${rec.priority}">${rec.priority.toUpperCase()}</span>
                    </div>
                    <p class="recommendation-message">${rec.message}</p>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    function displayMaterialInfo(materialInfo) {
        // Add material info to the page header
        const header = document.querySelector('header p');
        if (materialInfo && materialInfo.name !== 'Consumable Material') {
            header.innerHTML = `Western Coalfields Limited - ${materialInfo.name} Forecasting System<br>
                              <small style="opacity: 0.8; font-size: 0.9em;">
                                Data: ${materialInfo.data_points} points (${materialInfo.date_range})
                              </small>`;
        }
    }
    
    function setupReportDownloads() {
        document.getElementById('downloadExcel').addEventListener('click', function() {
            this.innerHTML = '⏳ Generating Excel...';
            this.disabled = true;
            
            const link = document.createElement('a');
            link.href = '/download/excel';
            link.download = 'WCL_Comprehensive_Report.xlsx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            setTimeout(() => {
                this.innerHTML = '📄 Download Excel Report';
                this.disabled = false;
            }, 2000);
        });
        
        document.getElementById('downloadPDF').addEventListener('click', function() {
            this.innerHTML = '⏳ Generating PDF...';
            this.disabled = true;
            
            const link = document.createElement('a');
            link.href = '/download/pdf';
            link.download = 'WCL_Comprehensive_Report.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            setTimeout(() => {
                this.innerHTML = '📄 Download PDF Report';
                this.disabled = false;
            }, 3000);
        });
    }
    
    function updateWorkflowStep(currentStep) {
        // Reset all steps
        for (let i = 1; i <= 5; i++) {
            const step = document.getElementById(`step${i}`);
            step.classList.remove('active', 'completed');
            
            if (i < currentStep) {
                step.classList.add('completed');
            } else if (i === currentStep) {
                step.classList.add('active');
            }
        }
    }
});