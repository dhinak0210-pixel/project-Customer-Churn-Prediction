document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictor-form');
    const probCircle = document.getElementById('prob-circle');
    const probText = document.getElementById('prob-text');
    const riskLabel = document.getElementById('risk-label');
    const shapContainer = document.getElementById('shap-container');
    const btnPredict = document.querySelector('.btn-predict');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        btnPredict.innerText = 'Calculating...';
        btnPredict.disabled = true;
        
        const formData = new FormData(form);
        const features = {
            gender: formData.get('gender'),
            SeniorCitizen: parseInt(formData.get('SeniorCitizen')),
            Partner: "No", // default placeholders for simple form
            Dependents: "No",
            tenure: parseInt(formData.get('tenure')),
            PhoneService: "Yes",
            MultipleLines: "No",
            InternetService: formData.get('InternetService'),
            OnlineSecurity: "No",
            OnlineBackup: "No",
            DeviceProtection: "No",
            TechSupport: "No",
            StreamingTV: "No",
            StreamingMovies: "No",
            Contract: formData.get('Contract'),
            PaperlessBilling: "Yes",
            PaymentMethod: "Electronic check",
            MonthlyCharges: parseFloat(formData.get('MonthlyCharges'))
        };

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(features)
            });

            if (!response.ok) throw new Error('API Error');

            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error('Fetch error:', error);
            riskLabel.innerText = 'Service Unavailable';
            riskLabel.style.color = 'var(--danger)';
        } finally {
            btnPredict.innerText = 'Analyze Risk';
            btnPredict.disabled = false;
        }
    });

    function updateUI(data) {
        const percent = Math.round(data.churn_probability * 100);
        
        // Update Circle
        probCircle.style.setProperty('--percent', percent);
        probText.innerText = `${percent}%`;
        
        // Update Risk Label
        if (percent > 70) {
            riskLabel.innerText = 'CRITICAL RISK';
            riskLabel.style.color = 'var(--danger)';
        } else if (percent > 40) {
            riskLabel.innerText = 'ELEVATED RISK';
            riskLabel.style.color = 'var(--warning)';
        } else {
            riskLabel.innerText = 'STABLE CUSTOMER';
            riskLabel.style.color = 'var(--success)';
        }

        // Render SHAP features
        renderSHAP(data);
    }

    function renderSHAP(data) {
        shapContainer.innerHTML = '';
        
        // Pair feature names with values and sort by absolute intensity
        const features = data.feature_names.map((name, i) => ({
            name: name,
            value: data.shap_values[i],
            absVal: Math.abs(data.shap_values[i])
        })).sort((a, b) => b.absVal - a.absVal);

        // Take top 6 influential features
        const topFeatures = features.slice(0, 6);
        const maxVal = Math.max(...topFeatures.map(f => f.absVal));

        topFeatures.forEach(f => {
            const item = document.createElement('div');
            item.className = 'shap-item';
            
            const isPositive = f.value > 0;
            const width = (f.absVal / maxVal) * 100;

            item.innerHTML = `
                <div class="shap-label-row">
                    <span class="feature-name">${f.name}</span>
                    <span class="feature-impact ${isPositive ? 'pos' : 'neg'}">${isPositive ? '+' : ''}${f.value.toFixed(3)}</span>
                </div>
                <div class="shap-bar-container">
                    <div class="shap-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${width}%"></div>
                </div>
            `;
            shapContainer.appendChild(item);
        });
    }
});
