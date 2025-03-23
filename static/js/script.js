document.addEventListener('DOMContentLoaded', function() {
    const trainForm = document.getElementById('train-form');
    const predictionForm = document.getElementById('prediction-form');
    const resultsDisplay = document.getElementById('results-display');
    const visualizationContainer = document.getElementById('visualization-container');
    const featureInputs = document.getElementById('feature-inputs');
    const predictionResult = document.getElementById('prediction-result');

    trainForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(trainForm);
        
        fetch('/train', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultsDisplay.innerHTML = `<p class="error">${data.error}</p>`;
                return;
            }
            
            resultsDisplay.innerHTML = `
                <p>Accuracy: ${data.accuracy.toFixed(4)}</p>
                <p>Number of iterations: ${data.iterations}</p>
                <p>Final weights: ${data.weights.map(w => w.toFixed(4)).join(', ')}</p>
                <p>Final bias: ${data.bias.toFixed(4)}</p>
            `;
            
            // Display visualization if available
            if (data.visualization) {
                visualizationContainer.innerHTML = `<img src="${data.visualization}" alt="Decision Boundary" class="visualization">`;
            } else {
                visualizationContainer.innerHTML = '';
            }
            
            // Create input fields for features
            featureInputs.innerHTML = '';
            for (let i = 0; i < data.num_features; i++) {
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.name = `feature_${i}`;
                input.placeholder = `Feature ${i + 1}`;
                input.required = true;
                featureInputs.appendChild(input);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultsDisplay.innerHTML = '<p class="error">Error occurred during training.</p>';
        });
    });

    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(predictionForm);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                predictionResult.innerHTML = `<p class="error">${data.error}</p>`;
                return;
            }
            predictionResult.innerHTML = `<p>Prediction: ${data.prediction}</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
            predictionResult.innerHTML = '<p class="error">Error occurred during prediction.</p>';
        });
    });
});
