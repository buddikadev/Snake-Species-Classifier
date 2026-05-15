document.getElementById('predictBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const resultCard = document.getElementById('resultCard');
    const previewImg = document.getElementById('previewImg');
    const predSpecies = document.getElementById('predSpecies');
    const confLevel = document.getElementById('confLevel');

    if (fileInput.files.length === 0) {
        alert("Please select an image first.");
        return;
    }

    const file = fileInput.files[0];
    
    // Show Preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
    }
    reader.readAsDataURL(file);

    // Show Card and Loading State
    resultCard.classList.remove('hidden');
    predSpecies.innerText = "Analyzing...";
    confLevel.innerText = "";

    // Prepare Data for API
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Send to FastAPI Backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();

        if (data.success) {
            predSpecies.innerText = data.prediction;
            confLevel.innerText = "Confidence Level: " + data.confidence;
        } else {
            predSpecies.innerText = "Error analyzing image.";
            confLevel.innerText = data.error;
        }
    } catch (error) {
        predSpecies.innerText = "Connection Error.";
        console.error("Error:", error);
    }
});