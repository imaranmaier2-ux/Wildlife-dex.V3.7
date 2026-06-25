// HARDWARE ELEMENT TARGETS
const videoElement = document.querySelector('video');
const hudDisplay = document.getElementById('detection-status-text');
const manualLabelField = document.getElementById('manual-label-field');
const lockBtn = document.getElementById('lock-signature-btn') || document.getElementById('lock-btn'); 

let netModel = null;
let memoryBank = []; // Stores your captured visual blueprints
const MATCH_THRESHOLD = 0.85; // Tuning threshold for embedding matches

// Calculate mathematical alignment vector distance (Cosine Similarity)
function calculateCosineSimilarity(vectorA, vectorB) {
    let dotProduct = 0.0;
    let normA = 0.0;
    let normB = 0.0;
    for (let i = 0; i < vectorA.length; i++) {
        dotProduct += vectorA[i] * vectorB[i];
        normA += vectorA[i] * vectorA[i];
        normB += vectorB[i] * vectorB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Boot up AI system layers
async function initAISystem() {
    try {
        if (hudDisplay) hudDisplay.innerText = "DEX CORE: INITIALIZING AI...";
        
        // Load MobileNet for classification and internal embeddings
        netModel = await mobilenet.load({ version: 1, alpha: 1.0 });
        
        if (hudDisplay) {
            hudDisplay.innerText = "AI SYSTEM READY";
            hudDisplay.style.color = "#00FF00";
        }
        
        // Start the processing layer loop (runs roughly every 40ms)
        setInterval(processLiveTelemetry, 40);
        
    } catch (err) {
        console.error(err);
        if (hudDisplay) hudDisplay.innerText = `CORE EXCEPTION: ${err.message}`;
    }
}

// Checking loop for live camera data
async function processLiveTelemetry() {
    if (!videoElement || videoElement.readyState !== 4) return;

    let currentEmbedding = null;

    // Extract feature embeddings using TensorFlow tidy to prevent memory leaks
    try {
        const logits = tf.tidy(() => {
            const tfImage = tf.browser.fromPixels(videoElement);
            const resized = tf.image.resizeBilinear(tfImage, [224, 224]);
            const batched = resized.expandDims(0);
            
            // Activate activation layer for internal signatures
            return netModel.model.predict(batched).squeeze();
        });

        currentEmbedding = await logits.data();
        logits.dispose();
    } catch (e) {
        // Fallback if model internal prediction structural layers differ
        currentEmbedding = null;
    }

    let nearestMatch = null;
    let highestSimilarity = -1.0;

    // Scan through saved visual blueprints in the memory bank
    if (currentEmbedding && memoryBank.length > 0) {
        memoryBank.forEach(memoryItem => {
            const similarity = calculateCosineSimilarity(currentEmbedding, memoryItem.embedding);
            if (similarity > highestSimilarity) {
                highestSimilarity = similarity;
                nearestMatch = memoryItem.label;
            }
        });
    }

    // UI Decision Tree: Custom Blueprint vs Standard Classification
    if (highestSimilarity > MATCH_THRESHOLD && nearestMatch) {
        if (hudDisplay) {
            hudDisplay.innerText = `LOCK: ${nearestMatch.toUpperCase()}\n[Manual Override Match]`;
            hudDisplay.style.color = "#00FF00";
        }
    } else {
        // Run standard MobileNet classification if no custom lock matches
        const predictions = await netModel.classify(videoElement);
        if (predictions && predictions.length > 0) {
            const detectedName = predictions[0].className.split(',')[0];
            const confidence = Math.round(predictions[0].probability * 100);
            
            if (hudDisplay) {
                hudDisplay.innerText = `DETECT: ${detectedName.toUpperCase()} (${confidence}%)`;
                hudDisplay.style.color = "#00FF00";
            }
        }
    }
}

// Trigger Button: Lock parameters directly into memory array
if (lockBtn) {
    lockBtn.addEventListener('click', async () => {
        const inputField = document.getElementById('input-label') || manualLabelField;
        const assignmentLabel = inputField && inputField.value ? inputField.value.trim().toLowerCase() : '';
        
        if (!assignmentLabel) {
            alert("Target label name first!");
            return;
        }
        
        if (!netModel || videoElement.readyState !== 4) return;
        
        if (hudDisplay) hudDisplay.innerText = "CAPTURING VISUAL BLUEPRINT...";
        
        // Capture frame and store vector data
        const webCamTensor = tf.browser.fromPixels(videoElement);
        const logits = tf.tidy(() => {
            const resized = tf.image.resizeBilinear(webCamTensor, [224, 224]);
            const batched = resized.expandDims(0);
            return netModel.model.predict(batched).squeeze();
        });
        
        const embeddingData = await logits.data();
        logits.dispose();
        webCamTensor.dispose();
        
        // Add to tracking bank
        memoryBank.push({
            label: assignmentLabel,
            embedding: embeddingData
        });
        
        if (inputField) inputField.value = '';
        if (hudDisplay) hudDisplay.innerText = `SUCCESS: LOCKED [${assignmentLabel.toUpperCase()}]`;
    });
}

// Fire up hardware capture elements when page initializes
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
    .then(stream => {
        videoElement.srcObject = stream;
        videoElement.muted = true;
        videoElement.play();
        initAISystem();
    })
    .catch(err => {
        console.error("Camera access denied:", err);
        if (hudDisplay) hudDisplay.innerText = "CORE ERROR: CAMERA ACCESS DENIED";
    });
