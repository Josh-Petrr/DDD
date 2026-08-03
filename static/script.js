/**
 * DrowsiGuard — Live Drowsiness Detection Client
 * 
 * Handles webcam capture, WebSocket streaming to the FastAPI backend,
 * and dynamic UI updates based on inference results.
 */

// ─── State ───
let ws = null;
let stream = null;
let captureInterval = null;
let isRunning = false;
let sessionStart = null;
let frameCount = 0;
let alertCount = 0;
let confidenceSum = 0;
let confidenceCount = 0;
let lastFrameTime = 0;
let fpsValues = [];

// Audio alarm
let audioCtx = null;

// ─── DOM Elements ───
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('captureCanvas');
const ctx = canvas.getContext('2d');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraOverlay = document.getElementById('cameraOverlay');
const connectionStatus = document.getElementById('connectionStatus');

const threatCard = document.getElementById('threatCard');
const threatIcon = document.getElementById('threatIcon');
const threatLabel = document.getElementById('threatLabel');
const threatConfidence = document.getElementById('threatConfidence');
const threatBar = document.getElementById('threatBar');
const alertFlash = document.getElementById('alertFlash');

const bufferBar = document.getElementById('bufferBar');
const bufferCount = document.getElementById('bufferCount');
const bufferHint = document.getElementById('bufferHint');

const faceStatus = document.getElementById('faceStatus');
const fpsBadge = document.getElementById('fpsBadge');

const earBar = document.getElementById('earBar');
const marBar = document.getElementById('marBar');
const ebBar = document.getElementById('ebBar');
const tiltBar = document.getElementById('tiltBar');
const earValue = document.getElementById('earValue');
const marValue = document.getElementById('marValue');
const ebValue = document.getElementById('ebValue');
const tiltValue = document.getElementById('tiltValue');

const statDuration = document.getElementById('statDuration');
const statFrames = document.getElementById('statFrames');
const statAlerts = document.getElementById('statAlerts');
const statAvgConf = document.getElementById('statAvgConf');


// ─── Start Detection ───
async function startDetection() {
    try {
        // 1. Request webcam access
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false,
        });
        webcam.srcObject = stream;
        
        // Set canvas size
        canvas.width = 640;
        canvas.height = 480;
        
        // 2. Connect WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            setConnectionStatus(true);
            
            // 3. Start capturing frames at ~10 FPS
            captureInterval = setInterval(captureAndSend, 100);
            isRunning = true;
            sessionStart = Date.now();
            frameCount = 0;
            alertCount = 0;
            confidenceSum = 0;
            confidenceCount = 0;
            
            // Update session duration every second
            setInterval(updateDuration, 1000);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateUI(data);
        };
        
        ws.onclose = () => {
            console.log('WebSocket closed');
            setConnectionStatus(false);
        };
        
        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            setConnectionStatus(false);
        };
        
        // Update button states
        startBtn.disabled = true;
        stopBtn.disabled = false;
        cameraOverlay.classList.add('hidden');
        
        // Initialize audio context
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
    } catch (err) {
        console.error('Failed to start:', err);
        alert('Could not access webcam. Please grant camera permissions.');
    }
}


// ─── Stop Detection ───
function stopDetection() {
    isRunning = false;
    
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    webcam.srcObject = null;
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    cameraOverlay.classList.remove('hidden');
    setConnectionStatus(false);
    
    // Reset threat display
    threatCard.className = 'threat-card glass-panel';
    threatIcon.className = 'threat-icon';
    threatLabel.className = 'threat-label';
    threatLabel.textContent = 'STANDBY';
    threatConfidence.textContent = '--';
    threatBar.style.width = '0%';
    alertFlash.className = 'alert-flash';
}


// ─── Capture Frame & Send ───
function captureAndSend() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !isRunning) return;
    
    // Draw webcam frame to canvas
    ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    
    // Convert to JPEG base64
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
    
    // Send to server
    ws.send(dataUrl);
    
    // Track FPS
    const now = performance.now();
    if (lastFrameTime > 0) {
        const fps = 1000 / (now - lastFrameTime);
        fpsValues.push(fps);
        if (fpsValues.length > 10) fpsValues.shift();
        const avgFps = fpsValues.reduce((a, b) => a + b, 0) / fpsValues.length;
        fpsBadge.textContent = `${avgFps.toFixed(1)} FPS`;
    }
    lastFrameTime = now;
    
    frameCount++;
    statFrames.textContent = frameCount;
}


// ─── Update UI from Server Response ───
function updateUI(data) {
    if (data.error) {
        console.warn('Server error:', data.error);
        return;
    }
    
    // Buffer status
    const fillPct = Math.round(data.buffer_fill * 100);
    bufferBar.style.width = `${fillPct}%`;
    bufferCount.textContent = `${Math.round(data.buffer_fill * 30)} / 30`;
    
    if (data.status === 'buffering') {
        bufferHint.textContent = `Collecting frames... ${fillPct}% ready`;
    } else {
        bufferHint.textContent = 'Buffer full - LSTM is predicting in real-time';
    }
    
    // Face detection status
    if (data.face_detected) {
        faceStatus.innerHTML = '<span class="status-dot detected"></span> Face detected';
    } else {
        faceStatus.innerHTML = '<span class="status-dot offline"></span> No face';
    }
    
    // Geometric features
    if (data.geo_features) {
        const geo = data.geo_features;
        
        earValue.textContent = geo.ear.toFixed(3);
        marValue.textContent = geo.mar.toFixed(3);
        ebValue.textContent = geo.eyebrow_dist.toFixed(3);
        tiltValue.textContent = geo.head_tilt.toFixed(3);
        
        // Bars (scale to reasonable visual range)
        earBar.style.width = `${Math.min(geo.ear / 0.4 * 100, 100)}%`;
        marBar.style.width = `${Math.min(geo.mar / 0.8 * 100, 100)}%`;
        ebBar.style.width = `${Math.min(geo.eyebrow_dist / 0.3 * 100, 100)}%`;
        tiltBar.style.width = `${Math.min(geo.head_tilt / 0.3 * 100, 100)}%`;
    }
    
    // Threat level (only when LSTM is predicting)
    if (data.status === 'predicting') {
        const drowsyProb = data.drowsy_prob;
        const confidence = data.confidence;
        
        // Track confidence for average
        confidenceSum += confidence;
        confidenceCount++;
        statAvgConf.textContent = `${((confidenceSum / confidenceCount) * 100).toFixed(0)}%`;
        
        // Update threat bar
        threatBar.style.width = `${(drowsyProb * 100).toFixed(0)}%`;
        
        if (data.is_drowsy) {
            // DANGER - Drowsy detected!
            threatCard.className = 'threat-card glass-panel danger';
            threatIcon.className = 'threat-icon danger';
            threatLabel.className = 'threat-label danger';
            threatLabel.textContent = 'DROWSY';
            threatConfidence.textContent = `${(confidence * 100).toFixed(1)}% confidence`;
            alertFlash.className = 'alert-flash danger';
            
            // Increment alert counter
            alertCount++;
            statAlerts.textContent = alertCount;
            
            // Play alarm sound
            playAlarm();
            
        } else {
            // SAFE - Awake
            threatCard.className = 'threat-card glass-panel safe';
            threatIcon.className = 'threat-icon safe';
            threatLabel.className = 'threat-label safe';
            threatLabel.textContent = 'AWAKE';
            threatConfidence.textContent = `${(confidence * 100).toFixed(1)}% confidence`;
            alertFlash.className = 'alert-flash';
        }
    }
}


// ─── Audio Alarm ───
let lastAlarmTime = 0;

function playAlarm() {
    const now = Date.now();
    // Only play alarm once every 2 seconds to avoid spam
    if (now - lastAlarmTime < 2000) return;
    lastAlarmTime = now;
    
    if (!audioCtx) return;
    
    // Create a sharp, attention-grabbing beep
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.type = 'square';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime);
    osc.frequency.setValueAtTime(660, audioCtx.currentTime + 0.1);
    osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.2);
    
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);
    
    osc.start(audioCtx.currentTime);
    osc.stop(audioCtx.currentTime + 0.4);
}


// ─── Helpers ───
function setConnectionStatus(isConnected) {
    const chip = connectionStatus;
    const dot = chip.querySelector('.status-dot');
    const label = chip.querySelector('span:last-child');
    
    if (isConnected) {
        dot.className = 'status-dot online';
        label.textContent = 'Connected';
    } else {
        dot.className = 'status-dot';
        label.textContent = 'Disconnected';
    }
}

function updateDuration() {
    if (!sessionStart || !isRunning) return;
    
    const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    statDuration.textContent = `${mins}:${secs}`;
}
