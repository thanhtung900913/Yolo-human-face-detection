/**
 * main.js
 * Main entry point for the application
 */

import { initCamera, loadCameras } from './camera.js';
import { initDetection } from './detection.js';
import { initStats } from './stats.js';
import { initUI, setupEventListeners, configureUIForDevice, resetCanvas } from './ui.js';
import state from './state.js';

// Initialize application when DOM is fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    // Get DOM elements
    const elements = {
        // Video elements
        video: document.getElementById('webcam'),
        overlay: document.getElementById('overlay'),
        
        // Control elements
        cameraSelect: document.getElementById('cameraSelect'),
        startButton: document.getElementById('startButton'),
        stopButton: document.getElementById('stopButton'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        toggleRearCamera: document.getElementById('toggleRearCamera'),
        
        // Display toggles
        togglePerson: document.getElementById('togglePerson'),
        toggleFace: document.getElementById('toggleFace'),
        toggleConfidence: document.getElementById('toggleConfidence'),
        toggleEmotions: document.getElementById('toggleEmotions'),
        toggleActions: document.getElementById('toggleActions'),
        toggleNames: document.getElementById('toggleNames'),
        
        // Stat elements
        personCountElement: document.getElementById('personCount'),
        faceCountElement: document.getElementById('faceCount'),
        fpsCounterElement: document.getElementById('fpsCounter'),
        emotionsStatsElement: document.getElementById('emotionsStats'),
        actionsStatsElement: document.getElementById('actionsStats'),
    };
    
    // Initialize modules
    initCamera(elements);
    initDetection(elements);
    initStats(elements);
    initUI(elements);
    
    // Configure UI for device type
    configureUIForDevice();
    
    // Load available cameras
    await loadCameras();
    
    // Setup event listeners
    setupEventListeners();
    
    // Disable stop button initially
    elements.stopButton.disabled = true;
    
    // Hide loading overlay
    elements.loadingOverlay.classList.add('hidden');
    
    // Reset canvas to initial state
    resetCanvas();
});