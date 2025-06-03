/**
 * state.js
 * Manages application state
 */

// Application state object
const state = {
    // Running state
    isRunning: false,
    isProcessing: false,
    
    // Timers
    processingTimerId: null,
    fpsTimerId: null,
    
    // Detection data
    personBoxes: [],
    faceBoxes: [],
    
    // Stream
    stream: null,
    
    // Performance metrics
    lastFrameTime: performance.now(),
    frameCount: 0
};

export default state;