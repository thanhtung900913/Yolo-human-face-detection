/**
 * stats.js
 * Handle statistics tracking (FPS, counts, emotions)
 */

import state from './state.js';
import config from './config.js';

// DOM elements
let personCountElement, faceCountElement, fpsCounterElement, emotionsStatsElement, actionsStatsElement;

/**
 * Initialize stats module
 * @param {Object} elements - DOM elements
 */
export function initStats(elements) {
    personCountElement = elements.personCountElement;
    faceCountElement = elements.faceCountElement;
    fpsCounterElement = elements.fpsCounterElement;
    emotionsStatsElement = document.getElementById('emotionsStats');
    actionsStatsElement = document.getElementById('actionsStats');
}

/**
 * Setup FPS counter
 */
export function setupFpsCounter() {
    // Clear existing timer if any
    if (state.fpsTimerId) {
        clearInterval(state.fpsTimerId);
    }
    
    // Setup new timer
    state.frameCount = 0;
    state.lastFrameTime = performance.now();
    
    state.fpsTimerId = setInterval(() => {
        const currentTime = performance.now();
        const elapsedTime = (currentTime - state.lastFrameTime) / 1000;
        
        if (elapsedTime > 0) {
            const fps = Math.round(state.frameCount / elapsedTime);
            fpsCounterElement.textContent = fps;
            
            // Reset counters
            state.frameCount = 0;
            state.lastFrameTime = currentTime;
        }
    }, 1000);
    
    console.log("FPS counter started");
}

/**
 * Update statistics display
 * @param {number} personCount - Number of detected persons
 * @param {number} faceCount - Number of detected faces
 * @param {Array} personBoxes - Person detection boxes with action data
 * @param {Array} faceBoxes - Face detection boxes with emotion data
 */
export function updateStats(personCount, faceCount, personBoxes, faceBoxes) {
    personCountElement.textContent = personCount;
    faceCountElement.textContent = faceCount;
    
    // Update emotion statistics if enabled
    if (config.showEmotions && faceBoxes && faceBoxes.length > 0) {
        updateEmotionStats(faceBoxes);
    } else {
        // Clear emotion stats if disabled or no faces
        if (emotionsStatsElement) {
            emotionsStatsElement.innerHTML = '';
        }
    }
    
    // Update action statistics if enabled
    if (config.showActions && personBoxes && personBoxes.length > 0) {
        updateActionStats(personBoxes);
    } else {
        // Clear action stats if disabled or no persons
        if (actionsStatsElement) {
            actionsStatsElement.innerHTML = '';
        }
    }
}

/**
 * Update emotion statistics
 * @param {Array} faceBoxes - Face detection boxes with emotion data
 */
function updateEmotionStats(faceBoxes) {
    // Skip if element doesn't exist yet
    if (!emotionsStatsElement) return;
    
    // Count emotions
    const emotionCounts = {};
    
    faceBoxes.forEach(box => {
        if (box.emotion) {
            emotionCounts[box.emotion] = (emotionCounts[box.emotion] || 0) + 1;
        }
    });
    
    // Build HTML for emotion stats
    let emotionStatsHtml = '<h4>Cảm xúc:</h4>';
    
    if (Object.keys(emotionCounts).length === 0) {
        emotionStatsHtml += '<div class="emotion-stat-item">Không có dữ liệu</div>';
    } else {
        Object.entries(emotionCounts).forEach(([emotion, count]) => {
            const color = config.emotionColors[emotion] || config.emotionColor;
            emotionStatsHtml += `
                <div class="emotion-stat-item">
                    <span class="emotion-indicator" style="background-color: ${color}"></span>
                    <span class="emotion-name">${emotion}:</span>
                    <span class="emotion-count">${count}</span>
                </div>
            `;
        });
    }
    
    // Update the DOM
    emotionsStatsElement.innerHTML = emotionStatsHtml;
}

/**
 * Update action statistics
 * @param {Array} personBoxes - Person detection boxes with action data
 */
function updateActionStats(personBoxes) {
    // Skip if element doesn't exist yet
    if (!actionsStatsElement) return;
    
    // Count actions
    const actionCounts = {};
    
    personBoxes.forEach(box => {
        if (box.action) {
            actionCounts[box.action] = (actionCounts[box.action] || 0) + 1;
        }
    });
    
    // Build HTML for action stats
    let actionStatsHtml = '<h4>Hành vi:</h4>';
    
    if (Object.keys(actionCounts).length === 0) {
        actionStatsHtml += '<div class="action-stat-item">Không có dữ liệu</div>';
    } else {
        Object.entries(actionCounts).forEach(([action, count]) => {
            const color = config.actionColors[action] || config.actionColor;
            actionStatsHtml += `
                <div class="action-stat-item">
                    <span class="action-indicator" style="background-color: ${color}"></span>
                    <span class="action-name">${action}:</span>
                    <span class="action-count">${count}</span>
                </div>
            `;
        });
    }
    
    // Update the DOM
    actionsStatsElement.innerHTML = actionStatsHtml;
}

/**
 * Reset statistics counters
 */
export function resetStats() {
    personCountElement.textContent = '0';
    faceCountElement.textContent = '0';
    fpsCounterElement.textContent = '0';
    
    // Clear emotion stats
    if (emotionsStatsElement) {
        emotionsStatsElement.innerHTML = '';
    }
    
    // Clear action stats
    if (actionsStatsElement) {
        actionsStatsElement.innerHTML = '';
    }
}
