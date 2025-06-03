/**
 * config.js
 * Configuration settings for the application
 */

// Check if the device is mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Export application settings
const config = {
    // Detection settings
    showPersons: true,
    showFaces: true,
    showConfidence: true,
    showEmotions: true,  // Hiển thị cảm xúc / Show emotions
    showActions: true,   // Hiển thị hành vi / Show actions
    showNames: true, 
    personColor: '#e74c3c',
    faceColor: '#2ecc71',
    emotionColor: '#ff9800',  // Màu mặc định cho cảm xúc / Default color for emotions
    
    // Màu sắc cho từng cảm xúc / Colors for different emotions
    emotionColors: {
        'Giận dữ': '#e74c3c',     // Đỏ / Red
        'Ghê tởm': '#9b59b6',     // Tím / Purple
        'Sợ hãi': '#34495e',      // Xám đậm / Dark gray
        'Vui vẻ': '#f1c40f',      // Vàng / Yellow
        'Buồn bã': '#3498db',     // Xanh dương / Blue
        'Ngạc nhiên': '#e67e22',  // Cam / Orange
        'Bình thường': '#95a5a6', // Xám nhạt / Light gray
        'Không xác định': '#7f8c8d' // Xám / Gray
    },

    // Màu sắc cho từng hành vi / Colors for different actions
    actionColors: {
        'Gọi điện': '#8e44ad',      // Tím đậm / Dark purple
        'Vỗ tay': '#16a085',        // Xanh lá đậm / Dark green
        'Đạp xe': '#2980b9',        // Xanh dương đậm / Dark blue
        'Khiêu vũ': '#c0392b',      // Đỏ đậm / Dark red
        'Uống nước': '#27ae60',     // Xanh lá / Green
        'Ăn uống': '#f39c12',       // Cam vàng / Yellow orange
        'Đánh nhau': '#d35400',     // Cam đậm / Dark orange
        'Ôm': '#1abc9c',            // Xanh ngọc / Turquoise
        'Cười': '#f1c40f',          // Vàng / Yellow
        'Nghe nhạc': '#3498db',     // Xanh dương / Blue
        'Chạy': '#e74c3c',          // Đỏ / Red
        'Ngồi': '#2c3e50',          // Đen xanh / Dark blue-black
        'Ngủ': '#34495e',           // Xám đậm / Dark gray
        'Nhắn tin': '#9b59b6',      // Tím / Purple
        'Dùng laptop': '#2980b9',   // Xanh dương đậm / Dark blue
        'Không xác định': '#7f8c8d'  // Xám / Gray
    },
    
    // Server settings
    serverUrl: 'http://127.0.0.1:8000/process_frame',
    
    // Performance settings
    frameRate: isMobile ? 20 : 30, // Lower FPS on mobile devices
    
    // Device info
    isMobile: isMobile,
    
    // Nhãn kết quả nhận diện / Detection label settings
    desktopLabelFontSize: 10,     // Kích thước chữ trên desktop (px)
    mobileLabelFontSize: 12,      // Kích thước chữ trên thiết bị di động (px)
    labelPadding: 4,              // Padding cho nhãn (px)
    labelMargin: 6,               // Khoảng cách từ nhãn đến khung (px)
    borderWidth: 2                // Độ dày viền khung (px)
};

export default config;