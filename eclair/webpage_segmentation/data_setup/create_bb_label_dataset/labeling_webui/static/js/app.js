// Automatically load an image when the page is loaded
window.onload = loadNextImage;

document.getElementById('labelInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        submitLabel(this.value);
    }
});

document.getElementById('skipButton').addEventListener('click', function () {
    loadNextImage();
});

function submitLabel(label) {
    // Retrieve the selected value from the dropdown
    const selectedElement = document.getElementById('elementSelect').value;

    // Send the label and the selected element to the backend
    fetch('/submit_label', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
            label: label,
            elementType: selectedElement 
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Label and element type submitted:', data);
        loadNextImage();
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function loadNextImage() {
    fetch('/get_image')
    .then(response => response.json())
    .then(data => {
        console.log('Received image data:', data);
        // Call a function to render the image and bounding box
        drawBoundingBox(data.imagePath, data.boundingBox);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function drawBoundingBox(imagePath, boundingBox) {
    const canvas = document.getElementById('imageCanvas');
    const container = document.getElementById('canvasContainer');

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Scale coordinates and dimensions to fit the image size
        const x_center_scaled = boundingBox[1] * img.width;
        const y_center_scaled = boundingBox[2] * img.height;
        const width_scaled = boundingBox[3] * img.width;
        const height_scaled = boundingBox[4] * img.height;

        // Calculate top-left and bottom-right coordinates
        const top_left_x = x_center_scaled - (width_scaled / 2);
        const top_left_y = y_center_scaled - (height_scaled / 2);
        const bottom_right_x = x_center_scaled + (width_scaled / 2);
        const bottom_right_y = y_center_scaled + (height_scaled / 2);

        // Draw the rectangle
        ctx.strokeStyle = 'red';
        ctx.strokeRect(top_left_x, top_left_y, width_scaled, height_scaled);

        // Draw a green dot at the center
        ctx.fillStyle = 'green';
        ctx.beginPath();
        ctx.arc(x_center_scaled, y_center_scaled, 2, 0, 2 * Math.PI);
        ctx.fill();

        // Scroll the container to center the bounding box
        const bboxCenterX = (x_center_scaled + (width_scaled / 2));
        const bboxCenterY = (y_center_scaled + (height_scaled / 2));

        container.scrollLeft = bboxCenterX - container.offsetWidth / 2;
        container.scrollTop = bboxCenterY - container.offsetHeight / 2;
    };

    img.src = imagePath;
}

