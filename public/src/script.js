// Import the pipeline and env from the Hugging Face transformers library
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

// Disable local models
env.allowLocalModels = false;

// Get references to the image upload input and image container elements
const imageUpload = document.getElementById("image-upload");
const imageContainer = document.getElementById("image-container");

// Initialize the object detection pipeline with a pre-trained model
const imgRec = await pipeline('object-detection', 'Xenova/detr-resnet-50');

// Add an event listener to handle image uploads
imageUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }
    const reader = new FileReader();

    // When the file is read, display the image and run object detection
    reader.onload = function (evnt) {
        imageContainer.innerHTML = ''; // Clear previous images
        const image = document.createElement('img');
        image.src = evnt.target.result;
        imageContainer.appendChild(image);
        detect(image); // Run object detection on the uploaded image
    };
    reader.readAsDataURL(file); // Read the file as a data URL
});

// Function to run object detection on the image
async function detect(img) {
    const output = await imgRec(img.src, {
        threshold: 0.5, // Set the detection threshold
        percentage: true, // Return results as percentages
    });
    output.forEach(renderBox); // Render bounding boxes for detected objects
}

// Function to render a bounding box around a detected object
function renderBox({ box, label }) {
    const { xmax, xmin, ymax, ymin } = box;

    // Generate a random color for the bounding box
    const color = '#' + Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, 0);

    // Create a div element for the bounding box
    const boxElement = document.createElement('div');
    boxElement.className = 'bounding-box';
    Object.assign(boxElement.style, {
        borderColor: color,
        left: 100 * xmin + '%',
        top: 100 * ymin + '%',
        width: 100 * (xmax - xmin) + '%',
        height: 100 * (ymax - ymin) + '%',
    });

    // Create a span element for the label
    const labelElement = document.createElement('span');
    labelElement.textContent = label;
    labelElement.className = 'bounding-box-label';
    labelElement.style.backgroundColor = color;

    // Append the label to the bounding box and the bounding box to the image container
    boxElement.appendChild(labelElement);
    imageContainer.appendChild(boxElement);
}