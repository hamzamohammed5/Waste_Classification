let isPredicting = false;
var prediction_time;

const MODEL_URL = 'assets/model.json';
const class_map = {
                    0: "Organic",
                    1: "Recyclable"
                    }
class L2 {
  static className = 'L2';
  constructor(config) {
     return tf.regularizers.l1l2(config)
  }}
tf.serialization.registerClass(L2);

async function loadMyModel() {
  MyModel = await tf.loadLayersModel(MODEL_URL);
  return MyModel;
}

async function loadImage() {
  // Get the file input and output image element
  var fileInput = document.getElementById('fileInput');

  // Check if a file is selected
  if (fileInput.files.length > 0) {
      // Get the selected file
      var selectedFile = fileInput.files[0];

      // Create a FileReader to read the file
      var reader = new FileReader();

      // Set a callback function to be called when the file is loaded
      reader.onload = async function (e) {
          // Create an image element
          var imageElement = new Image();

          // Set the src attribute of the image element with the file data URL
          imageElement.src = e.target.result;

          // Set additional attributes if needed
          imageElement.alt = "Loaded Image";

          // Load the image into a tensor
          const tensor = await loadImageToTensor(imageElement);

          // Predict
          predict(tensor);
          console.log(tensor);
      };

      // Read the selected file as a data URL
      reader.readAsDataURL(selectedFile);
  } else {
      alert("Please select an image file.");
  }
}

async function loadImageToTensor(imageElement) {
  // Create a canvas element
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  // Draw the image onto the canvas
  context.drawImage(imageElement, 0, 0, imageElement.width, imageElement.height);

  // Get the image data from the canvas
  const imageData = context.getImageData(0, 0, 224, 224);

  // Convert the image data to a tensor
  let tensor = tf.browser.fromPixels(imageData);
  tensor= tensor.expandDims(0);
  tensor= tensor.toFloat().div(tf.scalar(255));
  return tensor;
}

// Visualize the image
document.getElementById('fileInput').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const imageDataUrl = e.target.result;
      document.getElementById('previewImage').src = imageDataUrl;
    };
    reader.readAsDataURL(file); // Load the image as a data URL
  }
});


async function predict(tensor) {

    const predictedClass = tf.tidy(() => {
      var startTime = performance.now();
      const predictions = MyModel.predict(tensor);
      var endTime = performance.now();
      prediction_time = endTime - startTime
      return predictions
    });
    const class_indx = (await predictedClass.as1D().argMax().data());
    const score = ((await predictedClass.max().dataSync()[0])*100).toFixed(2);
    console.log(class_indx)
	document.getElementById("class").innerText = 'Class: ' + class_map[class_indx]; 
	document.getElementById("score").innerText ="Score: "+ score+"%"; 
  document.getElementById("prediction_time").innerText ="Prediction Time: "+ Math.round(prediction_time)+' ms'; 
    predictedClass.dispose();
    await tf.nextFrame();
  
}


function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	const MyModel = await loadMyModel();
}



init();