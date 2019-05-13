const NUM_FRAMES = 2;
// each frame is 23ms of audio containing 232 numbers that correspond to different frequencies [note: 232 frequency buckets are needed to capture the human voice]
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];


let model;
let recognizer;
// examples is where all the data for collect() is stored, contains label and vals
// label = 0 or 1 for "voice" or "music" respectively
// vals = 696 numbers holding the frequency information (the spectorgram)
let examples = [];


// associates a label with the output of recognizer.listen()
// normalizes the raw spectrogram and drops all but the last NUM_FRAMES frames
function collect(label) {
    if (recognizer.isListening()) {
        return recognizer.stopListening();
    }
    if (label == null) {
        return;
    }
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
        // since we want to use short sounds instead of words to control the slider we are only taking the last 3 frames (~70ms) into consideration
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        examples.push({vals, label});
        document.querySelector('#console').textContent = `${examples.length} examples collected`;
    }, {
        overlapFactor: 0.999,
        // since includeSpectorgram is set to true, recognizer.listen() will gibe the raw spectrogram (which is frequency data) for 1 sec of audio divided into 43 frames, so each frame is ~23ms of audio
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
    });
}


// to avoid numerical issues we normalize the data to have an average of 0 and a standard deviation of 1
function normalize(x) {
    // spectrogram values are large negative numbers around -100
    const mean = -100
    // spectrogram values have a deviation of 10
    const std = 10;
    return x.map(x => (x - mean) / std);
}


// trains the model using the collected data
async function train() {
    toggleButtons(false);
    const ys = tf.oneHot(examples.map(e => e.label), 2);
    const xsShape = [examples.length, ...INPUT_SHAPE];
    const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

    // The training runs over the data 10 times using a batch size of 16 and displays the current accuracy in the UI
    await model.fit(xs, ys, {
        batchSize: 16, 
        epochs: 10, 
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.querySelector('#console').textContent = `Accuracy: ${(logs.acc * 100).toFixed(1)} % Epoch: ${epoch + 1}`;
            }
        }
    });
    tf.dispose([xs, ys]);
    toggleButtons(true);
}


// defines the model architecture
// this model has four layers
function buildModel() {
    model = tf.sequential();
    // layer one is a convolutional layer that processes the audio data (represented as a spectrogram)
    model.add(tf.layers.depthwiseConv2d({
        depthMulitplier: 8,
        kernelSize: [NUM_FRAMES, 2],
        activation: 'relu',
        inputShape: INPUT_SHAPE
    }));
    // layer two is a max pool layer
    model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
    // layer three is a flatten layer
    model.add(tf.layers.flatten());
    // layer four is a dense layer that maps to the two actions
    model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
    // compiling our model to get it ready for training
    // the adam optimizer is a "Method for Stochatic Optimization" - stochatic is just a fancy word for random
    const optimizer = tf.train.adam(0.01);
    model.compile({
        optimizer,
        // standard loss function for classification
        // measures how far the predicted probabilities (one probability per class) are from having 100% probability in the true class and 0% for all the other classes
        loss: 'categoricalCrossentropy',
        // accuracy will give us the percentage of examples the model gets correct after each epoch of training
        metrics: ['accuracy']
    });
}


function toggleButtons(enable) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}


function flatten(tensors) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr, i) => result.set(arr, i * size));
    return result;
}


// decreases the value of the slider if the label is 0 "voice", increases if the label is 1 "music"
async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    document.getElementById('console').textContent = label;
    let delta = 0.1;
    const prevValue = +document.getElementById('output').value;
    document.getElementById('output').value = prevValue + (label === 0 ? -delta : delta);
}


// uses the microphone and makes real time predictions
function listen() {
    if (recognizer.isListening()) {
        recognizer.stopListening();
        toggleButtons(true);
        document.getElementById('listen').textContent = 'Listen';
        return;
    }
    toggleButtons(false);
    document.getElementById('listen').textContent = 'Stop';
    document.getElementById('listen').disabled = false;

    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
        const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
        // call the trained model to get prediction
        // output of this will be a Tensor in the shape [1, numClasses]
        // ^ represents a probibility distribution over the number of classes - basically a set of confidences for each of the possible output classes which sum to 1
        // the Tensor has an outer dimension of 1 because this is the size of the batch
        const probs = model.predict(input);
        // returns the class index with the highest probability
        // pass "1" as the axis parameter because we want to compute the argMax over the last dimension which is numClasses
        const predLabel = probs.argMax(1);
        await moveSlider(predLabel);
        // to clean up the GPU memory it's important to manually call dispose()
        tf.dispose([input, probs, predLabel]);
    }, {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
    });
}


// this is not being called for now -- this is the most basic functionality, you call this, say words and the model guesses
// this function does not require the use of the buttons in the html
function predictWord() {
    // Array of words that the recognizer is trained to recognize
    const words = recognizer.wordLabels();
    recognizer.listen(({scores}) => {
        // Turn scores into a list of (score,word) pairs
        scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
        // Find the most probable word
        scores.sort((s1, s2) => s2.score - s1.score);
        document.querySelector('#console').textContent = scores[0].word;
    }, {probabilityThreshold: 0.75});
}

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    // predictWord() no longer called, but available if wanted
    buildModel();
}

app();