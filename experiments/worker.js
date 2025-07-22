const { parentPort, workerData } = require('worker_threads');
const fs = require('fs');

async function runWorker() {
    const { modelPath, wasmPath, testDataPath } = workerData;
    
    // Load the model module
    const Module = require(modelPath);
    const wasmBinary = fs.readFileSync(wasmPath);
    
    // Initialize the module - it returns a promise
    console.log('Initializing WASM module...');
    const moduleInstance = await Module({ 
        wasmBinary,
        onRuntimeInitialized: () => {
            console.log('Runtime initialized');
        }
    });
    
    // Wait a bit to ensure full initialization
    await new Promise(resolve => setTimeout(resolve, 100));
    
    console.log('Module initialized:', typeof moduleInstance);
    
    // Load test data
    const testDataBuffer = fs.readFileSync(testDataPath);
    const testDataView = new DataView(testDataBuffer.buffer);
    
    // Read header
    const magic = testDataView.getUint32(0, true);
    const version = testDataView.getUint32(4, true);
    const numSamples = testDataView.getUint32(8, true);
    const numFeatures = testDataView.getUint32(12, true);
    
    // Verify magic number
    if (magic !== 0xCAFEBABE) {
        throw new Error(`Invalid magic number: 0x${magic.toString(16)}`);
    }
    
    console.log(`Processing ${numSamples} samples with ${numFeatures} features`);
    
    // In the data format:
    // - 6 float features: carat, depth, table, x, y, z
    // - 3 categorical features (encoded as floats): cut, color, clarity
    const numFloatFeatures = 6;
    const numCatFeatures = 3;
    
    // Allocate memory for all inputs and outputs
    const bytesPerSample = numFeatures * 4; // All features stored as float32
    const totalInputBytes = numSamples * bytesPerSample;
    const totalOutputBytes = numSamples * 8; // double per prediction
    
    const inputPtr = moduleInstance._malloc(totalInputBytes);
    const outputPtr = moduleInstance._malloc(totalOutputBytes);
    
    if (!inputPtr || !outputPtr) {
        throw new Error('Failed to allocate memory');
    }
    
    console.log(`Allocated memory: input=${inputPtr}, output=${outputPtr}, totalBytes=${totalInputBytes}`);
    
    try {
        // Copy all input data to WASM memory
        let dataOffset = 16; // Skip header (4 * 4 bytes)
        
        // Copy data using setValue if direct heap access isn't available
        console.log('Copying input data to WASM memory...');
        
        // First, let's try to access HEAPF32
        const HEAPF32 = moduleInstance.HEAPF32;
        
        if (HEAPF32) {
            // Direct heap access
            for (let i = 0; i < numSamples; i++) {
                const sampleOffset = inputPtr + i * bytesPerSample;
                
                // Copy all features (they're all stored as float32 in the file)
                for (let j = 0; j < numFeatures; j++) {
                    const value = testDataView.getFloat32(dataOffset, true);
                    const heapIndex = (sampleOffset >> 2) + j;
                    HEAPF32[heapIndex] = value;
                    dataOffset += 4;
                }
            }
        } else {
            // Alternative: use Module.setValue if available
            console.log('Direct heap access not available, trying alternative method...');
            
            // Create a temporary buffer and copy all at once
            const inputBuffer = new Float32Array(numSamples * numFeatures);
            for (let i = 0; i < numSamples * numFeatures; i++) {
                inputBuffer[i] = testDataView.getFloat32(dataOffset, true);
                dataOffset += 4;
            }
            
            // Copy to WASM memory
            const bytesPerFloat = 4;
            for (let i = 0; i < inputBuffer.length; i++) {
                const ptr = inputPtr + i * bytesPerFloat;
                // Write float directly as bytes
                const view = new DataView(new ArrayBuffer(4));
                view.setFloat32(0, inputBuffer[i], true);
                for (let j = 0; j < 4; j++) {
                    moduleInstance.HEAPU8[ptr + j] = view.getUint8(j);
                }
            }
        }
        
        // Process all data at once
        console.log('Starting prediction for all samples...');
        const startTime = performance.now();
        
        moduleInstance._catboostPredictAll(
            inputPtr,
            outputPtr,
            numSamples,
            numFloatFeatures,
            numCatFeatures
        );
        
        const endTime = performance.now();
        console.log(`Prediction completed in ${(endTime - startTime).toFixed(2)}ms`);
        
        // Calculate throughput
        const totalTime = endTime - startTime;
        const throughput = numSamples / (totalTime / 1000);
        
        console.log(`Throughput: ${throughput.toFixed(2)} samples/second`);
        
        // Send results
        parentPort.postMessage({
            type: 'result',
            numSamples: numSamples,
            predictionTime: totalTime
        });
        
    } finally {
        // Clean up
        moduleInstance._free(inputPtr);
        moduleInstance._free(outputPtr);
    }
}

runWorker().catch(err => {
    console.error('Worker error:', err);
    process.exit(1);
});