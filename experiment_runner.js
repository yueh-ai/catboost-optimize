#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { Worker } = require('worker_threads');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

// Parse command line arguments
const argv = yargs(hideBin(process.argv))
    .option('wrapper', {
        alias: 'w',
        type: 'string',
        description: 'Path to wrapper C++ file',
        default: './experiments/batch_wrapper.cpp'
    })
    .option('model', {
        alias: 'm',
        type: 'string',
        description: 'Path to model C++ file (if not included in wrapper)',
        default: null
    })
    .option('emflags', {
        alias: 'e',
        type: 'string',
        description: 'Emscripten compiler flags',
        default: '-O3'
    })
    .option('batch-sizes', {
        alias: 'b',
        type: 'array',
        description: 'Batch sizes to test',
        default: [1, 10, 100, 1000, 5000, 10000]
    })
    .option('test-data', {
        alias: 'd',
        type: 'string',
        description: 'Path to test data file',
        default: './test_data/test_data_1M.bin'
    })
    .option('output-dir', {
        alias: 'o',
        type: 'string',
        description: 'Output directory for results',
        default: './experiment_results'
    })
    .option('experiment-name', {
        alias: 'n',
        type: 'string',
        description: 'Name of the experiment',
        default: `exp_${Date.now()}`
    })
    .option('simd', {
        type: 'boolean',
        description: 'Enable SIMD optimizations',
        default: false
    })
    .option('threads', {
        type: 'boolean',
        description: 'Enable threading support',
        default: false
    })
    .option('use-batch-api', {
        type: 'boolean',
        description: 'Use batch prediction API if available',
        default: true
    })
    .help()
    .argv;

class ExperimentRunner {
    constructor(config) {
        this.config = config;
        this.results = {
            experimentName: config.experimentName,
            timestamp: new Date().toISOString(),
            config: config,
            batchResults: []
        };
        
        // Create output directory if it doesn't exist
        if (!fs.existsSync(config.outputDir)) {
            fs.mkdirSync(config.outputDir, { recursive: true });
        }
    }
    
    compileWASM() {
        console.log('Compiling WASM module...');
        console.log(`Wrapper: ${this.config.wrapper}`);
        console.log(`Emscripten flags: ${this.config.emflags}`);
        
        const outputDir = path.join(this.config.outputDir, this.config.experimentName);
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        const wasmPath = path.join(outputDir, 'model.js');
        
        // Build emscripten command
        let emFlags = this.config.emflags;
        
        // Add SIMD if requested
        if (this.config.simd) {
            emFlags += ' -msimd128';
        }
        
        // Add threading if requested
        if (this.config.threads) {
            emFlags += ' -pthread -s PTHREAD_POOL_SIZE=4';
        }
        
        // Build the compile command
        const exportedFunctions = [
            '_catboostPredict',
            '_catboostPredictBatch',
            '_catboostPredictBatchOptimized',
            '_getMaxBatchSize',
            '_malloc',
            '_free'
        ];
        
        const compileCmd = `emcc ${this.config.wrapper} ` +
            `-o ${wasmPath} ` +
            `${emFlags} ` +
            `-s EXPORTED_FUNCTIONS='[${exportedFunctions.join(',')}]' ` +
            `-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPF64","HEAP32","HEAPU8"]' ` +
            `-s ALLOW_MEMORY_GROWTH=1 ` +
            `-s MODULARIZE=1 ` +
            `-s EXPORT_NAME='CatBoostModule'`;
        
        console.log(`Compile command: ${compileCmd}`);
        
        try {
            execSync(compileCmd, { stdio: 'inherit' });
            console.log('Compilation successful!');
            return wasmPath;
        } catch (error) {
            console.error('Compilation failed:', error.message);
            throw error;
        }
    }
    
    async loadTestData() {
        console.log(`Loading test data from ${this.config.testData}...`);
        
        if (!fs.existsSync(this.config.testData)) {
            throw new Error(`Test data file not found: ${this.config.testData}`);
        }
        
        const buffer = fs.readFileSync(this.config.testData);
        const dataView = new DataView(buffer.buffer);
        
        // Read header
        const numSamples = dataView.getInt32(0, true);
        const numFeatures = dataView.getInt32(4, true);
        
        console.log(`Loaded ${numSamples} samples with ${numFeatures} features`);
        
        // Extract features and labels
        const features = [];
        const groundTruth = [];
        
        let offset = 8;
        for (let i = 0; i < numSamples; i++) {
            const sampleFeatures = [];
            for (let j = 0; j < numFeatures; j++) {
                sampleFeatures.push(dataView.getFloat32(offset, true));
                offset += 4;
            }
            features.push(sampleFeatures);
            groundTruth.push(dataView.getFloat64(offset, true));
            offset += 8;
        }
        
        return { features, groundTruth, numSamples, numFeatures };
    }
    
    createWorkerScript(wasmPath, batchSize, useBatchAPI) {
        const absoluteWasmPath = path.resolve(wasmPath);
        const workerCode = `
const { parentPort } = require('worker_threads');
const fs = require('fs');
const path = require('path');
const CatBoostModule = require('${absoluteWasmPath}');

let moduleInstance;
let testData;

parentPort.on('message', async (msg) => {
    if (msg.type === 'init') {
        try {
            const wasmFile = '${absoluteWasmPath}'.replace('.js', '.wasm');
            
            moduleInstance = await CatBoostModule({
                wasmBinary: fs.readFileSync(wasmFile),
                locateFile: (filename) => {
                    if (filename.endsWith('.wasm')) {
                        return wasmFile;
                    }
                    return path.join(path.dirname('${absoluteWasmPath}'), filename);
                }
            });
            
            // Wait a moment for heap to be fully initialized
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Check if HEAP arrays are available
            if (!moduleInstance.HEAPF32 || !moduleInstance.HEAPF64) {
                throw new Error('HEAP arrays not initialized. Available properties: ' + Object.keys(moduleInstance).join(', '));
            }
            
            testData = msg.testData;
            parentPort.postMessage({ type: 'ready' });
        } catch (error) {
            parentPort.postMessage({ type: 'error', error: error.message });
        }
    } else if (msg.type === 'run') {
        try {
            const { features, numSamples, numFeatures } = testData;
            const batchSize = ${batchSize};
            const useBatchAPI = ${useBatchAPI};
            const predictions = new Float64Array(numSamples);
            
            const startTime = Date.now();
            let processed = 0;
            
            if (useBatchAPI && moduleInstance._catboostPredictBatch) {
                // Use batch API
                const maxBatchSize = moduleInstance._getMaxBatchSize ? 
                    moduleInstance._getMaxBatchSize() : batchSize;
                const actualBatchSize = Math.min(batchSize, maxBatchSize);
                
                // Allocate memory for batch
                const featuresPtr = moduleInstance._malloc(4 * numFeatures * actualBatchSize);
                const predictionsPtr = moduleInstance._malloc(8 * actualBatchSize);
                
                while (processed < numSamples) {
                    const currentBatchSize = Math.min(actualBatchSize, numSamples - processed);
                    
                    // Copy features to WASM memory
                    const heapF32 = moduleInstance.HEAPF32;
                    const ptrOffset = featuresPtr >> 2;
                    
                    for (let i = 0; i < currentBatchSize; i++) {
                        const sampleFeatures = features[processed + i];
                        for (let j = 0; j < numFeatures; j++) {
                            heapF32[ptrOffset + i * numFeatures + j] = sampleFeatures[j];
                        }
                    }
                    
                    // Make batch prediction
                    moduleInstance._catboostPredictBatch(
                        featuresPtr,
                        predictionsPtr,
                        currentBatchSize,
                        numFeatures
                    );
                    
                    // Copy predictions back
                    const heapF64 = moduleInstance.HEAPF64;
                    const predPtrOffset = predictionsPtr >> 3;
                    
                    for (let i = 0; i < currentBatchSize; i++) {
                        predictions[processed + i] = heapF64[predPtrOffset + i];
                    }
                    
                    processed += currentBatchSize;
                    
                    if (processed % 10000 === 0) {
                        parentPort.postMessage({
                            type: 'progress',
                            processed,
                            total: numSamples
                        });
                    }
                }
                
                moduleInstance._free(featuresPtr);
                moduleInstance._free(predictionsPtr);
                
            } else {
                // Use single prediction API
                const featuresPtr = moduleInstance._malloc(4 * numFeatures);
                
                for (let i = 0; i < numSamples; i++) {
                    const sampleFeatures = features[i];
                    
                    // Copy features to WASM memory
                    const heapF32 = moduleInstance.HEAPF32;
                    const ptrOffset = featuresPtr >> 2;
                    
                    for (let j = 0; j < numFeatures; j++) {
                        heapF32[ptrOffset + j] = sampleFeatures[j];
                    }
                    
                    // Make prediction
                    predictions[i] = moduleInstance._catboostPredict(featuresPtr, numFeatures);
                    
                    if ((i + 1) % 10000 === 0) {
                        parentPort.postMessage({
                            type: 'progress',
                            processed: i + 1,
                            total: numSamples
                        });
                    }
                }
                
                moduleInstance._free(featuresPtr);
            }
            
            const endTime = Date.now();
            const totalTime = (endTime - startTime) / 1000;
            
            parentPort.postMessage({
                type: 'complete',
                predictions: Array.from(predictions),
                totalTime,
                predictionsPerSecond: numSamples / totalTime
            });
            
        } catch (error) {
            parentPort.postMessage({ type: 'error', error: error.message });
        }
    }
});
`;
        
        const workerPath = path.join(this.config.outputDir, this.config.experimentName, `worker_${batchSize}.js`);
        fs.writeFileSync(workerPath, workerCode);
        return workerPath;
    }
    
    async runBatchExperiment(wasmPath, batchSize, testData) {
        console.log(`\nTesting batch size: ${batchSize}`);
        
        const workerPath = this.createWorkerScript(wasmPath, batchSize, this.config.useBatchApi);
        
        return new Promise((resolve, reject) => {
            const worker = new Worker('./' + workerPath);
            
            worker.on('message', (msg) => {
                switch (msg.type) {
                    case 'ready':
                        worker.postMessage({ type: 'run' });
                        break;
                    case 'progress':
                        process.stdout.write(`\rProgress: ${msg.processed}/${msg.total}`);
                        break;
                    case 'complete':
                        process.stdout.write('\n');
                        
                        // Calculate accuracy
                        let correctPredictions = 0;
                        const tolerance = 0.01;
                        
                        for (let i = 0; i < testData.numSamples; i++) {
                            if (Math.abs(msg.predictions[i] - testData.groundTruth[i]) < tolerance) {
                                correctPredictions++;
                            }
                        }
                        
                        const accuracy = correctPredictions / testData.numSamples;
                        
                        const result = {
                            batchSize,
                            totalTime: msg.totalTime,
                            predictionsPerSecond: msg.predictionsPerSecond,
                            accuracy,
                            correctPredictions,
                            totalPredictions: testData.numSamples
                        };
                        
                        console.log(`Results for batch size ${batchSize}:`);
                        console.log(`  Time: ${msg.totalTime.toFixed(2)}s`);
                        console.log(`  Speed: ${msg.predictionsPerSecond.toFixed(0)} predictions/sec`);
                        console.log(`  Accuracy: ${(accuracy * 100).toFixed(2)}%`);
                        
                        worker.terminate();
                        resolve(result);
                        break;
                    case 'error':
                        console.error(`Worker error: ${msg.error}`);
                        worker.terminate();
                        reject(new Error(msg.error));
                        break;
                }
            });
            
            worker.on('error', (error) => {
                console.error('Worker error:', error);
                reject(error);
            });
            
            // Initialize worker with test data
            worker.postMessage({
                type: 'init',
                testData: {
                    features: testData.features,
                    numSamples: testData.numSamples,
                    numFeatures: testData.numFeatures
                }
            });
        });
    }
    
    async run() {
        console.log(`Starting experiment: ${this.config.experimentName}`);
        console.log('Configuration:', JSON.stringify(this.config, null, 2));
        
        try {
            // Compile WASM
            const wasmPath = this.compileWASM();
            this.results.wasmPath = wasmPath;
            
            // Load test data
            const testData = await this.loadTestData();
            this.results.testDataStats = {
                numSamples: testData.numSamples,
                numFeatures: testData.numFeatures
            };
            
            // Run experiments for each batch size
            for (const batchSize of this.config.batchSizes) {
                try {
                    const result = await this.runBatchExperiment(wasmPath, batchSize, testData);
                    this.results.batchResults.push(result);
                } catch (error) {
                    console.error(`Failed to run experiment for batch size ${batchSize}:`, error);
                    this.results.batchResults.push({
                        batchSize,
                        error: error.message
                    });
                }
            }
            
            // Find optimal batch size
            const validResults = this.results.batchResults.filter(r => !r.error);
            if (validResults.length > 0) {
                const optimal = validResults.reduce((best, current) => 
                    current.predictionsPerSecond > best.predictionsPerSecond ? current : best
                );
                this.results.optimal = {
                    batchSize: optimal.batchSize,
                    predictionsPerSecond: optimal.predictionsPerSecond,
                    speedupVsBaseline: optimal.predictionsPerSecond / validResults[0].predictionsPerSecond
                };
            }
            
            // Save results
            const resultsPath = path.join(this.config.outputDir, this.config.experimentName, 'results.json');
            fs.writeFileSync(resultsPath, JSON.stringify(this.results, null, 2));
            
            console.log('\nExperiment complete!');
            console.log(`Results saved to: ${resultsPath}`);
            
            if (this.results.optimal) {
                console.log(`\nOptimal configuration:`);
                console.log(`  Batch size: ${this.results.optimal.batchSize}`);
                console.log(`  Speed: ${this.results.optimal.predictionsPerSecond.toFixed(0)} predictions/sec`);
                console.log(`  Speedup: ${this.results.optimal.speedupVsBaseline.toFixed(2)}x`);
            }
            
        } catch (error) {
            console.error('Experiment failed:', error);
            this.results.error = error.message;
            
            const resultsPath = path.join(this.config.outputDir, this.config.experimentName, 'error.json');
            fs.writeFileSync(resultsPath, JSON.stringify(this.results, null, 2));
        }
    }
}

// Run the experiment
const runner = new ExperimentRunner({
    wrapper: argv.wrapper,
    model: argv.model,
    emflags: argv.emflags,
    batchSizes: argv.batchSizes,
    testData: argv.testData,
    outputDir: argv.outputDir,
    experimentName: argv.experimentName,
    simd: argv.simd,
    threads: argv.threads,
    useBatchApi: argv.useBatchApi
});

runner.run().catch(console.error);