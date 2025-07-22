const { Worker } = require('worker_threads');
const fs = require('fs');
const path = require('path');

async function runExperiment(experimentName) {
    console.log(`Running experiment: ${experimentName}`);
    const results = {};
    
    // Run the simple worker that processes all data at once
    const workerPath = path.join(__dirname, 'experiments', 'worker.js');
    const startTime = Date.now();
    
    const worker = new Worker(workerPath, {
        workerData: {
            modelPath: path.join(__dirname, 'experiment_results', experimentName, 'model.js'),
            wasmPath: path.join(__dirname, 'experiment_results', experimentName, 'model.wasm'),
            testDataPath: path.join(__dirname, 'models', 'test_data.bin')
        }
    });
    
    return new Promise((resolve, reject) => {
        worker.on('message', (message) => {
            if (message.type === 'result') {
                const endTime = Date.now();
                results.totalTime = endTime - startTime;
                results.throughput = message.numSamples / (results.totalTime / 1000);
                results.numSamples = message.numSamples;
                results.avgTimePerSample = results.totalTime / message.numSamples;
                console.log(`Completed: ${message.numSamples} samples in ${results.totalTime}ms`);
                console.log(`Throughput: ${results.throughput.toFixed(2)} samples/second`);
            }
        });
        
        worker.on('error', reject);
        worker.on('exit', (code) => {
            if (code !== 0) {
                reject(new Error(`Worker stopped with exit code ${code}`));
            } else {
                resolve(results);
            }
        });
    });
}

async function main() {
    const experimentName = process.argv[2];
    if (!experimentName) {
        console.error('Usage: node experiment_runner.js <experiment_name>');
        process.exit(1);
    }
    
    try {
        const results = await runExperiment(experimentName);
        
        // Save results
        const resultsPath = path.join(__dirname, 'experiment_results', experimentName, 'results.json');
        fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
        
        console.log(`Results saved to ${resultsPath}`);
    } catch (error) {
        console.error('Experiment failed:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}