import { render, show } from "@tensorflow/tfjs-vis";
import {
  sequential,
  layers,
  tidy,
  util,
  tensor2d,
  train,
  losses,
  linspace,
} from "@tensorflow/tfjs";
import type { Sequential, Tensor, Rank } from "@tensorflow/tfjs";

type CarData = {
  Miles_per_Gallon: number;
  Horsepower: number;
};

type TransformedData = {
  mpg: number;
  horsepower: number;
};

document.addEventListener("DOMContentLoaded", run);

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData(): Promise<TransformedData[]> {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );

  const carsData = await carsDataResponse.json();

  const cleaned = (carsData as CarData[])
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg !== null && car.horsepower !== null);

  return cleaned;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();

  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  render.scatterplot(
    {
      name: "Housepower v MPG",
    },
    {
      values,
    },
    {
      xLabel: "Housepower",
      yLabel: "MPG",
      height: 300,
    }
  );

  // Create the model
  const model = createModel();

  show.modelSummary(
    {
      name: "Model Summary",
    },
    model
  );

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log("Done Training");

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
}

function createModel() {
  // Create a sequential model
  const model = sequential();

  // Add a single input layer
  model.add(layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Add an output layer
  model.add(layers.dense({ units: 1, useBias: true }));

  return model;
}

function convertToTensor(data: TransformedData[]) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tidy(() => {
    // Step 1. Shuffle the data
    util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(
  model: Sequential,
  inputs: ReturnType<typeof convertToTensor>["inputs"],
  labels: ReturnType<typeof convertToTensor>["labels"]
) {
  // Prepare the model for training.
  model.compile({
    optimizer: train.adam(),
    loss: losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function testModel(
  model: Sequential,
  inputData: TransformedData[],
  normalizationData: ReturnType<typeof convertToTensor>
) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tidy(() => {
    const xs = linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = (preds as Tensor<Rank>)
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => ({
    x: val,
    y: preds[i],
  }));

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
}
