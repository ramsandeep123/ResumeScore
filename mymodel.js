import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import padSequences from "./padsequence.js";
import "@tensorflow/tfjs-node";

const trainingData = JSON.parse(fs.readFileSync("training.json"));

const jobTitles = trainingData.map((entry) => entry.jobTitle);
const resumes = trainingData.map((entry) => entry.resume);
const scores = trainingData.map((entry) => entry.score);

const tokenizedJobTitles = jobTitles.map((title) =>
	title.toLowerCase().split(/\s+/)
);
const tokenizedResumes = resumes.map((resume) =>
	resume.toLowerCase().split(/\s+/)
);

const vocabulary = new Set([
	...tokenizedJobTitles.flat(),
	...tokenizedResumes.flat(),
]);

const jobTitleSequences = tokenizedJobTitles.map((tokens) =>
	tokens.map((token) => Array.from(vocabulary).indexOf(token))
);
const resumeSequences = tokenizedResumes.map((tokens) =>
	tokens.map((token) => Array.from(vocabulary).indexOf(token))
);

// Pad sequences to a fixed length
const paddedJobTitles = padSequences(jobTitleSequences, 100);
const paddedResumes = padSequences(resumeSequences, 100);

// Combine job title and resume sequences into a single input array
const xTrain = [paddedJobTitles, paddedResumes];

// Convert scores to binary labels (0 for scores below 90, 1 for scores equal to or above 90)
const labels = scores.map((score) => (score >= 90 ? 1 : 0));
const yTrain = tf.tensor1d(labels);

// Define the model architecture
const jobTitleInput = tf.input({ shape: [100] });
const resumeInput = tf.input({ shape: [100] });

const embeddingLayer = tf.layers.embedding({
	inputDim: vocabulary.size,
	outputDim: 16,
	inputLength: 100,
});

const jobTitleEmbedded = embeddingLayer.apply(jobTitleInput);
const resumeEmbedded = embeddingLayer.apply(resumeInput);

const concatenated = tf.layers
	.concatenate()
	.apply([jobTitleEmbedded, resumeEmbedded]);

const flattened = tf.layers.flatten().apply(concatenated);
const dense1 = tf.layers
	.dense({ units: 16, activation: "relu" })
	.apply(flattened);
const output = tf.layers
	.dense({ units: 1, activation: "sigmoid" })
	.apply(dense1);

const model = tf.model({
	inputs: [jobTitleInput, resumeInput],
	outputs: output,
});

// Compile the model
model.compile({
	optimizer: "adam",
	loss: "binaryCrossentropy",
	metrics: ["accuracy"],
});

// Train the model
model
	.fit(xTrain, yTrain, {
		epochs: 100,
		batchSize: 32,
		validationSplit: 0.2,
		callbacks: {
			onEpochEnd: async (epoch, logs) => {
				console.log(
					`Epoch ${epoch + 1}: val_loss=${logs.val_loss}, val_accuracy=${
						logs.val_accuracy
					}`
				);
				if (epoch === 99) {
					await model.save("file://./my-model");
					console.log("Model saved.");
				}
			},
		},
	})
	.then(() => {
		console.log("Training completed.");
	})
	.catch((err) => {
		console.error("Error:", err);
	});
