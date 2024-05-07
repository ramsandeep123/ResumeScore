import express from "express";
import multer from "multer";
import cors from "cors";
import editer from "text-from-pdf";
import * as tf from "@tensorflow/tfjs";
import fs from "fs";
import padSequences from "./padsequence.js";
import "@tensorflow/tfjs-node";

const app = express();
app.use(cors());
app.use(express.json());
const port = 3333;

const upload = multer({ dest: "uploads/" });

const model = await tf.loadLayersModel("file://./my-model/model.json");

// Load the training data
const trainingData = JSON.parse(fs.readFileSync("training.json"));

// Extract vocabulary from the training data
const jobTitles = trainingData.map((entry) => entry.jobTitle);
const resumes = trainingData.map((entry) => entry.resume);
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
console.log("Vocabulary size:", vocabulary.size);

function preprocessJobtitle(jobTitle) {
	const tokenizedText = jobTitle.toLowerCase().split(/\s+/);

	const numericalRepresentation = tokenizedText.map((token) => {
		const index = Array.from(vocabulary).indexOf(token);
		return index !== -1 ? index : 0;
	});

	const paddedSequence = padSequences([numericalRepresentation], 100);

	return paddedSequence;
}

function preprocessResume(resumeText) {
	const tokenizedText = resumeText.toLowerCase().split(/\s+|\n+/);

	const numericalRepresentation = tokenizedText.map((token) => {
		const index = Array.from(vocabulary).indexOf(token);
		return index !== -1 ? index : 0;
	});

	const paddedSequence = padSequences([numericalRepresentation], 100);

	return paddedSequence;
}

app.post("/upload", upload.single("resume"), async (req, res) => {
	const filePath = req.file.path;

	const resumeText = await editer.pdfToText(filePath);

	fs.unlink(filePath, () => {
		console.log("file deleted successfully");
	});

	const { jobTitle } = req.query;
	console.log(jobTitle);
	const resumeInput = preprocessResume(resumeText);
	const jobTitleInput = preprocessJobtitle(jobTitle);

	const prediction = model.predict([jobTitleInput, resumeInput]);

	const scorePrediction = prediction.dataSync()[0] * 100;
	res.status(200).send({ predictedScore: scorePrediction });
});

app.listen(port, () => {
	console.log(`Server is running on http://localhost:${port}`);
});
