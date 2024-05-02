import * as tf from "@tensorflow/tfjs";
export const PAD_INDEX = 0;
export const OOV_INDEX = 2;

export default function padSequences(
	sequences,
	maxLen,
	padding = "pre",
	truncating = "pre",
	value = PAD_INDEX
) {
	const paddedSequences = sequences.map((seq) => {
		if (seq.length > maxLen) {
			if (truncating === "pre") {
				seq.splice(0, seq.length - maxLen);
			} else {
				seq.splice(maxLen, seq.length - maxLen);
			}
		}

		if (seq.length < maxLen) {
			const pad = new Array(maxLen - seq.length).fill(value);
			if (padding === "pre") {
				seq = pad.concat(seq);
			} else {
				seq = seq.concat(pad);
			}
		}

		return seq;
	});

	return tf.tensor2d(paddedSequences);
}
