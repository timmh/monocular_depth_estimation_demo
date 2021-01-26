import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import { serializeTensor, deserializeTensor } from "./utils";

let model;
const modelLoaded = loadGraphModel("midas_u8/model.json")
  .then((loaded_model) => {
    model = loaded_model;
  })
  .catch((err) => {
    alert("Failed to load model");
  });

export async function infer(input) {
  input = deserializeTensor(input);
  await modelLoaded;
  input = tf.div(input, 255);
  input = tf.transpose(input, [2, 0, 1]);
  input = tf.expandDims(input);

  let output = await model.executeAsync(input);

  output = tf.transpose(output, [1, 2, 0]);
  output = tf.div(
    tf.sub(output, tf.min(output)),
    tf.sub(tf.max(output), tf.min(output))
  );
  return serializeTensor(output);
}
