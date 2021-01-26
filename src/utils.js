import * as tf from "@tensorflow/tfjs";

export const serializeTensor = (tensor) => ({
  data: tensor.dataSync(),
  shape: tensor.shape,
});
export const deserializeTensor = ({ data, shape }) => tf.tensor(data, shape);
