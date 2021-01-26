/* eslint import/no-webpack-loader-syntax: off */

import "./App.css";
import { useCallback, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import * as tf from "@tensorflow/tfjs";
import InferenceWorker from "workerize-loader!./inference.worker";
import { serializeTensor, deserializeTensor } from "./utils";

const inferenceWorker = InferenceWorker();

function App() {
  const [inputSource, setInputSource] = useState("/cat.jpg");
  const inputRef = useRef(null);
  const outputRef = useRef(null);
  const [processing, setProcessing] = useState(false);

  const inferCallback = useCallback(() => {
    if (processing) {
      return;
    }
    setProcessing(true);
    (async () => {
      try {
        let input = tf.browser.fromPixels(inputRef.current);
        const originalSize = [input.shape[0], input.shape[1]];
        input = tf.image.resizeBilinear(input, [256, 256]);
        let output = deserializeTensor(
          await inferenceWorker.infer(serializeTensor(input))
        );
        output = tf.image.resizeBilinear(output, originalSize);
        tf.browser.toPixels(output, outputRef.current);
        setProcessing(false);
      } catch (err) {
        alert("An unknown error occured");
      }
    })();
  }, [inputRef, processing, setProcessing]);

  const onDrop = useCallback(
    (acceptedFiles) => {
      if (processing) {
        return;
      }
      const fr = new FileReader();
      fr.onload = () => {
        setInputSource(fr.result);
      };
      fr.readAsDataURL(acceptedFiles[0]);
    },
    [setInputSource, processing]
  );
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: "image/*",
  });

  return (
    <>
      <h1>
        Monocular Depth Estimation using{" "}
        <a href="https://arxiv.org/abs/1907.01341v3">MiDaS</a>
      </h1>
      <div className="App">
        <div
          className={[
            "image-dropzone",
            isDragActive
              ? "image-dropzone--highlight"
              : "image-dropzone--default",
          ].join(" ")}
          {...getRootProps()}
        >
          <h2>Choose an input image by dropping or clicking to select</h2>
          <input {...getInputProps()} />
          <img ref={inputRef} alt="" src={inputSource} />
        </div>
        <div className="output">
          <h2>Generated Depth Estimation</h2>
          <canvas ref={outputRef}></canvas>
        </div>
        <div className="controls">
          <button onClick={inferCallback} disabled={processing}>
            {processing ? "Inferring..." : "Estimate Depth"}
          </button>
        </div>
      </div>
    </>
  );
}

export default App;
