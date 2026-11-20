
import numpy as np
import onnxruntime as ort


class SileroOnnxModel:

    def __init__(self, model_path):

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.state = np.zeros(
            (2, 1, 128),
            dtype=np.float32,
        )

        self.context = np.zeros(
            (1, 64),
            dtype=np.float32,
        )

        self.sr = np.array(
            [16000],
            dtype=np.int64,
        )

        print(
            "ONNX providers:",
            self.session.get_providers(),
            flush=True,
        )

    def reset_states(self):

        self.state.fill(0)
        self.context.fill(0)

    def __call__(self, audio):

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(1, -1)

        if audio.shape[1] != 512:
            raise ValueError(
                "Silero requires exactly 512 samples at 16 kHz"
            )

        # Silero requires 64 samples of context
        # from the previous audio chunk.

        model_input = np.concatenate(
            (self.context, audio),
            axis=1,
        )

        outputs = self.session.run(
            ["output", "stateN"],
            {
                "input": model_input,
                "state": self.state,
                "sr": self.sr,
            },
        )

        probability, new_state = outputs

        self.state = new_state

        self.context = model_input[:, -64:]

        return float(probability[0][0])


class VADIterator:

    def __init__(
        self,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    ):

        if sampling_rate != 16000:
            raise ValueError(
                "Only 16000 Hz is supported"
            )

        self.model = model
        self.sampling_rate = sampling_rate
        self.threshold = threshold

        self.neg_threshold = max(
            threshold - 0.15,
            0.01,
        )

        self.min_silence_samples = (
            sampling_rate
            * min_silence_duration_ms
            // 1000
        )

        self.speech_pad_samples = (
            sampling_rate
            * speech_pad_ms
            // 1000
        )

        self.reset_states()

    def reset_states(self):

        self.model.reset_states()

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(
        self,
        audio,
        return_seconds=False,
    ):

        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)

        window_size_samples = len(audio)

        if window_size_samples != 512:
            raise ValueError(
                "Expected 512 samples per chunk"
            )

        self.current_sample += window_size_samples

        speech_prob = self.model(audio)

        # Speech resumed.
        if (
            speech_prob >= self.threshold
            and self.temp_end
        ):
            self.temp_end = 0

        # Speech started.
        if (
            speech_prob >= self.threshold
            and not self.triggered
        ):

            self.triggered = True

            speech_start = max(
                0,
                self.current_sample
                - self.speech_pad_samples
                - window_size_samples,
            )

            if return_seconds:
                speech_start = round(
                    speech_start / self.sampling_rate,
                    1,
                )

            return {
                "start": int(speech_start)
                if not return_seconds
                else speech_start
            }

        # Silence detected.
        if (
            speech_prob < self.neg_threshold
            and self.triggered
        ):

            if not self.temp_end:
                self.temp_end = self.current_sample

            if (
                self.current_sample - self.temp_end
                < self.min_silence_samples
            ):
                return None

            speech_end = (
                self.temp_end
                + self.speech_pad_samples
                - window_size_samples
            )

            self.temp_end = 0
            self.triggered = False

            if return_seconds:
                speech_end = round(
                    speech_end / self.sampling_rate,
                    1,
                )

            return {
                "end": int(speech_end)
                if not return_seconds
                else speech_end
            }

        return None

