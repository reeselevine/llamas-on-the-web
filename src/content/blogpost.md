---
title: Introducing WebGPU support for llama.cpp
byline: April 2026
---

After nearly a year of work, we're officially introducing WebGPU support for llama.cpp. With this release, users can now run almost any open-weight model in their browser, _with GPU acceleration_, making fast and local LLM inference more accessible and available than it ever has been before.

Skeptical about whether browser-based inference will work for you? Go ahead and try it out below:

:::demo

## Why Use Local Models in the Browser?

You might be impresssed or underwhelmed by the demo above, depending on your experience using the powerful LLMs served by big companies like OpenAI or Anthropic. However, the capabilities of smaller language models are [improving rapidly](https://hazyresearch.stanford.edu/blog/2025-11-11-ipw) and they may play a large part in the [future of agentic AI](https://research.nvidia.com/labs/lpr/slm-agents/). Additionally, the [increasing power demands of AI](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report) motivates a turn towards more energy-efficient solutions (such as taking advantage of local computation). And all that is before you even consider how the companies serving large models over the network are [using your information](https://hai.stanford.edu/news/be-careful-what-you-tell-your-ai-chatbot), and to what degree you are comfortable allowing these companies access to your AI conversations, and corresponding private data.

That being said, local inference, and especially inference in the browser, is not going to solve all these problems immediately. WebGPU implementations are still maturing, and we ran into issues in the WebGPU implementations of every major browser ([1](https://github.com/gfx-rs/wgpu/issues/8896), [2](https://bugs.webkit.org/show_bug.cgi?id=311598), [3](https://issues.chromium.org/issues/504928024), [4](https://github.com/gpuweb/gpuweb/issues/5315)) while implementing WebGPU support for llama.cpp. Local models, especially in the sizes supported by mobile browsers, are not as capable as users might want, and browsers' same-origin policy means that if a model has been downloaded on one site, it will need to be downloaded again to be used on a different site. Nevertheless, we're still early, and the hope is that implementations like this one lead to demos and use-cases that can help influence the direction of local inference and browser support.

### Disclaimer: Walking in the Footsteps of Giants

Before we go any further into the llama.cpp x WebGPU integration, we'd first like to acknowledge that we are not the first LLM inference runtime for WebGPU. [ONNX Runtime](https://onnxruntime.ai/) was one of the first machine learning frameworks to add web support, and its newer WebGPU backend helps power [Transformers.js](https://huggingface.co/docs/transformers.js/index). [WebLLM](https://webllm.mlc.ai/) brings the power of [TVM](https://tvm.apache.org/) to the browser, efficiently compiling models into deployable formats. In addition, llama.cpp can already be run through the browser, albeit without WebGPU support, using the existing [Wllama](https://github.com/ngxson/wllama) package. We are grateful to everyone involved in these efforts, as well as many others that we probably missed, for providing comparisons and starting points. We're looking forward to seeing the ecosystem around WebGPU and browser-based machine learning continue to evolve.

## Llama.cpp x WebGPU

:::rewrite
Now let's get into the exciting parts! First, as you've already seen in the demo above, you can now accelerate almost any model supported by llama.cpp using WebGPU, if your device and browser supports it. If you're reading this post, you're likely already aware that llama.cpp is already a popular choice for running local models on consumer hardware. With WebGPU support, that choice becomes even easier, as developers can integrate llama.cpp directly into their websites so that even completely non-technical users can benefit from fast, efficient, and private local models. 
:::

If you were entertained by the above demo, feel free to copy the below and share it; we'd love some help sharing this fun project!

We also want to note that despite the name, the WebGPU backend is not constrained to the browser. If you are developing a native application that needs wide GPU support (especially if it isn't covered by other llama.cpp [backends](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src), the WebGPU backend works natively using Google's [Dawn](https://github.com/google/dawn) implementation, directly in C++.

### Functionality

As of today, the WebGPU backend supports most recent transformer-based models and llama.cpp quantization types, including unquantized 32-bit and 16-bit float types, traditional quantization types like q8_0, and all k-quants. While we also technically support i-quants, their performance has not been optimized and you may run into stability issues using them in the browser, especially on more limited devices.

Given that WebGPU is still new, there are not mature, portable, and performant libraries for linear algebra routines in WebGPU, such as NVIDIA's [cuBLAS](https://developer.nvidia.com/cublas). Thus, we wrote the WebGPU backend in llama.cpp from the ground up, developing custom WGSL shaders for matrix multiplication, FlashAttention, and all the other important machine learning operations. In fact, since the WebGPU backend lives within the [ggml](https://ggml.ai/) tensor library, you can actually use it independently of llama.cpp as well! However, we expect that most people will want to use it in the context of llama.cpp, which highlights one of our major advantages—llama.cpp's ecosystem.

Nowadays, when a company releases a new open-weight model, it almost always has its architecture baked into llama.cpp on day one. Additionally, thanks to the great work of companies like [Unsloth](https://unsloth.ai/) and the rest of the community, there are numerous model quantizations which allow users to pick the variant with the best capabilities and size for any given hardware. The WebGPU backend inherits this support automatically, meaning that you should be able to deploy new models as soon as they become available.

While you're welcome to roll your own browser-frontend for the WebGPU llama.cpp backend, we have also integrated it into Wllama, which is available with WebGPU support as an [NPM package](https://www.npmjs.com/package/@reeselevine/wllama-webgpu). Thanks to the power of Emscripten and WebAssembly, we are able to run llama.cpp directly in the browser. Wllama runs llama.cpp within a Web Worker so that inference never blocks the UI-thread, and includes features like automatic model caching. Due to browser limitations, we currently only support models up to 4GB in size, and models over 2GB need to be split into smaller shards prior to usage.

### Performance

Of course, we are also very focused on the performance of the WebGPU backend. And since seeing is believing, we invite you to run a micro-benchmark yourself below, which compares CPU and GPU performance on your machine:

:::benchmark

Hopefully, that benchmark convinced you that WebGPU is completely worth it. And if it didn't, or the CPU ended up being faster, then please reach out and let us know; we're always looking to improve our performance on every system.

Behind the scenes, we've been working hard to optimize the performance of the major bottlenecks in inference, including matrix and matrix-vector multiplication. One of the most exciting and challenging parts about WebGPU is finding shaders and parameters that work well across diverse GPUs, from high-end desktop setups to the smallest mobile phones. We have some very interesting ongoing work in this domain, so stay tuned for more information on it!

<TODO: information on sharing your results if you run it? is there a community database that people can contribute to that Abhijit is working on?>

Additionally, we've been focusing on optimizing the WebGPU llama.cpp runtime as well, optimizing our scheduling to avoid CPU-GPU synchronization and finding the right batching and submission parameters to maximize GPU utilization and maintain stability on many systems. We've found that every major browser's WebGPU implementation is at different stages and has its own quirks, and we're looking forward to seeing how these implementations evolve and continue to become more performant and stable.

### Future Work and Technical Report

Obviously there is a ton of juicy technical details that we didn't want to weigh down this release blog post with. If you're interested, we've released a technical report [here](TODO), which discusses more about kernel designs and gives performance numbers across different browsers, operating systems, models, etc. 

With this initial release we're just getting started. We're already starting to see more contributions to the WebGPU backend in llama.cpp, from improving the performance and portability of different operations, to support for multi-modal vision-language models, to initial kernel fusion implementations. We will also continue to work with browser vendors to find bugs in their implementations and find ways to increase our performance across many devices. While right now WebGPU support is in a fork of the original Wllama library, we will continue to work on integrating them into a single package. Within Wllama, we're currently working on techniques to reduce wasted memory usage, allowing smaller devices to run more powerful models, and are interested in investigating WebAssembly's 64-bit memory indexing to support even larger models on some devices.

<TODO: Didn't the above get fixed?>

If you're interested in helping out or have ideas to help improve the llama.cpp WebGPU backend, please reach out or start a discussion in the [llama.cpp](https://github.com/ggml-org/llama.cpp) repository.

### More Demos

We're interested in seeing how people start using browser-based inference. For a more full-featured chat interface using Wllama, visit [this demo](https://reeselevine.github.io/wllama/). And if you make your own, feel free to let us know so we can add it here!

<TODO: Did Nikhil have a demo to show?>

### Acknowledgements

This release wouldn't be possible without the help of many talented people. [Reese Levine](https://reeselevine.github.io) leads the implementation of the WebGPU llama.cpp work as part of a group at UC Santa Cruz working under [Tyler Sorensen](https://tyler-utah.github.io/) that includes [Rithik Sharma](https://sharmarithik.github.io/rithiksharma/), [Zheyuan Chen](https://arbersephirotheca.github.io/), [Nikhil Jain](https://nikhiljain17.github.io/), [Abhijit Ramesh](https://abhijitramesh.me/), [Neha Abbas](https://www.linkedin.com/in/neha-a-827406275/), and [James Contini](https://jamescontini.com/).

Several outside contributors have also made signficant contributions to the WebGPU llama.cpp backend, and we want to especially thank [@yomaytk](https://github.com/yomaytk) and [@Constannnnnt](https://github.com/Constannnnnt) for their work. We'd also like to thank [Georgi Gerganov](https://ggerganov.com/), [@CISC](https://github.com/cisc), [Xuan-Son Nguyen](https://ngxson.com/), and the other llama.cpp maintainers for their support of this effort and for making this integration possible.

### A Note on the Model Used in This Post

The default model used here is Gemma-3, specifically a 270M parameter Unsloth Q4\_K\_M quantized version. The reason we chose this model is that it is small enough to be loaded on many mobile phones and downloads fairly quickly. However, we had to tailor the demos here to things that come across reasonably well with such a small model. If you looked at the advanced usage section in the model loader, you might have noticed that if you were on a larger machine, you could use a Q4\_K version of Qwen 3.5 2B, or choose any model from a Hugging Face repository. Generally, larger and more recent models will feel more capable, at the expense of longer download times and less portability.
