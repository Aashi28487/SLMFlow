# Project Execution Walkthrough

This document explains what we are doing in simple terms so you can discuss it with your interviewer.

## What is SLMFlow?
SLMFlow is an orchestration framework for Small Language Models (SLMs). Instead of relying on one massive, expensive cloud AI (like ChatGPT), this project runs several small AI models locally on your PC. It schedules tasks intelligently based on your PC's available memory (6GB VRAM) and automatically routes questions to the most suitable model, reducing costs and keeping data private.

## What we are doing step-by-step:

### 1. Setting up the Environment
- **Creating a Virtual Environment:** We created an isolated space (`.venv`) for this project so the required Python libraries don't interfere with other projects on your computer.
- **Generating `requirements.txt`:** We extracted all necessary libraries from the project's configuration and put them into a list.
- **Installing Dependencies:** We installed everything needed to run the framework, including testing tools and Ollama connectors.

### 2. Testing the Core Logic
- **Running Mock Tests:** Before downloading heavy AI models, we ran tests using simulated "mock" models. This proves that the core routing and memory management logic works without needing the actual AI weights.

### 3. Hardware Probing
- **Environment Check:** We ran a command to detect your PC's CPU, RAM, and GPU (NVIDIA 6GB VRAM). The framework needs this to know exactly how much memory it can use when scheduling AI models.

### 4. Downloading and Profiling Real Models
- **Ollama Integration:** We used Ollama to download two real, small language models:
  - `qwen2.5:0.5b-instruct-q4_K_M` (a tiny 0.5 billion parameter model)
  - `llama3.2:3b-instruct-q4_K_M` (a larger 3 billion parameter model)
- **Benchmarking:** We tested how fast these models load into memory and how quickly they generate text on your specific hardware.

### 5. Running the Actual Benchmarks (Generating Scores)
- **Regret Benchmarks:** We tested how good our smart scheduler is compared to a naive "just load it when needed" approach.
- **Router Training:** We trained a small internal router to decide which of the two models to use for a given task.
- **Final Evaluation (Harness):** We ran the entire system end-to-end to generate accuracy scores, speed metrics, and memory usage numbers, automatically filling them into the `docs/RESULTS.md` file.
