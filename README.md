# Adobe India Hackathon: Connecting the Dots - Round 1A Submission

<p align="center">
  <em>A Machine Learning-Powered Solution for Semantic PDF Structure Extraction</em>
</p>

---

## Overview

Welcome to my submission for Round 1A of the Adobe India Hackathon. The "Connecting the Dots" challenge asks us to rethink how we interact with documents, transforming static PDFs into intelligent, structured sources of knowledge.

This project tackles the foundational step of that vision: **making sense of a document's structure**. It provides a robust, offline, and high-performance system that ingests any PDF and, using a sophisticated machine learning pipeline, extracts its core semantic outline—the title and all hierarchical headings (H1, H2, H3).

The result is a clean, machine-readable JSON output that serves as the essential blueprint for any future document intelligence task, from semantic search to automated summarization.

---

## Table of Contents

- [The Challenge](#the-challenge)
- [Core Features](#core-features)
- [Architectural Deep Dive](#architectural-deep-dive)
  - [Why a Machine Learning Approach?](#why-a-machine-learning-approach)
  - [Feature Engineering Philosophy](#feature-engineering-philosophy)
  - [The Two-Stage Classification Pipeline](#the-two-stage-classification-pipeline)
  - [Post-Processing: Ensuring Logical Integrity](#post-processing-ensuring-logical-integrity)
- [Performance and Constraints](#performance-and-constraints)
- [Technology Stack](#technology-stack)
- [How to Build and Run the Solution](#how-to-build-and-run-the-solution)
- [Project Structure](#project-structure)
- [Future Work and Potential Improvements](#future-work-and-potential-improvements)

---

## The Challenge

The mission for Round 1A is to build a containerized solution that can automatically process any PDF (up to 50 pages), identify its title and headings, and output a structured JSON file. This entire process must be completed offline, on a CPU, and within strict time and size limits.

## Core Features

- **Automated Title Extraction:** Identifies the document title using a heuristic based on the most prominent text on the first page.
- **Hierarchical Heading Detection:** Accurately classifies text lines as H1, H2, or H3 headings.
- **Structured JSON Output:** Generates a clean, predictable JSON file for each processed PDF.
- **Fully Offline & Containerized:** Packaged in a lightweight Docker image with no external network dependencies.
- **High Performance:** Optimized to meet the strict execution time and model size constraints of the hackathon.

---

## Architectural Deep Dive

### Why a Machine Learning Approach?

While simple rule-based systems (e.g., "the biggest font is H1") are fast, they are brittle and fail on the vast diversity of PDF layouts. A corporate report, a scientific paper, and a legal document all use different styling conventions. A machine learning model, trained on a variety of features, can learn these nuanced patterns and generalize far more effectively.

### Feature Engineering Philosophy

The model's accuracy is built on a foundation of rich, descriptive features extracted by **`PyMuPDF`**. Each line of text is converted into a vector of attributes that capture its visual and textual context:

- **Font-Based Features (`font_size`, `bold`):** These are the most obvious indicators of importance, but are used as just one signal among many.
- **Positional Features (`indentation`, `v_spacing`, `centered`):** These features capture the document's visual layout. Headings are often distinguished by their unique indentation or the empty space that surrounds them.
- **Textual Features (`len`, `upper_ratio`, `ends_colon`):** These analyze the content of the text itself. Headings are often short, written in uppercase, or end with a colon.
- **Contextual Features (`font_rank`):** This powerful feature normalizes font size on a per-page basis, allowing the model to understand relative importance even in documents with unconventional font choices.

### The Two-Stage Classification Pipeline

To maximize precision, I designed a two-stage classification pipeline using **`scikit-learn`**:

1.  **Stage 1: Heading Identification (Binary Classifier)**
    - A `RandomForestClassifier` acts as a high-precision filter. Its sole job is to answer the question: "Is this line a heading of *any* kind?" This removes the vast majority of body text from consideration, allowing the next stage to focus only on relevant candidates.

2.  **Stage 2: Level Determination (Multi-class Classifier)**
    - A second `RandomForestClassifier` takes the lines identified by Stage 1 and performs the fine-grained task of assigning a specific level: `H1`, `H2`, or `H3`. This division of labor makes the overall model more accurate and robust.

### Post-Processing: Ensuring Logical Integrity

A machine learning model can still make occasional errors. To guarantee a logically sound output, a final post-processing step sorts all predicted headings by their location in the document and enforces a consistent hierarchy. For example, if the model predicts an H3 immediately following an H1, this step intelligently corrects the H3 to an H2, preserving the document's natural flow.

---

## Performance and Constraints

This solution was built from the ground up to meet the hackathon's strict constraints:
- **Execution Time:** Well under the 10-second limit for a 50-page PDF.
- **Model Size:** The serialized `joblib` model is under 200MB.
- **Runtime Environment:** Runs entirely on CPU with no GPU dependencies.
- **Offline Operation:** The container is self-contained and requires no network access.

## Technology Stack

- **Core Libraries:** `PyMuPDF`, `scikit-learn`, `pandas`, `numpy`
- **Model Persistence:** `joblib`
- **Containerization:** `Docker`

---

## How to Build and Run the Solution

The entire solution is containerized with Docker for easy and consistent execution.

### Prerequisites

- Docker must be installed and the Docker daemon must be running.

### 1. Build the Docker Image

Navigate to the project's root directory in your terminal and run the following command. The `.` at the end is important as it specifies the build context.

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

### 2. Run the Container

Place your test PDFs into a local folder named `input`. The container will automatically process them and place the results in a local `output` folder.

**On Linux or macOS:**

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

**On Windows PowerShell:**

```powershell
docker run --rm -v "${pwd}/input:/app/input" -v "${pwd}/output:/app/output" --network none mysolutionname:somerandomidentifier
```

---

## Project Structure

The project is organized to be clean, modular, and easy to understand.

```
/
├── app/
│   ├── models/
│   │   └── head_clf_rf_tuned.joblib  # Pre-trained and serialized model pipeline
│   ├── process_pdfs.py             # Main Python script with all logic
│   └── requirements.txt            # List of Python dependencies
│
├── input/                          # Mount point for input PDFs
├── output/                         # Mount point for output JSON files
│
├── Dockerfile                      # Instructions to build the Docker container
└── README.md                       # This file
```

---

## Future Work and Potential Improvements

- **Deeper Hierarchy:** Extend the model to recognize H4, H5, and H6 headings for more granular outlines.
- **Enhanced Multilingual Support:** While the current feature set is language-agnostic, the model could be explicitly trained on labeled data from non-English documents (e.g., Japanese, Hindi) to earn the multilingual bonus.
- **Advanced Title Heuristics:** Improve title detection by analyzing the document's metadata or by using a model to differentiate between a title and a cover page heading.
- **Table of Contents (ToC) Parsing:** Explicitly parse the ToC when available to create a highly accurate "ground truth" outline for the rest of the document.
