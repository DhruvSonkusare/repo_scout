## Your problem

> extract tables from scanned PDFs in Python

## Recommended repo
[pdf-extract-kit](https://github.com/opendatalab/pdf-extract-kit) is a comprehensive toolkit for high-quality PDF content extraction, including layout detection, formula recognition, OCR, and table parsing.

## Does it solve your problem?
**Fully**. The toolkit includes a dedicated table recognition module (`StructEqTable` and `PaddleOCR+TableMaster`) specifically designed to convert table images from PDFs into structured formats like LaTeX, HTML, or Markdown.

## Setup
1. Create and activate a Conda environment:
   ```bash
   conda create -n pdf-extract-kit-1.0 python=3.10 -y
   conda activate pdf-extract-kit-1.0
   ```
2. Install dependencies (choose based on your hardware):
   ```bash
   # For GPU devices
   pip install -r requirements.txt
   # For CPU devices
   pip install -r requirements-cpu.txt
   ```
3. Download the pre-trained model weights from [Hugging Face](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0) and place them in the `models/` directory as specified in the documentation.

## Step-by-step: solving YOUR problem with this repo
1. **Configure the Table Parsing Task**:
   Modify `configs/table_parsing.yaml` to point to your input PDF or image directory. The toolkit supports single files or entire directories.

2. **Run the Extraction Script**:
   Execute the provided script to process the tables. This script uses the models to detect table structures and recognize content.
   ```bash
   python scripts/table_parsing.py --config configs/table_parsing.yaml
   ```
   *Source: README.md*

3. **Retrieve Structured Output**:
   Check the `outputs/table_parsing` folder. The toolkit will output the recognized tables in formats like LaTeX, HTML, or Markdown depending on the model used (e.g., `StructTable-InternVL2-1B`).

4. **Integrate into Python Code**:
   If you need to run this programmatically, you can use the task runner logic found in the toolkit's scripts.
   ```python
   # Adapted from docs/zh_cn/_build/html/task_extend/code.html
   from pdf_extract_kit.utils.config_loader import load_config
   # You would initialize the table parsing task here using the registry
   # and call the process method on your PDF file.
   ```

## What you still need to build yourself
* **Post-processing**: While the tool extracts the table content, you may need to write custom logic to clean the resulting Markdown or LaTeX if you intend to insert it into a specific database or document format.
* **Reading Order**: If your tables are part of a larger document flow, you will need to manually handle the integration of table data back into the text, as the "Reading Order" model is currently listed as "Coming Soon" in the README.

## Gotchas and next steps
* **Model Selection**: For the best results with tables, ensure you are using the `StructTable-InternVL2-1B` model integrated in version 1.0, as it supports multiple output formats (LaTeX/HTML/Markdown).
* **OCR Dependencies**: Table recognition relies on OCR. Ensure your `configs/table_parsing.yaml` correctly points to the PaddleOCR model paths (e.g., `models/OCR/PaddleOCR/det/ch_PP-OCRv4_det`).
* **Conversion to Markdown**: If your primary goal is converting the *entire* PDF to Markdown (not just the tables), the maintainers recommend using their sister project [MinerU](https://github.com/opendatalab/MinerU), which is optimized for that specific workflow.
