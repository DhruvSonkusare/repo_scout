## Your problem

> extract tables from scanned PDFs in Python

## Recommended repo
[opendatalab/pdf-extract-kit](https://github.com/opendatalab/pdf-extract-kit) is a modular toolkit for high-quality PDF content extraction, including layout detection, OCR, and table parsing.

## Does it solve your problem?
**Partly**. The toolkit includes a `table_parsing` task that handles table recognition from images. While it uses `PyMuPDF` (fitz) to convert PDF pages into images in `pdf_extract_kit/utils/data_preprocess.py`, the `TableParsingTask` class in the provided code only implements a `predict` method that calls `load_images` (which only supports `.png`, `.jpg`, and `.jpeg`). The code for `TableParsingTask` does not currently call `load_pdf_images`, so you would need to manually bridge the PDF loading to the table parsing model.

## Setup
1. Create and activate a Python 3.10 environment:
   ```bash
   conda create -n pdf-extract-kit-1.0 python=3.10 -y
   conda activate pdf-extract-kit-1.0
   ```
2. Install dependencies for GPU (recommended for table parsing models):
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The existence of `requirements-cpu.txt` is not confirmed by the source code.*

## Step-by-step: solving YOUR problem with this repo
1. **Configure the task**: Edit `configs/table_parsing.yaml` to point to your data. The default config uses `table_parsing_struct_eqtable` with `output_format: latex`.
2. **Load the PDF as images**: The system provides a utility in `pdf_extract_kit/utils/data_preprocess.py` to rasterize PDF pages. Note that if a page exceeds 3000px in width or height, it downscales to a 1:1 matrix.
   ```python
   import fitz
   from PIL import Image

   def load_pdf(pdf_path, dpi=144):
       images = []
       doc = fitz.open(pdf_path)
       for i in range(len(doc)):
           page = doc[i]
           # Logic from load_pdf_page in data_preprocess.py
           pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
           image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
           if pix.width > 3000 or pix.height > 3000:
               pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
               image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
           images.append(image)
       return images
   ```
3. **Initialize the Table Parsing task**: Use the config loader to set up the model defined in your YAML.
   ```python
   from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models

   config = load_config('configs/table_parsing.yaml')
   task_instances = initialize_tasks_and_models(config)
   table_task = task_instances['table_parsing']
   ```
4. **Run extraction**: Since `TableParsingTask.predict` only accepts image paths in the current code, you must pass image files or modify the task to use `load_pdf_images`.
   ```python
   input_data = config.get('inputs') 
   result_path = config.get('outputs', 'outputs/table_parsing')
   
   # The code shows TableParsingTask uses predict(), not process()
   detection_results = table_task.predict(input_data, result_path)
   ```

## What you still need to build yourself
* **PDF Integration for Tables**: Unlike the OCR task which has a `process` method handling PDFs, `TableParsingTask` only calls `load_images` (image files only). You will need to call `load_pdf_images` from `BaseTask` yourself and pass the resulting PIL images to the model.
* **Post-processing logic**: You will need to write the logic to integrate the model's output (e.g., LaTeX) into your application.
* **Reading Order**: The code does not show a reading order implementation; you may need to use coordinates from other tasks like `layout_detection` to sequence content.

## Gotchas and next steps
* **DPI Settings**: The default DPI in `load_pdf` is 144. You can adjust this in `pdf_extract_kit/utils/data_preprocess.py` to improve quality for scanned documents.
* **Model Config**: The `configs/table_parsing.yaml` shows the model `table_parsing_struct_eqtable` is used, with a `max_new_tokens` limit of 1024.
* **Input Restrictions**: The `load_images` and `load_pdf_images` methods in `BaseTask` will raise a `ValueError` if the input directory contains nested directories.
