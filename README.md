Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.

# Project Progress and How to Execute

## What Has Been Completed
- Data cleaning script (`starter/data/clean_data.py`) to preprocess census data.
- Model training script (`starter/starter/train_model.py`) that trains a RandomForestClassifier and saves model artifacts to `starter/model/`.
- Unit tests for model code in `starter/starter/ml/test_model.py` (run with `pytest`).
- Model slice metrics function and script (`starter/starter/ml/model.py` and `starter/compute_slice_metrics.py`) that output per-slice performance to `slice_output.txt`.
- Model card (`starter/model_card.md`) completed with all required sections and actual metrics.
- FastAPI app (`starter/main.py`) for inference, with Pydantic models and type hints.
- API unit tests in `starter/test_main.py`.

## How to Run Everything (from repo root)

1. **Set up the Conda environment**
   ```zsh
   conda env create -f starter/environment.yml
   conda activate fastapi
   ```

2. **Prepare the data**
   - Download `census.csv` to `starter/data/` if not already present.
   - Clean the data:
     ```zsh
     python starter/data/clean_data.py
     ```

3. **Train the model**
   ```zsh
   python starter/starter/train_model.py
   ```
   This will save the model and encoders to `starter/model/`.

4. **Run model unit tests**
   ```zsh
   pytest starter/starter/ml/test_model.py
   ```

5. **Run API unit tests**
   ```zsh
   pytest starter/test_main.py
   ```

6. **Compute and view slice metrics**
   ```zsh
   python starter/compute_slice_metrics.py
   cat slice_output.txt
   ```

7. **Start the FastAPI app**
   ```zsh
   uvicorn starter.main:app --reload
   ```
   - Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive API docs.

8. **Example API usage**
   - Use the Swagger UI or cURL to test the `/predict` endpoint.

9. **Model Card**
   - See `starter/model_card.md` for a full description of the model, data, metrics, and caveats.
